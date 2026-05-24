from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .lifecycle import VLLMLifecycleManager
from .manifest import load_manifest
from .matrix import expand_manifest
from .mst_adapter import MSTSearchAdapter
from .resources import GPULeaseManager, PortAllocator
from .scheduler import OrchestratorScheduler
from .state_store import RunStateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local_orchestrator.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--manifest", type=Path, required=True)
    dry_run.add_argument("--run-id", default=None)
    dry_run.set_defaults(handler=_dry_run_command)

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--run-id", default=None)
    run.set_defaults(handler=_run_command)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument(
        "--force",
        action="store_true",
        help="rerun all jobs, including previously succeeded ones",
    )
    resume.set_defaults(handler=_resume_command)

    status = subparsers.add_parser("status")
    _add_run_root_arg(status)
    status.set_defaults(handler=_status_command)

    progress = subparsers.add_parser("progress")
    _add_run_root_arg(progress)
    progress.set_defaults(handler=_status_command)

    energy_run = subparsers.add_parser("energy-run")
    energy_run.add_argument("--plan", type=Path, required=True)
    energy_run.add_argument("--run-id", default=None)
    energy_run.set_defaults(handler=_energy_run_command)

    energy_resume = subparsers.add_parser("energy-resume")
    energy_resume.add_argument("--run-root", type=Path, required=True)
    energy_resume.add_argument(
        "--force",
        action="store_true",
        help="rerun all energy jobs, including previously succeeded ones",
    )
    energy_resume.set_defaults(handler=_energy_resume_command)

    energy_status = subparsers.add_parser("energy-status")
    energy_status.add_argument("--run-root", type=Path, required=True)
    energy_status.set_defaults(handler=_energy_status_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _dry_run_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run_id = args.run_id or manifest.run.run_id or "dry-run"
    mst_output_root = _resolve_mst_output_root(manifest=manifest, run_id=run_id)
    jobs = expand_manifest(manifest, mst_output_root=mst_output_root)
    payload = {
        "manifest": str(manifest.manifest_path),
        "run_id": run_id,
        "mst_output_root": str(mst_output_root),
        "job_count": len(jobs),
        "max_active_gpus": manifest.run.max_active_gpus,
        "allowed_gpu_ids": list(manifest.run.allowed_gpu_ids),
        "jobs": [
            {
                "experiment_id": job.experiment_id,
                "model": job.model,
                "workload": str(job.workload),
                "endpoint": job.endpoint,
                "hardware": job.hardware.name,
                "gpu_count": job.launch.gpu_count,
                "tensor_parallel_size": job.launch.tensor_parallel_size,
                "max_model_len": job.launch.max_model_len,
                "max_request_rate": job.search.max_request_rate,
                "max_binary_steps": job.search.max_binary_steps,
                "open_loop_bracket_growth_factor": job.search.open_loop_bracket_growth_factor,
                "ttft_slo_ms": job.search.ttft_slo_ms,
                "tpot_slo_ms": job.search.tpot_slo_ms,
                "ttft_slo_mode": job.search.ttft_slo_mode,
                "longbench_ttft_static_preset": job.search.longbench_ttft_static_preset,
                "probe": None if job.probe is None else job.probe.to_payload(),
                "result_dir": str(job.result_dir),
                "server_signature_key": job.server_signature_key,
                "server_config_slug": job.server_config_slug,
            }
            for job in jobs
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run_id = args.run_id or manifest.run.run_id or _default_run_id()
    run_root = (manifest.run.output_root / run_id).resolve()
    mst_output_root = _resolve_mst_output_root(manifest=manifest, run_id=run_id)
    jobs = expand_manifest(manifest, mst_output_root=mst_output_root)

    state_store = RunStateStore(run_root)
    state = state_store.initialize_new(
        run_id=run_id,
        manifest_path=manifest.manifest_path,
        jobs=jobs,
        mst_output_root=mst_output_root,
    )
    scheduler = _build_scheduler(state_store=state_store, manifest=manifest)
    summary = scheduler.run(jobs=jobs, state=state, resume=False, force=False)

    print(json.dumps({"run_root": str(run_root), "summary": summary}, sort_keys=True))
    return 0


def _resume_command(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    state_store = RunStateStore(run_root)
    state = state_store.load()

    manifest_path = Path(str(state["manifest_path"]))
    manifest = load_manifest(manifest_path)
    mst_output_root = _mst_output_root_from_state(state)
    jobs = expand_manifest(manifest, mst_output_root=mst_output_root)

    state_job_ids = {str(job["experiment_id"]) for job in state.get("jobs", [])}
    manifest_job_ids = {job.experiment_id for job in jobs}
    if manifest_job_ids != state_job_ids:
        missing = sorted(manifest_job_ids - state_job_ids)
        extra = sorted(state_job_ids - manifest_job_ids)
        raise ValueError(
            "resume manifest/job mismatch: "
            f"missing_in_state={missing}, extra_in_state={extra}"
        )

    scheduler = _build_scheduler(state_store=state_store, manifest=manifest)
    summary = scheduler.run(jobs=jobs, state=state, resume=True, force=bool(args.force))
    print(json.dumps({"run_root": str(run_root), "summary": summary}, sort_keys=True))
    return 0


def _status_command(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    state_store = RunStateStore(run_root)
    state = state_store.load()
    summary = state_store.summarize(state)
    print(json.dumps({"run_root": str(run_root), "summary": summary}, sort_keys=True))
    return 0


def _energy_run_command(args: argparse.Namespace) -> int:
    from energy_profiler.local_scheduler import (
        EnergyProfilingAdapter,
        SchedulerEnergyStateStore,
        default_local_energy_run_id,
        expand_energy_plan_for_local_scheduler,
        local_energy_run_root,
        run_config_from_energy_plan,
    )
    from energy_profiler.planning import load_energy_plan
    from energy_profiler.reporting import EnergyRunStateStore

    plan = load_energy_plan(args.plan)
    run_id = args.run_id or default_local_energy_run_id()
    run_root = local_energy_run_root(plan, run_id)
    jobs = expand_energy_plan_for_local_scheduler(plan, run_root=run_root)
    state_store = SchedulerEnergyStateStore(EnergyRunStateStore(run_root), plan=plan)
    state = state_store.initialize_new(plan_path=args.plan)
    scheduler = _build_energy_scheduler(
        run_config=run_config_from_energy_plan(plan, run_id=run_id),
        state_store=state_store,
        adapter=EnergyProfilingAdapter(plan=plan),
    )
    summary = scheduler.run(jobs=jobs, state=state, resume=False, force=False)
    print(json.dumps({"run_root": str(run_root), "summary": summary}, sort_keys=True))
    return 0


def _energy_resume_command(args: argparse.Namespace) -> int:
    from energy_profiler.local_scheduler import (
        EnergyProfilingAdapter,
        SchedulerEnergyStateStore,
        expand_energy_plan_for_local_scheduler,
        run_config_from_energy_plan,
    )
    from energy_profiler.planning import load_energy_plan
    from energy_profiler.reporting import EnergyRunStateStore

    run_root = args.run_root.resolve()
    delegate = EnergyRunStateStore(run_root)
    state = delegate.load()
    plan = load_energy_plan(delegate.plan_copy_path)
    run_id = run_root.name
    jobs = expand_energy_plan_for_local_scheduler(plan, run_root=run_root)
    state_store = SchedulerEnergyStateStore(delegate, plan=plan)
    state_job_ids = {str(job["job_id"]) for job in state.get("jobs", [])}
    plan_job_ids = {job.experiment_id for job in jobs}
    if plan_job_ids != state_job_ids:
        missing = sorted(plan_job_ids - state_job_ids)
        extra = sorted(state_job_ids - plan_job_ids)
        raise ValueError(
            "resume energy plan/job mismatch: "
            f"missing_in_state={missing}, extra_in_state={extra}"
        )
    scheduler = _build_energy_scheduler(
        run_config=run_config_from_energy_plan(plan, run_id=run_id),
        state_store=state_store,
        adapter=EnergyProfilingAdapter(plan=plan),
    )
    summary = scheduler.run(jobs=jobs, state=state, resume=True, force=bool(args.force))
    print(json.dumps({"run_root": str(run_root), "summary": summary}, sort_keys=True))
    return 0


def _energy_status_command(args: argparse.Namespace) -> int:
    from energy_profiler.reporting import EnergyRunStateStore

    run_root = args.run_root.resolve()
    state_store = EnergyRunStateStore(run_root)
    state = state_store.load()
    state_store.reconcile_jobs(state)
    summary = state_store.write_summary_files(state)
    print(json.dumps({"run_root": str(run_root), "summary": summary}, sort_keys=True))
    return 0


def _add_run_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", "--runroot", dest="run_root", type=Path, required=True)


def _build_scheduler(*, state_store: RunStateStore, manifest) -> OrchestratorScheduler:
    gpu_manager = GPULeaseManager(
        allowed_gpu_ids=manifest.run.allowed_gpu_ids,
        max_active_gpus=manifest.run.max_active_gpus,
    )
    port_allocator = PortAllocator(
        base_port_start=manifest.run.base_port_start,
        base_port_end=manifest.run.base_port_end,
        metrics_port_offset=manifest.run.metrics_port_offset,
    )
    lifecycle = VLLMLifecycleManager()
    adapter = MSTSearchAdapter(
        python_executable=manifest.run.python_executable,
    )
    return OrchestratorScheduler(
        run_config=manifest.run,
        gpu_manager=gpu_manager,
        port_allocator=port_allocator,
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
        lifecycle_factory=VLLMLifecycleManager,
    )


def _build_energy_scheduler(*, run_config, state_store, adapter) -> OrchestratorScheduler:
    gpu_manager = GPULeaseManager(
        allowed_gpu_ids=run_config.allowed_gpu_ids,
        max_active_gpus=run_config.max_active_gpus,
    )
    port_allocator = PortAllocator(
        base_port_start=run_config.base_port_start,
        base_port_end=run_config.base_port_end,
        metrics_port_offset=run_config.metrics_port_offset,
    )
    lifecycle = VLLMLifecycleManager()
    return OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=gpu_manager,
        port_allocator=port_allocator,
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
        lifecycle_factory=VLLMLifecycleManager,
    )


def _default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"orchestrator-{ts}"


def _resolve_mst_output_root(*, manifest, run_id: str) -> Path:
    if manifest.run.mst_output_root is not None:
        return manifest.run.mst_output_root.resolve()
    return (manifest.run.output_root.parent / "mst" / run_id).resolve()


def _mst_output_root_from_state(state: dict[str, Any]) -> Path | None:
    raw = state.get("mst_output_root")
    if isinstance(raw, str) and raw:
        return Path(raw).resolve()
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
