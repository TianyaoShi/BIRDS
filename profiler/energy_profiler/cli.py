from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .executor import EnergyExecutor, EnergyExecutorConfig
from .models import (
    EnergyPlanExecution,
    EnergyPlanMode,
    EnergyPlanRounding,
    EnergyPlanSelection,
    EnergyPlanSelectionSweep,
)
from .planning import (
    generate_plan_from_orchestrator,
    generate_plan_from_orchestrator_runs,
    load_energy_plan,
    load_selection_overrides,
    render_dry_run,
    write_energy_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="energy_profiler.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    _add_plan_generation_args(plan)
    plan.set_defaults(handler=_plan_command, forced_mode=None)

    plan_from_orchestrator = subparsers.add_parser("plan-from-orchestrator")
    _add_plan_generation_args(plan_from_orchestrator)
    plan_from_orchestrator.set_defaults(handler=_plan_command, forced_mode=None)

    plan_explicit = subparsers.add_parser("plan-explicit")
    _add_plan_generation_args(plan_explicit)
    plan_explicit.set_defaults(handler=_plan_command, forced_mode="explicit")

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--plan", type=Path, required=True)
    dry_run.set_defaults(handler=_dry_run_command)

    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    _add_executor_args(run)
    run.set_defaults(handler=_run_command)

    live_trial = subparsers.add_parser("run-live-trial")
    live_trial.add_argument("--trial-id", required=True)
    live_trial.add_argument("--output-dir", type=Path, required=True)
    live_trial.add_argument("--workload", type=Path, required=True)
    live_trial.add_argument("--model", required=True)
    live_trial.add_argument("--base-url", required=True)
    live_trial.add_argument("--endpoint", default="/v1/chat/completions")
    live_trial.add_argument("--metrics-url", default=None)
    live_trial.add_argument("--gpu-ids", nargs="+", type=int, default=(0,))
    live_trial.add_argument("--duration-s", type=float, default=90.0)
    live_trial.add_argument("--request-rate", type=float, default=1.0)
    live_trial.add_argument("--request-timeout-s", type=float, default=300.0)
    live_trial.add_argument("--metrics-interval-s", type=float, default=1.0)
    live_trial.add_argument("--window-s", type=float, default=10.0)
    live_trial.add_argument("--idle-monitor-duration-s", type=float, default=10.0)
    live_trial.add_argument("--traffic-warmup-s", type=float, default=0.0)
    live_trial.add_argument("--repeats", type=int, default=1)
    live_trial.add_argument("--repeat-cooldown-s", type=float, default=0.0)
    live_trial.add_argument("--warmup-each-repeat", action="store_true")
    live_trial.add_argument("--gpu-monitor-interval-s", type=float, default=1.0)
    live_trial.add_argument("--gpu-monitor-truncate-s", type=float, default=0.0)
    live_trial.add_argument("--monitor-clock", action="store_true")
    live_trial.add_argument("--safety-max-outstanding", type=int, default=None)
    live_trial.add_argument("--force", action="store_true")
    live_trial.set_defaults(handler=_run_live_trial_command)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument("--force", action="store_true")
    _add_executor_args(resume)
    resume.set_defaults(handler=_resume_command)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.set_defaults(handler=_status_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _plan_command(args: argparse.Namespace) -> int:
    mode: EnergyPlanMode = args.forced_mode or args.mode
    selection = _load_and_merge_selection(args)
    rounding = _parse_rounding_policy(args.rounding_policy)
    plan = generate_plan_from_orchestrator(
        orchestrator_run_root=args.orchestrator_run_root[0],
        output_plan=args.output_plan,
        rate_source=args.rate_source,
        mode=mode,
        selection=selection,
        rounding=rounding,
    ) if len(args.orchestrator_run_root) == 1 else generate_plan_from_orchestrator_runs(
        orchestrator_run_roots=tuple(args.orchestrator_run_root),
        output_plan=args.output_plan,
        rate_source=args.rate_source,
        mode=mode,
        selection=selection,
        rounding=rounding,
    )
    output_path = write_energy_plan(plan, args.output_plan)
    print(
        json.dumps(
            {
                "output_plan": str(output_path),
                "plan_id": plan.plan.plan_id,
                "mode": plan.plan.mode,
                "job_count": len(plan.jobs),
            },
            sort_keys=True,
        )
    )
    return 0


def _dry_run_command(args: argparse.Namespace) -> int:
    from .planning import load_energy_plan

    plan = load_energy_plan(args.plan)
    print(json.dumps(render_dry_run(plan), indent=2, sort_keys=True))
    return 0


def _run_command(args: argparse.Namespace) -> int:
    plan = load_energy_plan(args.plan)
    executor = EnergyExecutor(config=_executor_config_from_args(args, plan.execution))
    summary = executor.run_plan(args.plan)
    print(
        json.dumps(
            {
                "run_root": str(plan.plan.output_root / plan.plan.plan_id),
                "summary": summary,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_live_trial_command(args: argparse.Namespace) -> int:
    executor = EnergyExecutor()
    summary = executor.run_live_trial(
        trial_id=args.trial_id,
        output_dir=args.output_dir,
        workload=args.workload,
        model=args.model,
        base_url=args.base_url,
        endpoint=args.endpoint,
        metrics_url=args.metrics_url,
        gpu_ids=tuple(args.gpu_ids),
        duration_s=args.duration_s,
        request_rate=args.request_rate,
        request_timeout_s=args.request_timeout_s,
        metrics_interval_s=args.metrics_interval_s,
        window_s=args.window_s,
        idle_monitor_duration_s=args.idle_monitor_duration_s,
        traffic_warmup_s=args.traffic_warmup_s,
        repeats=args.repeats,
        repeat_cooldown_s=args.repeat_cooldown_s,
        warmup_each_repeat=bool(args.warmup_each_repeat),
        gpu_monitor_interval_s=args.gpu_monitor_interval_s,
        gpu_monitor_truncate_s=args.gpu_monitor_truncate_s,
        monitor_clock=bool(args.monitor_clock),
        safety_max_outstanding=args.safety_max_outstanding,
        force=bool(args.force),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _resume_command(args: argparse.Namespace) -> int:
    plan = load_energy_plan(args.run_root / "plan.yaml")
    executor = EnergyExecutor(config=_executor_config_from_args(args, plan.execution))
    summary = executor.resume_run(args.run_root, force=bool(args.force))
    print(json.dumps({"run_root": str(args.run_root), "summary": summary}, sort_keys=True))
    return 0


def _status_command(args: argparse.Namespace) -> int:
    executor = EnergyExecutor()
    summary = executor.status(args.run_root)
    print(json.dumps({"run_root": str(args.run_root), "summary": summary}, sort_keys=True))
    return 0


def _add_plan_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--orchestrator-run-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "orchestrator run root to consume; repeat to merge a main run with reruns. "
            "Later roots override earlier roots for the same model/workload/endpoint."
        ),
    )
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("mst-rounded", "sweep", "explicit"),
        default="mst-rounded",
    )
    parser.add_argument(
        "--rate-source",
        choices=("max_slo", "max_no_drift"),
        default="max_slo",
    )
    parser.add_argument("--rounding-policy", default="floor_decimal")
    parser.add_argument("--selection-yaml", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--workloads", nargs="*", default=None)
    parser.add_argument("--experiment-ids", nargs="*", default=None)
    parser.add_argument("--exclude-models", nargs="*", default=None)
    parser.add_argument("--exclude-workloads", nargs="*", default=None)
    parser.add_argument("--exclude-experiment-ids", nargs="*", default=None)
    parser.add_argument("--min-model-size-b", type=float, default=None)
    parser.add_argument("--request-rates", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-models", nargs="*", default=None)
    parser.add_argument("--sweep-experiment-ids", nargs="*", default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)


def _add_executor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allowed-gpu-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-active-gpus", type=int, default=None)
    parser.add_argument("--base-port-start", type=int, default=None)
    parser.add_argument("--base-port-end", type=int, default=None)
    parser.add_argument("--metrics-port-offset", type=int, default=None)


def _executor_config_from_args(
    args: argparse.Namespace,
    plan_execution: EnergyPlanExecution,
) -> EnergyExecutorConfig:
    allowed_gpu_ids = tuple(args.allowed_gpu_ids) if args.allowed_gpu_ids is not None else plan_execution.allowed_gpu_ids
    max_active_gpus = args.max_active_gpus if args.max_active_gpus is not None else plan_execution.max_active_gpus
    base_port_start = args.base_port_start if args.base_port_start is not None else plan_execution.base_port_start
    base_port_end = args.base_port_end if args.base_port_end is not None else plan_execution.base_port_end
    metrics_port_offset = (
        args.metrics_port_offset
        if args.metrics_port_offset is not None
        else plan_execution.metrics_port_offset
    )
    return EnergyExecutorConfig(
        allowed_gpu_ids=allowed_gpu_ids,
        max_active_gpus=max_active_gpus,
        base_port_start=base_port_start,
        base_port_end=base_port_end,
        metrics_port_offset=metrics_port_offset,
    )


def _load_and_merge_selection(args: argparse.Namespace) -> EnergyPlanSelection:
    selection = load_selection_overrides(args.selection_yaml) if args.selection_yaml is not None else EnergyPlanSelection()
    sweep = selection.sweep
    if args.sweep_models is not None:
        sweep = EnergyPlanSelectionSweep(
            enabled=True,
            models=tuple(args.sweep_models),
            experiment_ids=sweep.experiment_ids,
            max_steps=sweep.max_steps,
        )
    if args.sweep_experiment_ids is not None:
        sweep = EnergyPlanSelectionSweep(
            enabled=True,
            models=sweep.models,
            experiment_ids=tuple(args.sweep_experiment_ids),
            max_steps=sweep.max_steps,
        )
    if args.sweep_steps is not None:
        sweep = EnergyPlanSelectionSweep(
            enabled=True,
            models=sweep.models,
            experiment_ids=sweep.experiment_ids,
            max_steps=args.sweep_steps,
        )
    return EnergyPlanSelection(
        models=tuple(selection.models if args.models is None else args.models),
        workloads=tuple(selection.workloads if args.workloads is None else args.workloads),
        experiment_ids=tuple(selection.experiment_ids if args.experiment_ids is None else args.experiment_ids),
        exclude_models=tuple(selection.exclude_models if args.exclude_models is None else args.exclude_models),
        exclude_workloads=tuple(selection.exclude_workloads if args.exclude_workloads is None else args.exclude_workloads),
        exclude_experiment_ids=tuple(
            selection.exclude_experiment_ids
            if args.exclude_experiment_ids is None
            else args.exclude_experiment_ids
        ),
        min_model_size_b=selection.min_model_size_b if args.min_model_size_b is None else args.min_model_size_b,
        explicit_request_rates=tuple(
            selection.explicit_request_rates if args.request_rates is None else tuple(float(value) for value in args.request_rates)
        ),
        sweep=sweep,
    )


def _parse_rounding_policy(value: str) -> EnergyPlanRounding:
    if value == "floor_decimal":
        return EnergyPlanRounding()
    if value == "floor_preferred":
        return EnergyPlanRounding(mst_mode="floor_preferred", sweep_mode="floor_preferred")
    if value.startswith("{"):
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("--rounding-policy JSON must decode to an object")
        return EnergyPlanRounding.from_dict(payload)
    path = Path(value)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--rounding-policy file must decode to an object")
        return EnergyPlanRounding.from_dict(payload)
    raise ValueError(f"unsupported rounding policy: {value}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
