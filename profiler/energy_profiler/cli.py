from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .executor import EnergyExecutor, EnergyExecutorConfig
from .models import EnergyPlanMode, EnergyPlanRounding, EnergyPlanSelection, EnergyPlanSelectionSweep
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
    executor = EnergyExecutor(config=_executor_config_from_args(args))
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


def _resume_command(args: argparse.Namespace) -> int:
    executor = EnergyExecutor(config=_executor_config_from_args(args))
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
    parser.add_argument("--rounding-policy", default="floor_preferred")
    parser.add_argument("--selection-yaml", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--workloads", nargs="*", default=None)
    parser.add_argument("--experiment-ids", nargs="*", default=None)
    parser.add_argument("--request-rates", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-models", nargs="*", default=None)
    parser.add_argument("--sweep-experiment-ids", nargs="*", default=None)
    parser.add_argument("--sweep-steps", type=int, default=None)


def _add_executor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allowed-gpu-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-active-gpus", type=int, default=None)
    parser.add_argument("--base-port-start", type=int, default=8000)
    parser.add_argument("--base-port-end", type=int, default=8099)
    parser.add_argument("--metrics-port-offset", type=int, default=1000)


def _executor_config_from_args(args: argparse.Namespace) -> EnergyExecutorConfig:
    allowed_gpu_ids = tuple(args.allowed_gpu_ids) if args.allowed_gpu_ids else (0, 1, 2, 3)
    return EnergyExecutorConfig(
        allowed_gpu_ids=allowed_gpu_ids,
        max_active_gpus=args.max_active_gpus,
        base_port_start=args.base_port_start,
        base_port_end=args.base_port_end,
        metrics_port_offset=args.metrics_port_offset,
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
        explicit_request_rates=tuple(
            selection.explicit_request_rates if args.request_rates is None else tuple(float(value) for value in args.request_rates)
        ),
        sweep=sweep,
    )


def _parse_rounding_policy(value: str) -> EnergyPlanRounding:
    if value == "floor_preferred":
        return EnergyPlanRounding()
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
