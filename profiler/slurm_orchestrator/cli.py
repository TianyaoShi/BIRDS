from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from local_orchestrator.manifest import load_manifest

from .planning import (
    default_run_id,
    ensure_run_plan,
    load_run_plan,
    refresh_run_plan_for_resume,
    render_task_shell,
    submit_run_plan,
    submit_run_plan_tasks,
)
from .state import collect_run, finalize_task, mark_task_running


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slurm_orchestrator.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--run-id", default=None)
    plan.set_defaults(handler=_plan_command)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--manifest", type=Path, required=True)
    submit.add_argument("--run-id", default=None)
    submit.set_defaults(handler=_submit_command)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument(
        "--force",
        action="store_true",
        help="rerun all jobs, including previously succeeded ones",
    )
    resume.add_argument(
        "--include-experiment",
        action="append",
        default=[],
        help="shell-style experiment id pattern to include; may be repeated",
    )
    resume.add_argument(
        "--exclude-experiment",
        action="append",
        default=[],
        help="shell-style experiment id pattern to exclude; may be repeated",
    )
    resume.set_defaults(handler=_resume_command)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--run-root", type=Path, required=True)
    collect.set_defaults(handler=_collect_command)

    emit_task_shell = subparsers.add_parser("emit-task-shell", help=argparse.SUPPRESS)
    emit_task_shell.add_argument("--group-plan", type=Path, required=True)
    emit_task_shell.add_argument("--task-index", type=int, required=True)
    emit_task_shell.set_defaults(handler=_emit_task_shell_command)

    wait_ready = subparsers.add_parser("wait-ready", help=argparse.SUPPRESS)
    wait_ready.add_argument("--base-url", required=True)
    wait_ready.add_argument("--path", required=True)
    wait_ready.add_argument("--timeout-s", type=float, required=True)
    wait_ready.add_argument("--interval-s", type=float, required=True)
    wait_ready.add_argument("--pid", type=int, default=None)
    wait_ready.set_defaults(handler=_wait_ready_command)

    mark_running = subparsers.add_parser("mark-task-running", help=argparse.SUPPRESS)
    mark_running.add_argument("--group-plan", type=Path, required=True)
    mark_running.add_argument("--task-index", type=int, required=True)
    mark_running.set_defaults(handler=_mark_task_running_command)

    finalize = subparsers.add_parser("finalize-task", help=argparse.SUPPRESS)
    finalize.add_argument("--group-plan", type=Path, required=True)
    finalize.add_argument("--task-index", type=int, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.add_argument("--search-started", type=int, choices=(0, 1), required=True)
    finalize.set_defaults(handler=_finalize_task_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _plan_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run_id = args.run_id or manifest.run.run_id or default_run_id()
    payload = ensure_run_plan(manifest, run_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _submit_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run_id = args.run_id or manifest.run.run_id or default_run_id()
    plan = ensure_run_plan(manifest, run_id)
    submission = submit_run_plan(plan)
    print(json.dumps(submission, indent=2, sort_keys=True))
    return 0 if all(group.get("return_code", 1) == 0 for group in submission.get("groups", [])) else 1


def _resume_command(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    existing_plan = load_run_plan(run_root)
    manifest = load_manifest(Path(str(existing_plan["manifest_path"])))
    refreshed_plan, selected = refresh_run_plan_for_resume(
        manifest,
        run_root,
        force=bool(args.force),
        include_experiments=tuple(args.include_experiment or ()),
        exclude_experiments=tuple(args.exclude_experiment or ()),
    )
    submission = submit_run_plan_tasks(
        refreshed_plan,
        selected_task_indices_by_group=selected,
        submission_filename="resume-submission.json",
    )
    submission["selected_task_count"] = sum(len(indices) for indices in selected.values())
    print(json.dumps(submission, indent=2, sort_keys=True))
    return 0 if all(group.get("return_code", 1) == 0 for group in submission.get("groups", [])) else 1


def _collect_command(args: argparse.Namespace) -> int:
    payload = collect_run(args.run_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _emit_task_shell_command(args: argparse.Namespace) -> int:
    print(render_task_shell(args.group_plan, args.task_index))
    return 0


def _wait_ready_command(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        if args.pid is not None and not _pid_exists(args.pid):
            print(
                f"vLLM process exited before readiness: pid={args.pid}",
                file=sys.stderr,
            )
            return 1
        if _readiness_probe(args.base_url, args.path, timeout_s=2.0):
            return 0
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(args.interval_s, remaining_s))
    print(
        f"timed out waiting for readiness at {args.base_url.rstrip('/')}{args.path}",
        file=sys.stderr,
    )
    return 1


def _mark_task_running_command(args: argparse.Namespace) -> int:
    payload = mark_task_running(args.group_plan, args.task_index)
    print(json.dumps({"experiment_id": payload.get("experiment_id"), "status": payload.get("status")}))
    return 0


def _finalize_task_command(args: argparse.Namespace) -> int:
    payload = finalize_task(
        args.group_plan,
        args.task_index,
        exit_code=args.exit_code,
        search_started=bool(args.search_started),
    )
    print(json.dumps({"experiment_id": payload.get("experiment_id"), "status": payload.get("status")}))
    return 0


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _readiness_probe(base_url: str, path: str, *, timeout_s: float) -> bool:
    url = f"{base_url.rstrip('/')}{path}"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 0)
            return 200 <= int(status) < 500
    except urllib.error.URLError:
        return False
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
