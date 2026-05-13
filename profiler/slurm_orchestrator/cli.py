from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from local_orchestrator.manifest import load_manifest

from .energy import (
    collect_energy_run,
    ensure_energy_run_plan,
    finalize_energy_task,
    mark_energy_task_running,
    render_energy_task_shell,
    submit_energy_run_plan,
)
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


DEFAULT_RESULTS_SYNC_DEST = Path("/depot/yiding/data/BioLLM-results/results")
RESULTS_SYNC_DEST_ENV = "SLURM_ORCHESTRATOR_SYNC_RESULTS_TO"
RESULTS_SYNC_SCOPE_ENV = "SLURM_ORCHESTRATOR_SYNC_RESULTS_SCOPE"
RESULTS_SYNC_EXISTING_ENV = "SLURM_ORCHESTRATOR_SYNC_RESULTS_EXISTING"
RESULTS_SYNC_ROOT_ENV = "SLURM_ORCHESTRATOR_SYNC_RESULTS_ROOT"
RESULTS_SYNC_DISABLE_ENV = "SLURM_ORCHESTRATOR_DISABLE_RESULT_SYNC"


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
    collect.add_argument(
        "--sync-results-to",
        type=Path,
        default=None,
        help=(
            "publish selected result files to this shared results directory after collect; "
            f"defaults to ${RESULTS_SYNC_DEST_ENV} or {DEFAULT_RESULTS_SYNC_DEST} when available"
        ),
    )
    collect.add_argument(
        "--sync-results-scope",
        choices=("run", "all"),
        default=None,
        help=(
            "publish a compact subset for the collected run or mirror the full results tree; defaults to "
            f"${RESULTS_SYNC_SCOPE_ENV} or 'run'"
        ),
    )
    collect.add_argument(
        "--sync-results-existing",
        choices=("update", "missing"),
        default=None,
        help=(
            "update changed destination files or copy only files missing from the destination; "
            f"defaults to ${RESULTS_SYNC_EXISTING_ENV} or 'update'"
        ),
    )
    collect.add_argument(
        "--sync-results-root",
        type=Path,
        default=None,
        help=(
            "source results tree to publish from; defaults to "
            f"${RESULTS_SYNC_ROOT_ENV} or the nearest parent directory named 'results'"
        ),
    )
    collect.add_argument(
        "--no-sync-results",
        action="store_true",
        help=f"skip result publishing after collect, also set by ${RESULTS_SYNC_DISABLE_ENV}=1",
    )
    collect.set_defaults(handler=_collect_command)

    energy_submit = subparsers.add_parser("energy-submit")
    energy_submit.add_argument("--plan", type=Path, required=True)
    energy_submit.add_argument("--run-id", default=None)
    energy_submit.set_defaults(handler=_energy_submit_command)

    energy_collect = subparsers.add_parser("energy-collect")
    energy_collect.add_argument("--run-root", type=Path, required=True)
    energy_collect.set_defaults(handler=_energy_collect_command)

    emit_task_shell = subparsers.add_parser("emit-task-shell", help=argparse.SUPPRESS)
    emit_task_shell.add_argument("--group-plan", type=Path, required=True)
    emit_task_shell.add_argument("--task-index", type=int, required=True)
    emit_task_shell.set_defaults(handler=_emit_task_shell_command)

    emit_energy_task_shell = subparsers.add_parser("emit-energy-task-shell", help=argparse.SUPPRESS)
    emit_energy_task_shell.add_argument("--group-plan", type=Path, required=True)
    emit_energy_task_shell.add_argument("--task-index", type=int, required=True)
    emit_energy_task_shell.set_defaults(handler=_emit_energy_task_shell_command)

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

    mark_energy_running = subparsers.add_parser("mark-energy-task-running", help=argparse.SUPPRESS)
    mark_energy_running.add_argument("--group-plan", type=Path, required=True)
    mark_energy_running.add_argument("--task-index", type=int, required=True)
    mark_energy_running.set_defaults(handler=_mark_energy_task_running_command)

    finalize = subparsers.add_parser("finalize-task", help=argparse.SUPPRESS)
    finalize.add_argument("--group-plan", type=Path, required=True)
    finalize.add_argument("--task-index", type=int, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.add_argument("--search-started", type=int, choices=(0, 1), required=True)
    finalize.set_defaults(handler=_finalize_task_command)

    finalize_energy = subparsers.add_parser("finalize-energy-task", help=argparse.SUPPRESS)
    finalize_energy.add_argument("--group-plan", type=Path, required=True)
    finalize_energy.add_argument("--task-index", type=int, required=True)
    finalize_energy.add_argument("--exit-code", type=int, required=True)
    finalize_energy.add_argument("--trial-started", type=int, choices=(0, 1), required=True)
    finalize_energy.set_defaults(handler=_finalize_energy_task_command)

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
    sync_result = _sync_results_after_collect(args)
    payload["result_sync"] = sync_result
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if sync_result["status"] == "failed" else 0


def _energy_submit_command(args: argparse.Namespace) -> int:
    plan = ensure_energy_run_plan(args.plan, run_id=args.run_id)
    submission = submit_energy_run_plan(plan)
    print(json.dumps(submission, indent=2, sort_keys=True))
    return 0 if all(group.get("return_code", 1) == 0 for group in submission.get("groups", [])) else 1


def _energy_collect_command(args: argparse.Namespace) -> int:
    payload = collect_energy_run(args.run_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _emit_task_shell_command(args: argparse.Namespace) -> int:
    print(render_task_shell(args.group_plan, args.task_index))
    return 0


def _emit_energy_task_shell_command(args: argparse.Namespace) -> int:
    print(render_energy_task_shell(args.group_plan, args.task_index))
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


def _mark_energy_task_running_command(args: argparse.Namespace) -> int:
    payload = mark_energy_task_running(args.group_plan, args.task_index)
    print(json.dumps({"job_id": payload.get("job_id"), "status": payload.get("status")}))
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


def _finalize_energy_task_command(args: argparse.Namespace) -> int:
    payload = finalize_energy_task(
        args.group_plan,
        args.task_index,
        exit_code=args.exit_code,
        trial_started=bool(args.trial_started),
    )
    print(json.dumps({"job_id": payload.get("job_id"), "status": payload.get("status")}))
    return 0


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _sync_results_after_collect(args: argparse.Namespace) -> dict[str, object]:
    if bool(getattr(args, "no_sync_results", False)) or _env_flag(RESULTS_SYNC_DISABLE_ENV):
        return {"status": "skipped", "reason": "disabled"}

    scope = _option_or_env(args, "sync_results_scope", RESULTS_SYNC_SCOPE_ENV, "run")
    existing = _option_or_env(args, "sync_results_existing", RESULTS_SYNC_EXISTING_ENV, "update")
    if scope not in {"run", "all"}:
        return {"status": "failed", "reason": f"invalid sync scope: {scope}"}
    if existing not in {"update", "missing"}:
        return {"status": "failed", "reason": f"invalid sync existing mode: {existing}"}
    explicit_dest = getattr(args, "sync_results_to", None) or os.environ.get(RESULTS_SYNC_DEST_ENV)
    destination_root = Path(explicit_dest) if explicit_dest else DEFAULT_RESULTS_SYNC_DEST
    default_destination = explicit_dest is None
    if default_destination and not destination_root.parent.exists():
        return {
            "status": "skipped",
            "reason": f"default destination parent does not exist: {destination_root.parent}",
            "destination": str(destination_root),
        }
    if destination_root.is_symlink():
        return {
            "status": "failed",
            "reason": (
                "destination root is a symlink; replace it with a real directory before syncing: "
                f"rm '{destination_root}' && mkdir -p '{destination_root}'"
            ),
            "destination": str(destination_root),
        }

    explicit_source = getattr(args, "sync_results_root", None) or os.environ.get(RESULTS_SYNC_ROOT_ENV)
    results_root = Path(explicit_source).resolve() if explicit_source else _infer_results_root(args.run_root)
    run_root = Path(args.run_root).resolve()
    if scope == "all":
        source = results_root
        destination = destination_root
        files_from: list[str] | None = None
    else:
        source = results_root
        destination = destination_root
        files_from = _collect_publish_files(results_root=results_root, run_root=run_root)
        if not files_from:
            return {
                "status": "skipped",
                "reason": "no publishable summary, analysis, or plot files were found",
                "scope": scope,
                "existing": existing,
                "source": str(source),
                "destination": str(destination),
            }

    if not source.is_dir():
        return {"status": "failed", "reason": f"source directory does not exist: {source}", "source": str(source)}

    if destination.is_symlink():
        return {
            "status": "failed",
            "reason": (
                "destination is a symlink; replace it with a real directory before syncing: "
                f"rm '{destination}' && mkdir -p '{destination}'"
            ),
            "source": str(source),
            "destination": str(destination),
        }

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "status": "failed",
            "reason": f"could not create destination directory: {exc}",
            "source": str(source),
            "destination": str(destination),
        }

    command = [
        "rsync",
        "-a",
        "--partial",
        "--human-readable",
        "--info=stats2",
        "--chmod=D755,F644",
    ]
    if existing == "missing":
        command.append("--ignore-existing")
    elif files_from is None:
        command.append("--delete-delay")

    temp_list = None
    try:
        try:
            if files_from is not None:
                temp_list = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=True)
                temp_list.write("\n".join(files_from))
                temp_list.write("\n")
                temp_list.flush()
                command.extend(["--files-from", temp_list.name])
            command.extend([f"{source}/", f"{destination}/"])
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        finally:
            if temp_list is not None:
                temp_list.close()
    except FileNotFoundError:
        return {
            "status": "failed",
            "reason": "rsync executable was not found",
            "source": str(source),
            "destination": str(destination),
        }

    if completed.returncode != 0:
        return {
            "status": "failed",
            "reason": _last_output_line(completed.stderr) or _last_output_line(completed.stdout) or "rsync failed",
            "return_code": completed.returncode,
            "source": str(source),
            "destination": str(destination),
        }

    return {
        "status": "succeeded",
        "scope": scope,
        "existing": existing,
        "source": str(source),
        "destination": str(destination),
        "file_count": len(files_from) if files_from is not None else None,
        "return_code": completed.returncode,
        "summary": _last_output_line(completed.stdout),
    }


def _infer_results_root(run_root: str | Path) -> Path:
    resolved = Path(run_root).resolve()
    candidates = (resolved, *resolved.parents)
    for candidate in candidates:
        if candidate.name == "results":
            return candidate
    return resolved


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _option_or_env(args: argparse.Namespace, option: str, env_name: str, default: str) -> str:
    value = getattr(args, option, None) or os.environ.get(env_name) or default
    return str(value).strip().lower()


def _collect_publish_files(*, results_root: Path, run_root: Path) -> list[str]:
    candidates: set[Path] = set()
    for name in ("summary.json", "summary.md"):
        _add_relative_file(candidates, results_root=results_root, path=run_root / name)

    state_path = run_root / "state.json"
    state = _read_json_mapping(state_path)
    if state is not None:
        for job in state.get("jobs", []):
            if not isinstance(job, dict):
                continue
            artifacts = job.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            for key in ("final_report_json", "final_report_md"):
                artifact_path = artifacts.get(key)
                if isinstance(artifact_path, str) and artifact_path:
                    path = Path(artifact_path)
                    _add_relative_file(candidates, results_root=results_root, path=path)
                    _add_mst_publish_files(candidates, results_root=results_root, result_dir=path.parent)

    analysis_dir = results_root / "analysis" / run_root.name
    if analysis_dir.is_dir():
        for path in analysis_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
                _add_relative_file(candidates, results_root=results_root, path=path)

    return sorted(path.as_posix() for path in candidates)


def _add_mst_publish_files(candidates: set[Path], *, results_root: Path, result_dir: Path) -> None:
    if not result_dir.is_dir():
        return
    plot_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    allowed_names = {"final_report.json", "final_report.md", "summary.json", "summary.md", "analysis.json"}
    for path in result_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in allowed_names:
            _add_relative_file(candidates, results_root=results_root, path=path)
        elif "plots" in path.parts and path.suffix.lower() in plot_suffixes:
            _add_relative_file(candidates, results_root=results_root, path=path)


def _add_relative_file(candidates: set[Path], *, results_root: Path, path: Path) -> None:
    if not path.is_file():
        return
    try:
        candidates.add(path.resolve().relative_to(results_root))
    except ValueError:
        return


def _read_json_mapping(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _last_output_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


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
