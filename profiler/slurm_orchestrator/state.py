from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from local_orchestrator.state_store import RunStateStore
from local_orchestrator.utils import now_utc_iso

from .planning import deserialize_expanded_job, load_run_plan, read_group_task


def mark_task_running(group_plan_path: str | Path, task_index: int) -> dict[str, Any]:
    task = read_group_task(group_plan_path, task_index)
    state = _load_state_for_task(task)
    state["status"] = "running"
    attempts = state.setdefault("attempts", {"startup": 0, "search": 0})
    attempts["startup"] = max(int(attempts.get("startup", 0)), 1)
    _update_slurm_metadata(state, task)
    state["updated_at"] = now_utc_iso()
    _write_state(Path(str(task["status_path"])), state)
    return state


def finalize_task(
    group_plan_path: str | Path,
    task_index: int,
    *,
    exit_code: int,
    search_started: bool,
) -> dict[str, Any]:
    task = read_group_task(group_plan_path, task_index)
    state = _load_state_for_task(task)
    job = deserialize_expanded_job(task["job"])

    attempts = state.setdefault("attempts", {"startup": 0, "search": 0})
    attempts["startup"] = max(int(attempts.get("startup", 0)), 1)
    if search_started:
        attempts["search"] = max(int(attempts.get("search", 0)), 1)

    artifacts = state.setdefault("artifacts", {})
    search_trace = job.result_dir / "search_trace.json"
    final_report_json = job.result_dir / "final_report.json"
    final_report_md = job.result_dir / "final_report.md"
    artifacts["search_trace"] = str(search_trace) if search_trace.is_file() else None
    artifacts["final_report_json"] = str(final_report_json) if final_report_json.is_file() else None
    artifacts["final_report_md"] = str(final_report_md) if final_report_md.is_file() else None
    artifacts["stdout_log"] = str(task["mst_stdout_log"])
    artifacts["stderr_log"] = str(task["mst_stderr_log"])
    artifacts["vllm_stdout_log"] = str(task["vllm_stdout_log"])
    artifacts["vllm_stderr_log"] = str(task["vllm_stderr_log"])

    if exit_code == 0 and search_trace.is_file() and final_report_json.is_file():
        state["status"] = "succeeded"
        state["last_error"] = None
    elif exit_code == 0:
        state["status"] = "failed"
        state["last_error"] = (
            "MST task exited successfully but required artifacts are missing: "
            f"search_trace_exists={search_trace.is_file()}, "
            f"final_report_json_exists={final_report_json.is_file()}"
        )
    else:
        state["status"] = "failed"
        state["last_error"] = f"Slurm task exited with code {exit_code}"

    _update_slurm_metadata(state, task)
    state["updated_at"] = now_utc_iso()
    _write_state(Path(str(task["status_path"])), state)
    return state


def collect_run(run_root: str | Path) -> dict[str, Any]:
    plan = load_run_plan(run_root)
    run_root_path = Path(str(plan["run_root"]))
    latest_submissions = _load_latest_submission_by_group(run_root_path)
    active_slurm_arrays = _load_active_slurm_arrays(latest_submissions)
    jobs: list[dict[str, Any]] = []
    latest_update = str(plan.get("created_at", now_utc_iso()))
    for job_entry in plan.get("jobs", []):
        status_path = Path(str(job_entry["status_path"]))
        state = _read_json_mapping(status_path)
        if state is None:
            state = dict(job_entry["initial_state"])
        state = _reconcile_slurm_state(
            state=state,
            job_entry=job_entry,
            latest_submissions=latest_submissions,
            active_slurm_arrays=active_slurm_arrays,
        )
        _write_state_if_changed(status_path, state)
        jobs.append(state)
        updated_at = state.get("updated_at")
        if isinstance(updated_at, str) and updated_at > latest_update:
            latest_update = updated_at

    aggregate_state = {
        "run_id": plan.get("run_id"),
        "manifest_path": plan.get("manifest_path"),
        "status": _derive_run_status(jobs),
        "created_at": plan.get("created_at"),
        "updated_at": latest_update,
        "jobs": jobs,
    }
    store = RunStateStore(run_root_path)
    store.save(aggregate_state)
    summary = store.write_summary_files(aggregate_state)
    return {
        "run_root": str(plan["run_root"]),
        "summary": summary,
    }


def _load_state_for_task(task: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(str(task["status_path"]))
    payload = _read_json_mapping(status_path)
    if payload is not None:
        return payload
    initial_state = task.get("initial_state")
    if not isinstance(initial_state, dict):
        raise RuntimeError(f"task is missing initial state: {task.get('experiment_id')}")
    return dict(initial_state)


def _update_slurm_metadata(state: dict[str, Any], task: dict[str, Any]) -> None:
    slurm = state.setdefault("slurm", {})
    slurm.update(
        {
            "group_key": task.get("group_key"),
            "plan_index": task.get("plan_index"),
            "group_task_index": task.get("group_task_index"),
            "gpu_count": task.get("gpu_count"),
            "base_port": task.get("base_port"),
            "base_url": task.get("base_url"),
            "script_path": task.get("script_path"),
            "slurm_stdout_log": task.get("slurm_stdout_log"),
            "slurm_stderr_log": task.get("slurm_stderr_log"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node_name": os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),
        }
    )


def _derive_run_status(jobs: list[dict[str, Any]]) -> str:
    statuses = {str(job.get("status", "planned")) for job in jobs}
    if "running" in statuses or "planned" in statuses:
        return "running"
    if "failed" in statuses:
        return "failed"
    if statuses <= {"succeeded", "skipped"}:
        return "succeeded"
    return "planned"


def _reconcile_slurm_state(
    *,
    state: dict[str, Any],
    job_entry: dict[str, Any],
    latest_submissions: dict[str, str],
    active_slurm_arrays: set[str] | None,
) -> dict[str, Any]:
    status = str(state.get("status", "planned"))
    if status not in {"planned", "running"} or active_slurm_arrays is None:
        return state

    slurm = state.setdefault("slurm", {})
    group_key = str(slurm.get("group_key") or job_entry.get("group_key") or "")
    array_job_id = str(slurm.get("array_job_id") or latest_submissions.get(group_key) or "")
    array_task_id = str(
        slurm.get("array_task_id")
        or slurm.get("group_task_index")
        or job_entry.get("group_task_index")
        or ""
    )
    if not array_job_id or not array_task_id:
        return state
    if array_job_id in active_slurm_arrays:
        return state

    updated = dict(state)
    updated_slurm = dict(slurm)
    updated_slurm.setdefault("array_job_id", array_job_id)
    updated_slurm.setdefault("array_task_id", array_task_id)
    updated["slurm"] = updated_slurm
    updated["status"] = "failed"
    updated["last_error"] = (
        f"Slurm array task {array_job_id}_{array_task_id} is no longer active, "
        f"but orchestrator state remained {status}; the task likely ended before "
        "finalization, for example due to scancel, time limit, or node failure."
    )
    updated["updated_at"] = now_utc_iso()
    return updated


def _load_latest_submission_by_group(run_root: Path) -> dict[str, str]:
    latest: dict[str, tuple[str, str]] = {}
    for path in run_root.glob("*submission.json"):
        payload = _read_json_mapping(path)
        if payload is None:
            continue
        submitted_at = str(payload.get("submitted_at") or "")
        groups = payload.get("groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_key = str(group.get("group_key") or "")
            job_id = str(group.get("job_id") or "")
            if not group_key or not job_id:
                continue
            if group_key not in latest or submitted_at >= latest[group_key][0]:
                latest[group_key] = (submitted_at, job_id)
    return {group_key: job_id for group_key, (_, job_id) in latest.items()}


def _load_active_slurm_arrays(submissions: dict[str, str]) -> set[str] | None:
    array_job_ids = sorted({job_id for job_id in submissions.values() if job_id})
    if not array_job_ids:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", ",".join(array_job_ids), "-o", "%A|%F"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    active: set[str] = set()
    for line in result.stdout.splitlines():
        job_id, separator, array_job_id = line.strip().partition("|")
        for candidate in (job_id, array_job_id if separator else ""):
            if candidate:
                active.add(candidate)
    return active


def _write_state_if_changed(path: Path, payload: dict[str, Any]) -> None:
    existing = _read_json_mapping(path)
    if existing == payload:
        return
    _write_state(path, payload)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload
