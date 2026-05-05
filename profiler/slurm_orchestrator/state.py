from __future__ import annotations

import json
import os
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
    jobs: list[dict[str, Any]] = []
    latest_update = str(plan.get("created_at", now_utc_iso()))
    for job_entry in plan.get("jobs", []):
        status_path = Path(str(job_entry["status_path"]))
        state = _read_json_mapping(status_path)
        if state is None:
            state = dict(job_entry["initial_state"])
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
    run_root_path = Path(str(plan["run_root"]))
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
