from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from energy_profiler.models import EnergyPlan, EnergyPlanJob
from energy_profiler.planning import load_energy_plan
from energy_profiler.reporting import EnergyRunStateStore
from local_orchestrator.lifecycle import render_launch_command
from local_orchestrator.manifest import load_manifest
from local_orchestrator.models import SlurmConfig
from local_orchestrator.utils import now_utc_iso

from .planning import (
    _array_spec,
    _bash_array,
    _bash_assign,
    _bash_export,
    _cpus_per_task,
    _number_text,
    _read_json_mapping,
    _selected_array_spec,
    _write_json,
    resolve_profiler_root,
    resolve_repo_root,
)


def default_energy_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"slurm-energy-{ts}"


def ensure_energy_run_plan(plan_path: str | Path, *, run_id: str | None = None) -> dict[str, Any]:
    energy_plan_path = Path(plan_path).resolve()
    plan = _resolve_energy_plan(load_energy_plan(energy_plan_path))
    resolved_run_id = run_id or default_energy_run_id()
    run_root = _energy_run_root(plan, resolved_run_id)
    materialized_plan_path = run_root / "plan.json"
    if materialized_plan_path.is_file():
        return load_energy_run_plan(run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"energy run root already exists and is not empty: {run_root}")
    return materialize_energy_run_plan(plan=plan, plan_path=energy_plan_path, run_id=resolved_run_id)


def materialize_energy_run_plan(
    *,
    plan: EnergyPlan,
    plan_path: str | Path,
    run_id: str,
    slurm: SlurmConfig | None = None,
) -> dict[str, Any]:
    repo_root = resolve_repo_root()
    profiler_root = resolve_profiler_root()
    energy_plan_path = Path(plan_path).resolve()
    resolved_plan = _resolve_energy_plan(plan)
    run_root = _energy_run_root(resolved_plan, run_id)
    plan_json_path = run_root / "plan.json"
    if plan_json_path.exists():
        raise FileExistsError(f"energy run plan already exists: {plan_json_path}")

    jobs_dir = run_root / "jobs"
    state_jobs_dir = run_root / "job-state"
    logs_dir = run_root / "logs"
    groups_dir = run_root / "groups"
    scripts_dir = run_root / "scripts"
    for directory in (run_root, jobs_dir, state_jobs_dir, logs_dir, groups_dir, scripts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(energy_plan_path, run_root / "plan.yaml")
    resolved_slurm = slurm or _load_source_slurm_config(resolved_plan)
    python_executable = (
        resolved_slurm.python_executable
        or resolved_plan.plan.python_executable
        or sys.executable
    )
    created_at = now_utc_iso()
    group_entries: dict[str, list[dict[str, Any]]] = {}
    plan_jobs: list[dict[str, Any]] = []
    initial_jobs: list[dict[str, Any]] = []

    for plan_index, job in enumerate(resolved_plan.jobs):
        group_key = _energy_group_key(job)
        group_task_index = len(group_entries.get(group_key, ()))
        base_port = resolved_slurm.base_port + plan_index
        metrics_port = base_port + resolved_plan.execution.metrics_port_offset
        base_url = f"http://{job.launch.host}:{base_port}"
        result_dir = jobs_dir / job.id
        status_path = state_jobs_dir / f"{job.id}.json"
        vllm_stdout_log = logs_dir / f"{job.id}.vllm.stdout.log"
        vllm_stderr_log = logs_dir / f"{job.id}.vllm.stderr.log"
        profile_stdout_log = logs_dir / f"{job.id}.profile.stdout.log"
        profile_stderr_log = logs_dir / f"{job.id}.profile.stderr.log"

        initial_state = _energy_job_state(
            job=job,
            result_dir=result_dir,
            status_path=status_path,
            group_key=group_key,
            plan_index=plan_index,
            group_task_index=group_task_index,
            base_port=base_port,
            base_url=base_url,
            vllm_stdout_log=vllm_stdout_log,
            vllm_stderr_log=vllm_stderr_log,
            profile_stdout_log=profile_stdout_log,
            profile_stderr_log=profile_stderr_log,
        )
        _write_json(status_path, initial_state)
        initial_jobs.append(initial_state)

        task_payload = {
            "job_id": job.id,
            "plan_index": plan_index,
            "group_key": group_key,
            "group_task_index": group_task_index,
            "gpu_count": job.launch.gpu_count,
            "base_port": base_port,
            "metrics_port": metrics_port,
            "base_url": base_url,
            "status_path": str(status_path),
            "result_dir": str(result_dir),
            "vllm_stdout_log": str(vllm_stdout_log),
            "vllm_stderr_log": str(vllm_stderr_log),
            "profile_stdout_log": str(profile_stdout_log),
            "profile_stderr_log": str(profile_stderr_log),
            "job": job.to_dict(),
            "initial_state": initial_state,
        }
        group_entries.setdefault(group_key, []).append(task_payload)
        plan_jobs.append(
            {
                "job_id": job.id,
                "group_key": group_key,
                "plan_index": plan_index,
                "group_task_index": group_task_index,
                "gpu_count": job.launch.gpu_count,
                "base_port": base_port,
                "base_url": base_url,
                "status_path": str(status_path),
                "result_dir": str(result_dir),
                "initial_state": initial_state,
            }
        )

    groups_payload: list[dict[str, Any]] = []
    for group_key, tasks in group_entries.items():
        gpu_count = tasks[0]["gpu_count"]
        group_plan_path = groups_dir / f"{group_key}.json"
        script_path = scripts_dir / f"{group_key}.sbatch.sh"
        slurm_stdout_log = logs_dir / f"slurm-energy-{group_key}-%A_%a.out"
        slurm_stderr_log = logs_dir / f"slurm-energy-{group_key}-%A_%a.err"
        array_spec = _array_spec(
            task_count=len(tasks),
            concurrency_limit=resolved_slurm.array_concurrency_limit,
        )
        for task in tasks:
            task["script_path"] = str(script_path)
            task["slurm_stdout_log"] = str(slurm_stdout_log)
            task["slurm_stderr_log"] = str(slurm_stderr_log)

        group_payload = {
            "run_id": run_id,
            "run_root": str(run_root),
            "energy_plan_path": str(energy_plan_path),
            "repo_root": str(repo_root),
            "profiler_root": str(profiler_root),
            "python_executable": python_executable,
            "group_key": group_key,
            "gpu_count": gpu_count,
            "array_spec": array_spec,
            "script_path": str(script_path),
            "slurm_stdout_log": str(slurm_stdout_log),
            "slurm_stderr_log": str(slurm_stderr_log),
            "jobs": tasks,
        }
        _write_json(group_plan_path, group_payload)
        script_path.write_text(
            render_energy_sbatch_script(group_payload=group_payload, slurm=resolved_slurm),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        groups_payload.append(
            {
                "group_key": group_key,
                "gpu_count": gpu_count,
                "task_count": len(tasks),
                "array_spec": array_spec,
                "plan_path": str(group_plan_path),
                "script_path": str(script_path),
                "slurm_stdout_log": str(slurm_stdout_log),
                "slurm_stderr_log": str(slurm_stderr_log),
            }
        )

    state = {
        "plan_id": resolved_plan.plan.plan_id,
        "plan_path": str(energy_plan_path),
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "jobs": initial_jobs,
    }
    _write_json(run_root / "state.json", state)

    payload = {
        "run_id": run_id,
        "run_root": str(run_root),
        "created_at": created_at,
        "energy_plan_path": str(energy_plan_path),
        "source_orchestrator_run_root": str(resolved_plan.plan.source_orchestrator_run_root),
        "repo_root": str(repo_root),
        "profiler_root": str(profiler_root),
        "python_executable": python_executable,
        "job_count": len(plan_jobs),
        "groups": groups_payload,
        "jobs": plan_jobs,
    }
    _write_json(plan_json_path, payload)
    return payload


def load_energy_run_plan(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root).resolve() / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"energy run plan does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("energy run plan is malformed")
    return payload


def submit_energy_run_plan(
    run_plan: dict[str, Any],
    *,
    selected_task_indices_by_group: dict[str, set[int]] | None = None,
    submission_filename: str = "submission.json",
) -> dict[str, Any]:
    repo_root = Path(str(run_plan["repo_root"]))
    submissions: list[dict[str, Any]] = []
    for group in run_plan.get("groups", []):
        group_key = str(group.get("group_key"))
        array_spec: str | None = None
        if selected_task_indices_by_group is not None:
            selected_indices = selected_task_indices_by_group.get(group_key, set())
            if not selected_indices:
                continue
            array_spec = _selected_array_spec(
                selected_indices,
                base_array_spec=str(group["array_spec"]),
            )
        script_path = Path(str(group["script_path"]))
        command = ["sbatch", "--parsable"]
        if array_spec is not None:
            command.append(f"--array={array_spec}")
        command.append(str(script_path))
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        payload = {
            "group_key": group_key,
            "script_path": str(script_path),
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        if array_spec is not None:
            payload["array_spec"] = array_spec
        if result.returncode == 0:
            stdout = result.stdout.strip()
            payload["job_id"] = stdout.split(";", 1)[0] if stdout else None
        submissions.append(payload)
    submission_payload = {
        "run_id": run_plan.get("run_id"),
        "run_root": run_plan.get("run_root"),
        "submitted_at": now_utc_iso(),
        "groups": submissions,
    }
    _write_json(Path(str(run_plan["run_root"])) / submission_filename, submission_payload)
    return submission_payload


def read_energy_group_task(group_plan_path: str | Path, task_index: int) -> dict[str, Any]:
    path = Path(group_plan_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"energy group plan is malformed: {path}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError(f"energy group plan does not contain jobs: {path}")
    if task_index < 0 or task_index >= len(jobs):
        raise IndexError(f"task index out of range for {path}: {task_index}")
    task = jobs[task_index]
    if not isinstance(task, dict):
        raise RuntimeError(f"energy group plan task is malformed: {path}[{task_index}]")
    return task


def render_energy_task_shell(group_plan_path: str | Path, task_index: int) -> str:
    group_payload = _read_json_mapping(Path(group_plan_path).resolve())
    if group_payload is None:
        raise RuntimeError(f"energy group plan is missing or malformed: {group_plan_path}")
    task = read_energy_group_task(group_plan_path, task_index)
    job = EnergyPlanJob.from_dict(task["job"])
    launch_command = render_launch_command(
        launch=job.launch.to_launch_config(),
        model=job.model,
        gpu_ids=tuple(range(job.launch.gpu_count)),
        base_port=int(task["base_port"]),
        metrics_port=int(task["metrics_port"]),
    )
    run_trial_command = _build_energy_live_trial_command(
        plan=load_energy_plan(group_payload["energy_plan_path"]),
        job=job,
        python_executable=str(group_payload["python_executable"]),
        base_url=str(task["base_url"]),
        metrics_url=f"{task['base_url']}/metrics",
        output_dir=Path(str(task["result_dir"])),
        gpu_count=int(task["gpu_count"]),
    )
    lines = [
        _bash_assign("ENERGY_JOB_ID", job.id),
        _bash_assign("RESULT_DIR", str(task["result_dir"])),
        _bash_assign("BASE_URL", str(task["base_url"])),
        _bash_assign("STATUS_PATH", str(task["status_path"])),
        _bash_assign("VLLM_STDOUT", str(task["vllm_stdout_log"])),
        _bash_assign("VLLM_STDERR", str(task["vllm_stderr_log"])),
        _bash_assign("PROFILE_STDOUT", str(task["profile_stdout_log"])),
        _bash_assign("PROFILE_STDERR", str(task["profile_stderr_log"])),
        _bash_assign("READINESS_PATH", job.launch.readiness_path),
        _bash_assign("READINESS_TIMEOUT_S", _number_text(job.launch.readiness_timeout_s)),
        _bash_assign("READINESS_INTERVAL_S", _number_text(job.launch.readiness_interval_s)),
        _bash_array("VLLM_CMD", launch_command),
        _bash_array("ENERGY_TRIAL_CMD", run_trial_command),
    ]
    for name, value in sorted(job.launch.env.items()):
        lines.append(_bash_export(name, value))
    return "\n".join(lines)


def render_energy_sbatch_script(*, group_payload: dict[str, Any], slurm: SlurmConfig) -> str:
    group_key = str(group_payload["group_key"])
    gpu_count = int(group_payload["gpu_count"])
    cpus_per_task = _cpus_per_task(gpu_count, slurm=slurm)
    python_executable = str(group_payload["python_executable"])
    repo_root = str(group_payload["repo_root"])
    profiler_root = str(group_payload["profiler_root"])
    group_plan_path = str((Path(str(group_payload["run_root"])) / "groups" / f"{group_key}.json").resolve())
    slurm_stdout_log = str(group_payload["slurm_stdout_log"])
    slurm_stderr_log = str(group_payload["slurm_stderr_log"])
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {_energy_job_name(str(group_payload['run_id']), group_key)}",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --gres=gpu:{gpu_count}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --array={group_payload['array_spec']}",
        f"#SBATCH --output={slurm_stdout_log}",
        f"#SBATCH --error={slurm_stderr_log}",
    ]
    if slurm.account is not None:
        lines.append(f"#SBATCH -A {slurm.account}")
    if slurm.partition is not None:
        lines.append(f"#SBATCH -p {slurm.partition}")
    if slurm.qos is not None:
        lines.append(f"#SBATCH --qos={slurm.qos}")
    lines.append(f"#SBATCH -t {slurm.time or '00:30:00'}")
    if slurm.mem is not None:
        lines.append(f"#SBATCH --mem={slurm.mem}")
    for extra_arg in slurm.sbatch_extra_args:
        lines.append(f"#SBATCH {extra_arg}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f"REPO_ROOT={shlex.quote(repo_root)}",
            f"PROFILER_ROOT={shlex.quote(profiler_root)}",
            f"PYTHON_BIN={shlex.quote(python_executable)}",
            f"GROUP_PLAN={shlex.quote(group_plan_path)}",
            'TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"',
            "",
            'cd "$REPO_ROOT"',
            'export PYTHONPATH="$PROFILER_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
        ]
    )
    for module_name in slurm.modules:
        lines.append(f"module load {module_name}")
    for command in slurm.setup_commands:
        lines.append(command)
    lines.extend(
        [
            "",
            'eval "$("$PYTHON_BIN" -m slurm_orchestrator.cli emit-energy-task-shell --group-plan "$GROUP_PLAN" --task-index "$TASK_INDEX")"',
            "",
            "TRIAL_STARTED=0",
            "",
            "detect_gpu_ids() {",
            '  local raw="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"',
            '  raw="${raw//,/ }"',
            "  local ids=()",
            "  local item",
            "  for item in $raw; do",
            "    if [[ \"$item\" =~ ^[0-9]+$ ]]; then",
            "      ids+=(\"$item\")",
            "    fi",
            "  done",
            "  if [[ ${#ids[@]} -eq 0 ]]; then",
            f"    ids=($(seq 0 {gpu_count - 1}))",
            "  fi",
            '  printf "%s\\n" "${ids[@]}"',
            "}",
            "",
            "log_gpu_diagnostics() {",
            "  local phase=$1",
            '  echo "==== slurm energy gpu diagnostics: ${phase} $(date -Is) ===="',
            '  echo "energy_job_id=${ENERGY_JOB_ID:-unknown}"',
            '  echo "hostname=$(hostname)"',
            '  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"',
            '  echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"',
            '  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"',
            '  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"',
            '  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"',
            '  echo "VLLM_PID=${VLLM_PID:-}"',
            '  echo "VLLM_CMD=${VLLM_CMD[*]}"',
            "  if command -v nvidia-smi >/dev/null 2>&1; then",
            "    nvidia-smi || true",
            "    nvidia-smi pmon -c 1 || true",
            "  else",
            '    echo "nvidia-smi not found on PATH"',
            "  fi",
            '  echo "==== end slurm energy gpu diagnostics: ${phase} ===="',
            "}",
            "",
            "cleanup() {",
            "  local exit_code=$?",
            '  if [[ -n "${VLLM_PID:-}" ]]; then',
            '    kill -- -"$VLLM_PID" 2>/dev/null || true',
            '    wait "$VLLM_PID" 2>/dev/null || true',
            "  fi",
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli finalize-energy-task \\',
            '    --group-plan "$GROUP_PLAN" \\',
            '    --task-index "$TASK_INDEX" \\',
            '    --exit-code "$exit_code" \\',
            '    --trial-started "$TRIAL_STARTED"',
            "}",
            "trap cleanup EXIT",
            "",
            'mkdir -p "$(dirname "$STATUS_PATH")" "$(dirname "$VLLM_STDOUT")" "$RESULT_DIR"',
            'rm -rf "$RESULT_DIR"',
            'mkdir -p "$RESULT_DIR"',
            "",
            'log_gpu_diagnostics "before_vllm_start"',
            'setsid "${VLLM_CMD[@]}" >>"$VLLM_STDOUT" 2>>"$VLLM_STDERR" &',
            "VLLM_PID=$!",
            'log_gpu_diagnostics "after_vllm_start"',
            "",
            'GPU_IDS=($(detect_gpu_ids))',
            'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(IFS=,; echo "${GPU_IDS[*]}")}"',
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli mark-energy-task-running --group-plan "$GROUP_PLAN" --task-index "$TASK_INDEX"'.lstrip(),
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli wait-ready \\'.lstrip(),
            '    --base-url "$BASE_URL" \\'.lstrip(),
            '    --path "$READINESS_PATH" \\'.lstrip(),
            '    --timeout-s "$READINESS_TIMEOUT_S" \\'.lstrip(),
            '    --interval-s "$READINESS_INTERVAL_S" \\'.lstrip(),
            '    --pid "$VLLM_PID"'.lstrip(),
            "",
            "TRIAL_STARTED=1",
            '  "${ENERGY_TRIAL_CMD[@]}" --gpu-ids "${GPU_IDS[@]}" >>"$PROFILE_STDOUT" 2>>"$PROFILE_STDERR"'.lstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def mark_energy_task_running(group_plan_path: str | Path, task_index: int) -> dict[str, Any]:
    task = read_energy_group_task(group_plan_path, task_index)
    state = _load_energy_state_for_task(task)
    state["status"] = "running"
    state["attempts"] = int(state.get("attempts", 0)) + 1
    _update_energy_slurm_metadata(state, task)
    state["updated_at"] = now_utc_iso()
    _write_json(Path(str(task["status_path"])), state)
    return state


def finalize_energy_task(
    group_plan_path: str | Path,
    task_index: int,
    *,
    exit_code: int,
    trial_started: bool,
) -> dict[str, Any]:
    task = read_energy_group_task(group_plan_path, task_index)
    state = _load_energy_state_for_task(task)
    result_dir = Path(str(task["result_dir"]))
    artifacts = state.setdefault("artifacts", {})
    _fill_energy_artifacts(
        artifacts=artifacts,
        result_dir=result_dir,
        profile_stdout_log=Path(str(task["profile_stdout_log"])),
        profile_stderr_log=Path(str(task["profile_stderr_log"])),
        vllm_stdout_log=Path(str(task["vllm_stdout_log"])),
        vllm_stderr_log=Path(str(task["vllm_stderr_log"])),
    )
    if exit_code == 0 and (result_dir / "summary.json").is_file() and (result_dir / "energy_summary.json").is_file():
        state["status"] = "succeeded"
        state["last_error"] = None
    elif exit_code == 0:
        state["status"] = "failed"
        state["last_error"] = (
            "energy task exited successfully but required artifacts are missing: "
            f"summary_exists={(result_dir / 'summary.json').is_file()}, "
            f"energy_summary_exists={(result_dir / 'energy_summary.json').is_file()}"
        )
    else:
        state["status"] = "failed"
        phase = "trial" if trial_started else "startup"
        state["last_error"] = f"Slurm energy task exited with code {exit_code} during {phase}"
    _update_energy_slurm_metadata(state, task)
    state["updated_at"] = now_utc_iso()
    _write_json(Path(str(task["status_path"])), state)
    return state


def collect_energy_run(run_root: str | Path) -> dict[str, Any]:
    plan = load_energy_run_plan(run_root)
    jobs: list[dict[str, Any]] = []
    latest_update = str(plan.get("created_at", now_utc_iso()))
    for job_entry in plan.get("jobs", []):
        status_path = Path(str(job_entry["status_path"]))
        state = _read_json_mapping(status_path)
        if state is None:
            state = dict(job_entry["initial_state"])
        _reconcile_energy_state_artifacts(state)
        jobs.append(state)
        updated_at = state.get("updated_at")
        if isinstance(updated_at, str) and updated_at > latest_update:
            latest_update = updated_at

    aggregate_state = {
        "plan_id": _plan_id_from_run_plan(plan),
        "plan_path": plan.get("energy_plan_path"),
        "status": _derive_energy_run_status(jobs),
        "created_at": plan.get("created_at"),
        "updated_at": latest_update,
        "jobs": jobs,
    }
    run_root_path = Path(str(plan["run_root"]))
    store = EnergyRunStateStore(run_root_path)
    store.save(aggregate_state)
    summary = store.write_summary_files(aggregate_state)
    return {
        "run_root": str(plan["run_root"]),
        "summary": summary,
    }


def _build_energy_live_trial_command(
    *,
    plan: EnergyPlan,
    job: EnergyPlanJob,
    python_executable: str,
    base_url: str,
    metrics_url: str,
    output_dir: Path,
    gpu_count: int,
) -> tuple[str, ...]:
    del gpu_count
    command: list[str] = [
        python_executable,
        "-m",
        "energy_profiler.cli",
        "run-live-trial",
        "--trial-id",
        job.id,
        "--output-dir",
        str(output_dir),
        "--workload",
        str(job.workload),
        "--model",
        job.model,
        "--base-url",
        base_url,
        "--endpoint",
        job.endpoint,
        "--metrics-url",
        metrics_url,
        "--duration-s",
        str(plan.defaults.duration_s),
        "--request-rate",
        str(job.request_rate),
        "--request-timeout-s",
        str(plan.defaults.request_timeout_s),
        "--metrics-interval-s",
        str(plan.defaults.metrics_interval_s),
        "--window-s",
        str(plan.defaults.window_s),
        "--idle-monitor-duration-s",
        str(plan.defaults.idle_monitor_duration_s),
        "--gpu-monitor-interval-s",
        str(plan.defaults.gpu_monitor_interval_s),
        "--gpu-monitor-truncate-s",
        str(plan.defaults.gpu_monitor_truncate_s),
        "--force",
    ]
    if plan.defaults.monitor_clock:
        command.append("--monitor-clock")
    if plan.defaults.safety_max_outstanding is not None:
        command.extend(["--safety-max-outstanding", str(plan.defaults.safety_max_outstanding)])
    return tuple(command)


def _energy_run_root(plan: EnergyPlan, run_id: str) -> Path:
    return (plan.plan.output_root / plan.plan.plan_id / run_id).resolve()


def _resolve_energy_plan(plan: EnergyPlan) -> EnergyPlan:
    root = resolve_repo_root()
    header = plan.plan
    output_root = header.output_root if header.output_root.is_absolute() else (root / header.output_root).resolve()
    source_root = (
        header.source_orchestrator_run_root
        if header.source_orchestrator_run_root.is_absolute()
        else (root / header.source_orchestrator_run_root).resolve()
    )
    jobs = []
    for job in plan.jobs:
        source_result_dir = (
            job.source_result_dir
            if job.source_result_dir.is_absolute()
            else (root / job.source_result_dir).resolve()
        )
        workload = job.workload if job.workload.is_absolute() else (root / job.workload).resolve()
        jobs.append(replace(job, source_result_dir=source_result_dir, workload=workload))
    return replace(
        plan,
        plan=replace(header, output_root=output_root, source_orchestrator_run_root=source_root),
        jobs=tuple(jobs),
    )


def _load_source_slurm_config(plan: EnergyPlan) -> SlurmConfig:
    source_root = plan.plan.source_orchestrator_run_root
    manifest_path: Path | None = None
    state = _read_json_mapping(source_root / "state.json")
    if state is not None and isinstance(state.get("manifest_path"), str):
        manifest_path = Path(str(state["manifest_path"]))
    if manifest_path is None:
        run_plan = _read_json_mapping(source_root / "plan.json")
        if run_plan is not None and isinstance(run_plan.get("manifest_path"), str):
            manifest_path = Path(str(run_plan["manifest_path"]))
    if manifest_path is None:
        return SlurmConfig()
    try:
        return load_manifest(manifest_path).slurm
    except Exception:
        return SlurmConfig()


def _energy_group_key(job: EnergyPlanJob) -> str:
    return f"gpu{job.launch.gpu_count}"


def _energy_job_name(run_id: str, group_key: str) -> str:
    prefix = f"energy-{run_id}-{group_key}"
    if len(prefix) <= 120:
        return prefix
    return prefix[:120]


def _energy_job_state(
    *,
    job: EnergyPlanJob,
    result_dir: Path,
    status_path: Path,
    group_key: str,
    plan_index: int,
    group_task_index: int,
    base_port: int,
    base_url: str,
    vllm_stdout_log: Path,
    vllm_stderr_log: Path,
    profile_stdout_log: Path,
    profile_stderr_log: Path,
) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": "planned",
        "source_experiment_id": job.source_experiment_id,
        "model": job.model,
        "workload": str(job.workload),
        "endpoint": job.endpoint,
        "request_rate": job.request_rate,
        "mst_rate": job.mst_rate,
        "result_dir": str(result_dir),
        "server_signature_key": job.server_signature_key,
        "attempts": 0,
        "last_error": None,
        "gpu_ids": None,
        "base_url": None,
        "artifacts": {
            "summary_json": None,
            "request_records_jsonl": None,
            "server_metrics_jsonl": None,
            "windows_csv": None,
            "gpu_power_json": None,
            "energy_summary_json": None,
            "profile_stdout_log": str(profile_stdout_log),
            "profile_stderr_log": str(profile_stderr_log),
            "vllm_stdout_log": str(vllm_stdout_log),
            "vllm_stderr_log": str(vllm_stderr_log),
        },
        "slurm": {
            "group_key": group_key,
            "plan_index": plan_index,
            "group_task_index": group_task_index,
            "gpu_count": job.launch.gpu_count,
            "base_port": base_port,
            "base_url": base_url,
            "status_path": str(status_path),
        },
    }


def _load_energy_state_for_task(task: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(str(task["status_path"]))
    payload = _read_json_mapping(status_path)
    if payload is not None:
        return payload
    initial_state = task.get("initial_state")
    if not isinstance(initial_state, dict):
        raise RuntimeError(f"energy task is missing initial state: {task.get('job_id')}")
    return dict(initial_state)


def _update_energy_slurm_metadata(state: dict[str, Any], task: dict[str, Any]) -> None:
    state["base_url"] = task.get("base_url")
    gpu_ids = _env_gpu_ids()
    if gpu_ids:
        state["gpu_ids"] = gpu_ids
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


def _env_gpu_ids() -> list[int]:
    raw = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    ids: list[int] = []
    for item in raw.replace(",", " ").split():
        if item.isdigit():
            ids.append(int(item))
    return ids


def _fill_energy_artifacts(
    *,
    artifacts: dict[str, Any],
    result_dir: Path,
    profile_stdout_log: Path,
    profile_stderr_log: Path,
    vllm_stdout_log: Path,
    vllm_stderr_log: Path,
) -> None:
    artifacts.update(
        {
            "summary_json": str(result_dir / "summary.json") if (result_dir / "summary.json").is_file() else None,
            "request_records_jsonl": str(result_dir / "request_records.jsonl")
            if (result_dir / "request_records.jsonl").is_file()
            else None,
            "server_metrics_jsonl": str(result_dir / "server_metrics.jsonl")
            if (result_dir / "server_metrics.jsonl").is_file()
            else None,
            "windows_csv": str(result_dir / "windows.csv") if (result_dir / "windows.csv").is_file() else None,
            "gpu_power_json": str(result_dir / "gpu_power.json") if (result_dir / "gpu_power.json").is_file() else None,
            "energy_summary_json": str(result_dir / "energy_summary.json")
            if (result_dir / "energy_summary.json").is_file()
            else None,
            "profile_stdout_log": str(profile_stdout_log),
            "profile_stderr_log": str(profile_stderr_log),
            "vllm_stdout_log": str(vllm_stdout_log),
            "vllm_stderr_log": str(vllm_stderr_log),
        }
    )


def _reconcile_energy_state_artifacts(state: dict[str, Any]) -> None:
    result_dir = Path(str(state.get("result_dir")))
    artifacts = state.setdefault("artifacts", {})
    _fill_energy_artifacts(
        artifacts=artifacts,
        result_dir=result_dir,
        profile_stdout_log=Path(str(artifacts.get("profile_stdout_log") or "")),
        profile_stderr_log=Path(str(artifacts.get("profile_stderr_log") or "")),
        vllm_stdout_log=Path(str(artifacts.get("vllm_stdout_log") or "")),
        vllm_stderr_log=Path(str(artifacts.get("vllm_stderr_log") or "")),
    )
    if (
        state.get("status") != "succeeded"
        and (result_dir / "summary.json").is_file()
        and (result_dir / "energy_summary.json").is_file()
    ):
        state["status"] = "succeeded"
        state["last_error"] = None


def _derive_energy_run_status(jobs: list[dict[str, Any]]) -> str:
    statuses = {str(job.get("status", "planned")) for job in jobs}
    if "running" in statuses or "planned" in statuses:
        return "running"
    if "failed" in statuses:
        return "failed"
    if statuses <= {"succeeded", "skipped"}:
        return "completed"
    return "planned"


def _plan_id_from_run_plan(plan: dict[str, Any]) -> str | None:
    try:
        return load_energy_plan(str(plan["energy_plan_path"])).plan.plan_id
    except Exception:
        return None
