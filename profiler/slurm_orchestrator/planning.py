from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from local_orchestrator.lifecycle import render_launch_command
from local_orchestrator.matrix import expand_manifest
from local_orchestrator.models import (
    ExpandedExperimentJob,
    HardwareConfig,
    LaunchConfig,
    OrchestratorManifest,
    ResourceProbeResult,
    SearchConfig,
    SlurmConfig,
)
from local_orchestrator.mst_adapter import build_report_command, build_search_command
from local_orchestrator.state_store import build_job_state_payload
from local_orchestrator.utils import now_utc_iso


def default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"slurm-orchestrator-{ts}"


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_profiler_root() -> Path:
    return resolve_repo_root() / "profiler"


def ensure_run_plan(manifest: OrchestratorManifest, run_id: str) -> dict[str, Any]:
    run_root = (manifest.run.output_root / run_id).resolve()
    plan_path = run_root / "plan.json"
    if plan_path.is_file():
        return load_run_plan(run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run root already exists and is not empty: {run_root}")
    return materialize_run_plan(manifest, run_id)


def materialize_run_plan(manifest: OrchestratorManifest, run_id: str) -> dict[str, Any]:
    repo_root = resolve_repo_root()
    profiler_root = resolve_profiler_root()
    run_root = (manifest.run.output_root / run_id).resolve()
    jobs_dir = run_root / "jobs"
    logs_dir = run_root / "logs"
    groups_dir = run_root / "groups"
    scripts_dir = run_root / "scripts"
    plan_path = run_root / "plan.json"
    if plan_path.exists():
        raise FileExistsError(f"run plan already exists: {plan_path}")

    run_root.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    python_executable = manifest.slurm.python_executable or manifest.run.python_executable or sys.executable
    created_at = now_utc_iso()
    expanded_jobs = [
        _resolved_job(job, repo_root=repo_root)
        for job in expand_manifest(manifest)
    ]

    group_entries: dict[str, list[dict[str, Any]]] = {}
    plan_jobs: list[dict[str, Any]] = []

    for plan_index, job in enumerate(expanded_jobs):
        group_key = _group_key(job)
        group_task_index = len(group_entries.get(group_key, ()))
        base_port = manifest.slurm.base_port + plan_index
        metrics_port = base_port + 1000
        base_url = f"http://{job.launch.host}:{base_port}"
        status_path = jobs_dir / f"{job.experiment_id}.json"
        vllm_stdout_log = logs_dir / f"{job.experiment_id}.vllm.stdout.log"
        vllm_stderr_log = logs_dir / f"{job.experiment_id}.vllm.stderr.log"
        mst_stdout_log = logs_dir / f"{job.experiment_id}.mst.stdout.log"
        mst_stderr_log = logs_dir / f"{job.experiment_id}.mst.stderr.log"

        initial_state = build_job_state_payload(job)
        initial_state["result_dir"] = str(job.result_dir)
        initial_state["artifacts"]["stdout_log"] = str(mst_stdout_log)
        initial_state["artifacts"]["stderr_log"] = str(mst_stderr_log)
        initial_state["artifacts"]["vllm_stdout_log"] = str(vllm_stdout_log)
        initial_state["artifacts"]["vllm_stderr_log"] = str(vllm_stderr_log)
        initial_state["slurm"] = {
            "group_key": group_key,
            "plan_index": plan_index,
            "group_task_index": group_task_index,
            "gpu_count": job.launch.gpu_count,
            "base_port": base_port,
            "base_url": base_url,
        }
        status_path.write_text(json.dumps(initial_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        task_payload = {
            "experiment_id": job.experiment_id,
            "plan_index": plan_index,
            "group_key": group_key,
            "group_task_index": group_task_index,
            "gpu_count": job.launch.gpu_count,
            "base_port": base_port,
            "metrics_port": metrics_port,
            "base_url": base_url,
            "status_path": str(status_path),
            "vllm_stdout_log": str(vllm_stdout_log),
            "vllm_stderr_log": str(vllm_stderr_log),
            "mst_stdout_log": str(mst_stdout_log),
            "mst_stderr_log": str(mst_stderr_log),
            "job": serialize_expanded_job(job),
            "initial_state": initial_state,
        }
        group_entries.setdefault(group_key, []).append(task_payload)
        plan_jobs.append(
            {
                "experiment_id": job.experiment_id,
                "group_key": group_key,
                "plan_index": plan_index,
                "group_task_index": group_task_index,
                "gpu_count": job.launch.gpu_count,
                "base_port": base_port,
                "base_url": base_url,
                "status_path": str(status_path),
                "result_dir": str(job.result_dir),
                "initial_state": initial_state,
            }
        )

    groups_payload: list[dict[str, Any]] = []
    for group_key, tasks in group_entries.items():
        gpu_count = tasks[0]["gpu_count"]
        group_plan_path = groups_dir / f"{group_key}.json"
        script_path = scripts_dir / f"{group_key}.sbatch.sh"
        slurm_stdout_log = logs_dir / f"slurm-{group_key}-%A_%a.out"
        slurm_stderr_log = logs_dir / f"slurm-{group_key}-%A_%a.err"
        array_spec = _array_spec(
            task_count=len(tasks),
            concurrency_limit=manifest.slurm.array_concurrency_limit,
        )
        for task in tasks:
            task["script_path"] = str(script_path)
            task["slurm_stdout_log"] = str(slurm_stdout_log)
            task["slurm_stderr_log"] = str(slurm_stderr_log)

        group_payload = {
            "run_id": run_id,
            "run_root": str(run_root),
            "manifest_path": str(manifest.manifest_path),
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
            render_sbatch_script(
                group_payload=group_payload,
                slurm=manifest.slurm,
            ),
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

    plan_payload = {
        "run_id": run_id,
        "run_root": str(run_root),
        "created_at": created_at,
        "manifest_path": str(manifest.manifest_path),
        "repo_root": str(repo_root),
        "profiler_root": str(profiler_root),
        "python_executable": python_executable,
        "job_count": len(plan_jobs),
        "groups": groups_payload,
        "jobs": plan_jobs,
    }
    _write_json(plan_path, plan_payload)
    return plan_payload


def load_run_plan(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root).resolve() / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"run plan does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("run plan is malformed")
    return payload


def submit_run_plan(run_plan: dict[str, Any]) -> dict[str, Any]:
    return submit_run_plan_tasks(run_plan)


def submit_run_plan_tasks(
    run_plan: dict[str, Any],
    *,
    selected_task_indices_by_group: dict[str, set[int]] | None = None,
    submission_filename: str = "submission.json",
) -> dict[str, Any]:
    repo_root = Path(str(run_plan["repo_root"]))
    submissions: list[dict[str, Any]] = []
    for group in run_plan.get("groups", []):
        group_key = str(group.get("group_key"))
        selected_indices: set[int] | None = None
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


def refresh_run_plan_for_resume(
    manifest: OrchestratorManifest,
    run_root: str | Path,
    *,
    force: bool = False,
    include_experiments: tuple[str, ...] = (),
    exclude_experiments: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, set[int]]]:
    run_plan = load_run_plan(run_root)
    repo_root = Path(str(run_plan["repo_root"]))
    expanded_jobs = [_resolved_job(job, repo_root=repo_root) for job in expand_manifest(manifest)]
    jobs_by_id = {job.experiment_id: job for job in expanded_jobs}
    plan_jobs_by_id = {str(job["experiment_id"]): job for job in run_plan.get("jobs", [])}
    if set(jobs_by_id) != set(plan_jobs_by_id):
        missing = sorted(set(jobs_by_id) - set(plan_jobs_by_id))
        extra = sorted(set(plan_jobs_by_id) - set(jobs_by_id))
        raise ValueError(
            "resume manifest/job mismatch: "
            f"missing_in_plan={missing}, extra_in_plan={extra}"
        )

    group_entries = {str(group["group_key"]): group for group in run_plan.get("groups", [])}
    group_payloads: dict[str, dict[str, Any]] = {}
    for group_key, group in group_entries.items():
        group_payload = _read_json_mapping(Path(str(group["plan_path"])))
        if group_payload is None:
            raise RuntimeError(f"group plan is missing or malformed: {group['plan_path']}")
        group_payloads[group_key] = group_payload
    selected: dict[str, set[int]] = {}
    updated_at = now_utc_iso()

    for experiment_id, job in jobs_by_id.items():
        plan_job = plan_jobs_by_id[experiment_id]
        status_path = Path(str(plan_job["status_path"]))
        state = _read_json_mapping(status_path)
        if state is None:
            state = dict(plan_job.get("initial_state", {}))
        status = str(state.get("status", "planned"))
        if not force and status in {"succeeded", "skipped"}:
            continue
        if not _experiment_selected(
            experiment_id,
            include_patterns=include_experiments,
            exclude_patterns=exclude_experiments,
        ):
            continue

        group_key = _group_key(job)
        if group_key != str(plan_job["group_key"]):
            raise ValueError(
                f"resume cannot change Slurm GPU group for {experiment_id}: "
                f"planned={plan_job['group_key']}, current={group_key}"
            )
        task_index = int(plan_job["group_task_index"])
        group_payload = group_payloads[group_key]
        task = group_payload["jobs"][task_index]
        if str(task.get("experiment_id")) != experiment_id:
            raise RuntimeError(
                f"group task mismatch for {experiment_id}: "
                f"{group_key}[{task_index}]={task.get('experiment_id')}"
            )

        fresh_state = build_job_state_payload(job)
        _refresh_slurm_task_payload(task=task, job=job, fresh_state=fresh_state)
        _refresh_plan_job_entry(plan_job=plan_job, job=job, task=task, fresh_state=fresh_state)
        _refresh_status_for_resume(
            state=state,
            job=job,
            task=task,
            fresh_state=fresh_state,
            force=force,
            updated_at=updated_at,
        )
        _write_json(status_path, state)
        selected.setdefault(group_key, set()).add(task_index)

    for group_key, group_payload in group_payloads.items():
        _write_json(Path(str(group_entries[group_key]["plan_path"])), group_payload)
        script_path = Path(str(group_entries[group_key]["script_path"]))
        script_path.write_text(
            render_sbatch_script(group_payload=group_payload, slurm=manifest.slurm),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
    _write_json(Path(str(run_plan["run_root"])) / "plan.json", run_plan)
    return run_plan, selected


def read_group_task(group_plan_path: str | Path, task_index: int) -> dict[str, Any]:
    path = Path(group_plan_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"group plan is malformed: {path}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError(f"group plan does not contain jobs: {path}")
    if task_index < 0 or task_index >= len(jobs):
        raise IndexError(f"task index out of range for {path}: {task_index}")
    task = jobs[task_index]
    if not isinstance(task, dict):
        raise RuntimeError(f"group plan task is malformed: {path}[{task_index}]")
    return task


def render_task_shell(group_plan_path: str | Path, task_index: int) -> str:
    task = read_group_task(group_plan_path, task_index)
    job = deserialize_expanded_job(task["job"])
    launch_command = render_launch_command(
        launch=job.launch,
        model=job.model,
        gpu_ids=tuple(range(job.launch.gpu_count)),
        base_port=int(task["base_port"]),
        metrics_port=int(task["metrics_port"]),
    )
    search_command = build_search_command(
        job=job,
        base_url=str(task["base_url"]),
        metrics_url=f"{task['base_url']}/metrics",
        python_executable=str(_group_plan_value(group_plan_path, "python_executable")),
    )
    report_command = build_report_command(
        job=job,
        python_executable=str(_group_plan_value(group_plan_path, "python_executable")),
    )

    lines = [
        _bash_assign("EXPERIMENT_ID", job.experiment_id),
        _bash_assign("RESULT_DIR", str(job.result_dir)),
        _bash_assign("BASE_URL", str(task["base_url"])),
        _bash_assign("STATUS_PATH", str(task["status_path"])),
        _bash_assign("VLLM_STDOUT", str(task["vllm_stdout_log"])),
        _bash_assign("VLLM_STDERR", str(task["vllm_stderr_log"])),
        _bash_assign("MST_STDOUT", str(task["mst_stdout_log"])),
        _bash_assign("MST_STDERR", str(task["mst_stderr_log"])),
        _bash_assign("READINESS_PATH", job.launch.readiness_path),
        _bash_assign("READINESS_TIMEOUT_S", _number_text(job.launch.readiness_timeout_s)),
        _bash_assign("READINESS_INTERVAL_S", _number_text(job.launch.readiness_interval_s)),
        _bash_array("VLLM_CMD", launch_command),
        _bash_array("SEARCH_CMD", search_command),
        _bash_array("REPORT_CMD", report_command),
    ]
    for name, value in sorted(job.launch.env.items()):
        lines.append(_bash_export(name, value))
    return "\n".join(lines)


def render_sbatch_script(*, group_payload: dict[str, Any], slurm: SlurmConfig) -> str:
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
        f"#SBATCH -J {_job_name(str(group_payload['run_id']), group_key)}",
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
    if slurm.time is not None:
        lines.append(f"#SBATCH -t {slurm.time}")
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
            'eval "$("$PYTHON_BIN" -m slurm_orchestrator.cli emit-task-shell --group-plan "$GROUP_PLAN" --task-index "$TASK_INDEX")"',
            'export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"',
            "",
            "SEARCH_STARTED=0",
            "",
            "log_gpu_diagnostics() {",
            "  local phase=$1",
            '  echo "==== slurm gpu diagnostics: ${phase} $(date -Is) ===="',
            '  echo "experiment_id=${EXPERIMENT_ID:-unknown}"',
            '  echo "hostname=$(hostname)"',
            '  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"',
            '  echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"',
            '  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"',
            '  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"',
            '  echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"',
            '  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"',
            '  echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"',
            '  echo "VLLM_PID=${VLLM_PID:-}"',
            '  echo "VLLM_CMD=${VLLM_CMD[*]}"',
            "  if command -v nvidia-smi >/dev/null 2>&1; then",
            "    nvidia-smi || true",
            "    nvidia-smi pmon -c 1 || true",
            "  else",
            '    echo "nvidia-smi not found on PATH"',
            "  fi",
            '  echo "==== end slurm gpu diagnostics: ${phase} ===="',
            "}",
            "",
            "cleanup() {",
            "  local exit_code=$?",
            '  if [[ -n "${VLLM_PID:-}" ]]; then',
            '    kill -- -"$VLLM_PID" 2>/dev/null || true',
            '    wait "$VLLM_PID" 2>/dev/null || true',
            "  fi",
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli finalize-task \\',
            '    --group-plan "$GROUP_PLAN" \\',
            '    --task-index "$TASK_INDEX" \\',
            '    --exit-code "$exit_code" \\',
            '    --search-started "$SEARCH_STARTED"',
            "}",
            "trap cleanup EXIT",
            "",
            'mkdir -p "$(dirname "$STATUS_PATH")" "$(dirname "$VLLM_STDOUT")" "$(dirname "$MST_STDOUT")"',
            ': >"$VLLM_STDOUT"',
            ': >"$VLLM_STDERR"',
            ': >"$MST_STDOUT"',
            ': >"$MST_STDERR"',
            'rm -rf "$RESULT_DIR"',
            'mkdir -p "$RESULT_DIR"',
            "",
            'log_gpu_diagnostics "before_vllm_start"',
            'setsid "${VLLM_CMD[@]}" >>"$VLLM_STDOUT" 2>>"$VLLM_STDERR" &',
            "VLLM_PID=$!",
            'log_gpu_diagnostics "after_vllm_start"',
            "",
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli mark-task-running --group-plan "$GROUP_PLAN" --task-index "$TASK_INDEX"'.lstrip(),
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli wait-ready \\'.lstrip(),
            '    --base-url "$BASE_URL" \\'.lstrip(),
            '    --path "$READINESS_PATH" \\'.lstrip(),
            '    --timeout-s "$READINESS_TIMEOUT_S" \\'.lstrip(),
            '    --interval-s "$READINESS_INTERVAL_S" \\'.lstrip(),
            '    --pid "$VLLM_PID"'.lstrip(),
            "",
            "SEARCH_STARTED=1",
            '  "${SEARCH_CMD[@]}" >>"$MST_STDOUT" 2>>"$MST_STDERR"'.lstrip(),
            '  "${REPORT_CMD[@]}" >>"$MST_STDOUT" 2>>"$MST_STDERR"'.lstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def serialize_expanded_job(job: ExpandedExperimentJob) -> dict[str, Any]:
    return {
        "experiment_id": job.experiment_id,
        "source_index": job.source_index,
        "model": job.model,
        "workload": str(job.workload),
        "endpoint": job.endpoint,
        "launch": {
            "template": None if job.launch.template is None else list(job.launch.template),
            "executable": job.launch.executable,
            "extra_args": list(job.launch.extra_args),
            "env": dict(job.launch.env),
            "tensor_parallel_size": job.launch.tensor_parallel_size,
            "gpu_count": job.launch.gpu_count,
            "dtype": job.launch.dtype,
            "quantization": job.launch.quantization,
            "tokenizer_mode": job.launch.tokenizer_mode,
            "gpu_memory_utilization": job.launch.gpu_memory_utilization,
            "max_model_len": job.launch.max_model_len,
            "max_num_seqs": job.launch.max_num_seqs,
            "max_num_batched_tokens": job.launch.max_num_batched_tokens,
            "host": job.launch.host,
            "readiness_path": job.launch.readiness_path,
            "readiness_timeout_s": job.launch.readiness_timeout_s,
            "readiness_interval_s": job.launch.readiness_interval_s,
        },
        "search": {
            "search_mode": job.search.search_mode,
            "trial_min_duration_s": job.search.trial_min_duration_s,
            "trial_max_duration_s": job.search.trial_max_duration_s,
            "final_confirmation_duration_s": job.search.final_confirmation_duration_s,
            "rate_precision": job.search.rate_precision,
            "initial_request_rate": job.search.initial_request_rate,
            "max_request_rate": job.search.max_request_rate,
            "max_binary_steps": job.search.max_binary_steps,
            "max_bracket_trials": job.search.max_bracket_trials,
            "client_limited_retry_attempts": job.search.client_limited_retry_attempts,
            "client_limited_retry_cooldown_s": job.search.client_limited_retry_cooldown_s,
            "closed_loop_initial_concurrency": job.search.closed_loop_initial_concurrency,
            "closed_loop_min_trials": job.search.closed_loop_min_trials,
            "max_closed_loop_concurrency": job.search.max_closed_loop_concurrency,
            "closed_loop_plateau_relative_gain": job.search.closed_loop_plateau_relative_gain,
            "metrics_interval_s": job.search.metrics_interval_s,
            "window_s": job.search.window_s,
            "ttft_slo_ms": job.search.ttft_slo_ms,
            "tpot_slo_ms": job.search.tpot_slo_ms,
            "ttft_slo_field": job.search.ttft_slo_field,
            "tpot_slo_field": job.search.tpot_slo_field,
            "ttft_slo_mode": job.search.ttft_slo_mode,
            "longbench_ttft_static_preset": job.search.longbench_ttft_static_preset,
            "request_reuse_policy": job.search.request_reuse_policy,
            "max_num_seqs": job.search.max_num_seqs,
            "max_num_batched_tokens": job.search.max_num_batched_tokens,
        },
        "hardware": {
            "name": job.hardware.name,
            "gpu_memory_gb": job.hardware.gpu_memory_gb,
            "gpu_memory_utilization": job.hardware.gpu_memory_utilization,
        },
        "probe": None if job.probe is None else job.probe.to_payload(),
        "result_dir": str(job.result_dir),
        "model_slug": job.model_slug,
        "dataset_slug": job.dataset_slug,
        "server_config_slug": job.server_config_slug,
        "server_signature_key": job.server_signature_key,
        "server_metadata_file": None if job.server_metadata_file is None else str(job.server_metadata_file),
    }


def deserialize_expanded_job(payload: dict[str, Any]) -> ExpandedExperimentJob:
    probe_payload = payload.get("probe")
    probe = None
    if isinstance(probe_payload, dict):
        probe = ResourceProbeResult(
            hardware_name=str(probe_payload["hardware_name"]),
            gpu_memory_gb=probe_payload.get("gpu_memory_gb"),
            model_params_b=probe_payload.get("model_params_b"),
            estimated_weight_gb=probe_payload.get("estimated_weight_gb"),
            estimated_activation_gb=probe_payload.get("estimated_activation_gb"),
            estimated_kv_cache_gb=probe_payload.get("estimated_kv_cache_gb"),
            estimated_required_gb=probe_payload.get("estimated_required_gb"),
            usable_memory_per_gpu_gb=probe_payload.get("usable_memory_per_gpu_gb"),
            required_gpu_count=probe_payload.get("required_gpu_count"),
            context_tokens=int(probe_payload["context_tokens"]),
            warnings=tuple(probe_payload.get("warnings", ())),
        )
    launch_payload = dict(payload["launch"])
    search_payload = dict(payload["search"])
    hardware_payload = dict(payload["hardware"])
    return ExpandedExperimentJob(
        experiment_id=str(payload["experiment_id"]),
        source_index=int(payload["source_index"]),
        model=str(payload["model"]),
        workload=Path(str(payload["workload"])),
        endpoint=str(payload["endpoint"]),
        launch=LaunchConfig(
            template=None if launch_payload.get("template") is None else tuple(launch_payload["template"]),
            executable=str(launch_payload["executable"]),
            extra_args=tuple(launch_payload.get("extra_args", ())),
            env=dict(launch_payload.get("env", {})),
            tensor_parallel_size=int(launch_payload["tensor_parallel_size"]),
            gpu_count=int(launch_payload["gpu_count"]),
            dtype=launch_payload.get("dtype"),
            quantization=launch_payload.get("quantization"),
            tokenizer_mode=launch_payload.get("tokenizer_mode"),
            gpu_memory_utilization=launch_payload.get("gpu_memory_utilization"),
            max_model_len=launch_payload.get("max_model_len"),
            max_num_seqs=launch_payload.get("max_num_seqs"),
            max_num_batched_tokens=launch_payload.get("max_num_batched_tokens"),
            host=str(launch_payload["host"]),
            readiness_path=str(launch_payload["readiness_path"]),
            readiness_timeout_s=float(launch_payload["readiness_timeout_s"]),
            readiness_interval_s=float(launch_payload["readiness_interval_s"]),
        ),
        search=SearchConfig(
            search_mode=str(search_payload["search_mode"]),
            trial_min_duration_s=float(search_payload["trial_min_duration_s"]),
            trial_max_duration_s=search_payload.get("trial_max_duration_s"),
            final_confirmation_duration_s=search_payload.get("final_confirmation_duration_s"),
            rate_precision=float(search_payload["rate_precision"]),
            initial_request_rate=float(search_payload["initial_request_rate"]),
            max_request_rate=search_payload.get("max_request_rate"),
            max_binary_steps=int(search_payload["max_binary_steps"]),
            max_bracket_trials=int(search_payload["max_bracket_trials"]),
            client_limited_retry_attempts=int(search_payload.get("client_limited_retry_attempts", 1)),
            client_limited_retry_cooldown_s=float(
                search_payload.get("client_limited_retry_cooldown_s", 30.0)
            ),
            closed_loop_initial_concurrency=int(search_payload["closed_loop_initial_concurrency"]),
            closed_loop_min_trials=int(search_payload["closed_loop_min_trials"]),
            max_closed_loop_concurrency=int(search_payload["max_closed_loop_concurrency"]),
            closed_loop_plateau_relative_gain=float(search_payload["closed_loop_plateau_relative_gain"]),
            metrics_interval_s=float(search_payload["metrics_interval_s"]),
            window_s=float(search_payload["window_s"]),
            ttft_slo_ms=search_payload.get("ttft_slo_ms"),
            tpot_slo_ms=search_payload.get("tpot_slo_ms"),
            ttft_slo_field=str(search_payload["ttft_slo_field"]),
            tpot_slo_field=str(search_payload["tpot_slo_field"]),
            ttft_slo_mode=str(search_payload.get("ttft_slo_mode", "static")),
            longbench_ttft_static_preset=search_payload.get("longbench_ttft_static_preset"),
            request_reuse_policy=str(
                search_payload.get("request_reuse_policy", "no-repeat-across-search")
            ),
            max_num_seqs=search_payload.get("max_num_seqs"),
            max_num_batched_tokens=search_payload.get("max_num_batched_tokens"),
        ),
        hardware=HardwareConfig(
            name=str(hardware_payload["name"]),
            gpu_memory_gb=hardware_payload.get("gpu_memory_gb"),
            gpu_memory_utilization=float(hardware_payload["gpu_memory_utilization"]),
        ),
        probe=probe,
        result_dir=Path(str(payload["result_dir"])),
        model_slug=str(payload["model_slug"]),
        dataset_slug=str(payload["dataset_slug"]),
        server_config_slug=str(payload["server_config_slug"]),
        server_signature_key=str(payload["server_signature_key"]),
        server_metadata_file=(
            None
            if payload.get("server_metadata_file") is None
            else Path(str(payload["server_metadata_file"]))
        ),
    )


def _resolved_job(job: ExpandedExperimentJob, *, repo_root: Path) -> ExpandedExperimentJob:
    result_dir = job.result_dir if job.result_dir.is_absolute() else (repo_root / job.result_dir).resolve()
    return replace(job, result_dir=result_dir)


def _group_key(job: ExpandedExperimentJob) -> str:
    return f"gpu{job.launch.gpu_count}"


def _array_spec(*, task_count: int, concurrency_limit: int | None) -> str:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    suffix = ""
    if concurrency_limit is not None:
        suffix = f"%{concurrency_limit}"
    return f"0-{task_count - 1}{suffix}"


def _selected_array_spec(indices: set[int], *, base_array_spec: str) -> str:
    if not indices:
        raise ValueError("selected array indices must be non-empty")
    suffix = ""
    if "%" in base_array_spec:
        suffix = "%" + base_array_spec.rsplit("%", 1)[1]
    return ",".join(str(index) for index in sorted(indices)) + suffix


def _experiment_selected(
    experiment_id: str,
    *,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> bool:
    lowered = experiment_id.lower()
    if include_patterns and not any(fnmatchcase(lowered, pattern.lower()) for pattern in include_patterns):
        return False
    return not any(fnmatchcase(lowered, pattern.lower()) for pattern in exclude_patterns)


def _cpus_per_task(gpu_count: int, *, slurm: SlurmConfig | None = None) -> int:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if slurm is not None and slurm.cpus_per_task is not None:
        return slurm.cpus_per_task
    cpus_per_gpu = slurm.cpus_per_gpu if slurm is not None else 14
    return cpus_per_gpu * gpu_count


def _job_name(run_id: str, group_key: str) -> str:
    prefix = f"mst-{run_id}-{group_key}"
    if len(prefix) <= 120:
        return prefix
    return prefix[:120]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _refresh_slurm_task_payload(
    *,
    task: dict[str, Any],
    job: ExpandedExperimentJob,
    fresh_state: dict[str, Any],
) -> None:
    fresh_state["artifacts"]["stdout_log"] = str(task["mst_stdout_log"])
    fresh_state["artifacts"]["stderr_log"] = str(task["mst_stderr_log"])
    fresh_state["artifacts"]["vllm_stdout_log"] = str(task["vllm_stdout_log"])
    fresh_state["artifacts"]["vllm_stderr_log"] = str(task["vllm_stderr_log"])
    fresh_state["slurm"] = {
        "group_key": task.get("group_key"),
        "plan_index": task.get("plan_index"),
        "group_task_index": task.get("group_task_index"),
        "gpu_count": job.launch.gpu_count,
        "base_port": task.get("base_port"),
        "base_url": task.get("base_url"),
    }
    task["gpu_count"] = job.launch.gpu_count
    task["job"] = serialize_expanded_job(job)
    task["initial_state"] = fresh_state


def _refresh_plan_job_entry(
    *,
    plan_job: dict[str, Any],
    job: ExpandedExperimentJob,
    task: dict[str, Any],
    fresh_state: dict[str, Any],
) -> None:
    plan_job["gpu_count"] = job.launch.gpu_count
    plan_job["result_dir"] = str(job.result_dir)
    plan_job["base_port"] = task["base_port"]
    plan_job["base_url"] = task["base_url"]
    plan_job["initial_state"] = fresh_state


def _refresh_status_for_resume(
    *,
    state: dict[str, Any],
    job: ExpandedExperimentJob,
    task: dict[str, Any],
    fresh_state: dict[str, Any],
    force: bool,
    updated_at: str,
) -> None:
    for key in (
        "source_index",
        "model",
        "workload",
        "endpoint",
        "hardware",
        "gpu_count",
        "tensor_parallel_size",
        "max_model_len",
        "probe",
        "result_dir",
        "server_signature_key",
    ):
        state[key] = fresh_state[key]
    artifacts = state.setdefault("artifacts", {})
    artifacts["stdout_log"] = str(task["mst_stdout_log"])
    artifacts["stderr_log"] = str(task["mst_stderr_log"])
    artifacts["vllm_stdout_log"] = str(task["vllm_stdout_log"])
    artifacts["vllm_stderr_log"] = str(task["vllm_stderr_log"])
    state["slurm"] = {
        "group_key": task.get("group_key"),
        "plan_index": task.get("plan_index"),
        "group_task_index": task.get("group_task_index"),
        "gpu_count": task.get("gpu_count"),
        "base_port": task.get("base_port"),
        "base_url": task.get("base_url"),
    }
    if force:
        state["status"] = "planned"
        state["attempts"] = {"startup": 0, "search": 0}
        state["last_error"] = None
        state["artifacts"] = {
            "search_trace": None,
            "final_report_json": None,
            "final_report_md": None,
            "stdout_log": str(task["mst_stdout_log"]),
            "stderr_log": str(task["mst_stderr_log"]),
            "vllm_stdout_log": str(task["vllm_stdout_log"]),
            "vllm_stderr_log": str(task["vllm_stderr_log"]),
        }
    state["updated_at"] = updated_at


def _group_plan_value(group_plan_path: str | Path, key: str) -> Any:
    payload = json.loads(Path(group_plan_path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"group plan is malformed: {group_plan_path}")
    return payload[key]


def _bash_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _bash_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def _bash_array(name: str, values: tuple[str, ...]) -> str:
    items = " ".join(shlex.quote(value) for value in values)
    return f"declare -a {name}=({items})"


def _number_text(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric}"
