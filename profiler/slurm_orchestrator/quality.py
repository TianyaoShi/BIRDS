from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_orchestrator.lifecycle import render_launch_command
from local_orchestrator.matrix import _dataset_slug
from local_orchestrator.models import LaunchConfig, SlurmConfig
from local_orchestrator.utils import now_utc_iso
from output_quality_profiler.manifest import load_quality_manifest
from output_quality_profiler.matrix import expand_quality_manifest
from output_quality_profiler.models import (
    QualityDecodingConfig,
    QualityExperimentJob,
    QualityGenerationConfig,
    QualityRunManifest,
)

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


def default_quality_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"slurm-quality-{ts}"


def ensure_quality_run_plan(manifest_path: str | Path, *, run_id: str | None = None) -> dict[str, Any]:
    manifest = load_quality_manifest(manifest_path)
    resolved_run_id = run_id or manifest.run.run_id or default_quality_run_id()
    run_root = (manifest.run.output_root / resolved_run_id).resolve()
    plan_path = run_root / "plan.json"
    if plan_path.is_file():
        return load_quality_run_plan(run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"quality run root already exists and is not empty: {run_root}")
    return materialize_quality_run_plan(manifest=manifest, run_id=resolved_run_id)


def materialize_quality_run_plan(*, manifest: QualityRunManifest, run_id: str) -> dict[str, Any]:
    repo_root = resolve_repo_root()
    profiler_root = resolve_profiler_root()
    run_root = (manifest.run.output_root / run_id).resolve()
    plan_path = run_root / "plan.json"
    if plan_path.exists():
        raise FileExistsError(f"quality run plan already exists: {plan_path}")
    jobs_dir = run_root / "jobs"
    logs_dir = run_root / "logs"
    groups_dir = run_root / "groups"
    scripts_dir = run_root / "scripts"
    state_jobs_dir = run_root / "job-state"
    for directory in (run_root, jobs_dir, logs_dir, groups_dir, scripts_dir, state_jobs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest.manifest_path, run_root / "manifest.yaml")

    python_executable = manifest.slurm.python_executable or manifest.run.python_executable or sys.executable
    created_at = now_utc_iso()
    expanded_jobs = expand_quality_manifest(manifest, run_root=run_root)
    group_entries: dict[str, list[dict[str, Any]]] = {}
    plan_jobs: list[dict[str, Any]] = []
    initial_jobs: list[dict[str, Any]] = []

    for plan_index, job in enumerate(expanded_jobs):
        group_key = _quality_group_key(job)
        group_task_index = len(group_entries.get(group_key, ()))
        base_port = manifest.slurm.base_port + plan_index
        metrics_port = base_port + 1000
        base_url = f"http://{job.launch.host}:{base_port}"
        status_path = state_jobs_dir / f"{job.job_id}.json"
        vllm_stdout_log = logs_dir / f"{job.job_id}.vllm.stdout.log"
        vllm_stderr_log = logs_dir / f"{job.job_id}.vllm.stderr.log"
        quality_stdout_log = logs_dir / f"{job.job_id}.quality.stdout.log"
        quality_stderr_log = logs_dir / f"{job.job_id}.quality.stderr.log"
        initial_state = _quality_job_state(
            job=job,
            run_id=run_id,
            status_path=status_path,
            group_key=group_key,
            plan_index=plan_index,
            group_task_index=group_task_index,
            base_port=base_port,
            base_url=base_url,
            vllm_stdout_log=vllm_stdout_log,
            vllm_stderr_log=vllm_stderr_log,
            quality_stdout_log=quality_stdout_log,
            quality_stderr_log=quality_stderr_log,
        )
        _write_json(status_path, initial_state)
        initial_jobs.append(initial_state)
        task_payload = {
            "job_id": job.job_id,
            "run_id": run_id,
            "plan_index": plan_index,
            "group_key": group_key,
            "group_task_index": group_task_index,
            "gpu_count": job.launch.gpu_count,
            "base_port": base_port,
            "metrics_port": metrics_port,
            "base_url": base_url,
            "status_path": str(status_path),
            "result_dir": str(job.result_dir),
            "vllm_stdout_log": str(vllm_stdout_log),
            "vllm_stderr_log": str(vllm_stderr_log),
            "quality_stdout_log": str(quality_stdout_log),
            "quality_stderr_log": str(quality_stderr_log),
            "job": serialize_quality_job(job),
            "initial_state": initial_state,
        }
        group_entries.setdefault(group_key, []).append(task_payload)
        plan_jobs.append(
            {
                "job_id": job.job_id,
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
        slurm_stdout_log = logs_dir / f"slurm-quality-{group_key}-%A_%a.out"
        slurm_stderr_log = logs_dir / f"slurm-quality-{group_key}-%A_%a.err"
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
            render_quality_sbatch_script(group_payload=group_payload, slurm=manifest.slurm),
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
        "run_id": run_id,
        "manifest_path": str(manifest.manifest_path),
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
        "manifest_path": str(manifest.manifest_path),
        "repo_root": str(repo_root),
        "profiler_root": str(profiler_root),
        "python_executable": python_executable,
        "job_count": len(plan_jobs),
        "groups": groups_payload,
        "jobs": plan_jobs,
    }
    _write_json(plan_path, payload)
    return payload


def load_quality_run_plan(run_root: str | Path) -> dict[str, Any]:
    path = Path(run_root).resolve() / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"quality run plan does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("quality run plan is malformed")
    return payload


def submit_quality_run_plan(
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
        command = ["sbatch", "--parsable"]
        if array_spec is not None:
            command.append(f"--array={array_spec}")
        command.append(str(group["script_path"]))
        result = subprocess.run(command, cwd=str(repo_root), capture_output=True, text=True, check=False)
        payload = {
            "group_key": group_key,
            "script_path": str(group["script_path"]),
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


def read_quality_group_task(group_plan_path: str | Path, task_index: int) -> dict[str, Any]:
    payload = _read_json_mapping(Path(group_plan_path).resolve())
    if payload is None:
        raise RuntimeError(f"quality group plan is missing or malformed: {group_plan_path}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError(f"quality group plan does not contain jobs: {group_plan_path}")
    if task_index < 0 or task_index >= len(jobs):
        raise IndexError(f"task index out of range for {group_plan_path}: {task_index}")
    task = jobs[task_index]
    if not isinstance(task, dict):
        raise RuntimeError(f"quality group task is malformed: {group_plan_path}[{task_index}]")
    return task


def render_quality_task_shell(group_plan_path: str | Path, task_index: int) -> str:
    group_payload = _read_json_mapping(Path(group_plan_path).resolve())
    if group_payload is None:
        raise RuntimeError(f"quality group plan is missing or malformed: {group_plan_path}")
    task = read_quality_group_task(group_plan_path, task_index)
    job = deserialize_quality_job(task["job"])
    launch_command = render_launch_command(
        launch=job.launch,
        model=job.model,
        gpu_ids=tuple(range(job.launch.gpu_count)),
        base_port=int(task["base_port"]),
        metrics_port=int(task["metrics_port"]),
    )
    generation_commands = _build_quality_live_generation_commands(
        run_id=str(group_payload["run_id"]),
        job=job,
        python_executable=str(group_payload["python_executable"]),
        base_url=str(task["base_url"]),
        output_dir=Path(str(task["result_dir"])),
    )
    summarize_command = _build_quality_summarize_command(
        run_id=str(group_payload["run_id"]),
        job=job,
        python_executable=str(group_payload["python_executable"]),
        output_dir=Path(str(task["result_dir"])),
    )
    lines = [
        _bash_assign("QUALITY_JOB_ID", job.job_id),
        _bash_assign("RESULT_DIR", str(task["result_dir"])),
        _bash_assign("BASE_URL", str(task["base_url"])),
        _bash_assign("STATUS_PATH", str(task["status_path"])),
        _bash_assign("VLLM_STDOUT", str(task["vllm_stdout_log"])),
        _bash_assign("VLLM_STDERR", str(task["vllm_stderr_log"])),
        _bash_assign("QUALITY_STDOUT", str(task["quality_stdout_log"])),
        _bash_assign("QUALITY_STDERR", str(task["quality_stderr_log"])),
        _bash_assign("READINESS_PATH", job.launch.readiness_path),
        _bash_assign("READINESS_TIMEOUT_S", _number_text(job.launch.readiness_timeout_s)),
        _bash_assign("READINESS_INTERVAL_S", _number_text(job.launch.readiness_interval_s)),
        _bash_array("VLLM_CMD", launch_command),
    ]
    for index, command in enumerate(generation_commands):
        lines.append(_bash_array(f"QUALITY_GENERATION_CMD_{index}", command))
    lines.append(_bash_array("QUALITY_SUMMARIZE_CMD", summarize_command))
    for name, value in sorted(job.launch.env.items()):
        lines.append(_bash_export(name, value))
    return "\n".join(lines)


def render_quality_sbatch_script(*, group_payload: dict[str, Any], slurm: SlurmConfig) -> str:
    group_key = str(group_payload["group_key"])
    gpu_count = int(group_payload["gpu_count"])
    cpus_per_task = _cpus_per_task(gpu_count, slurm=slurm)
    python_executable = str(group_payload["python_executable"])
    repo_root = str(group_payload["repo_root"])
    profiler_root = str(group_payload["profiler_root"])
    group_plan_path = str((Path(str(group_payload["run_root"])) / "groups" / f"{group_key}.json").resolve())
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {_quality_job_name(str(group_payload['run_id']), group_key)}",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --gres=gpu:{gpu_count}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --array={group_payload['array_spec']}",
        f"#SBATCH --output={group_payload['slurm_stdout_log']}",
        f"#SBATCH --error={group_payload['slurm_stderr_log']}",
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
            'eval "$("$PYTHON_BIN" -m slurm_orchestrator.cli emit-quality-task-shell --group-plan "$GROUP_PLAN" --task-index "$TASK_INDEX")"',
            'export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READINESS_TIMEOUT_S}"',
            "",
            "GENERATION_STARTED=0",
            "",
            "terminate_vllm() {",
            '  if [[ -z "${VLLM_PID:-}" ]]; then return; fi',
            '  if ! kill -0 "$VLLM_PID" 2>/dev/null; then return; fi',
            '  kill -- -"$VLLM_PID" 2>/dev/null || true',
            "  for _ in {1..60}; do",
            '    if ! kill -0 "$VLLM_PID" 2>/dev/null; then return; fi',
            "    sleep 1",
            "  done",
            '  kill -KILL -- -"$VLLM_PID" 2>/dev/null || true',
            '  wait "$VLLM_PID" 2>/dev/null || true',
            "}",
            "",
            "cleanup() {",
            "  local exit_code=$?",
            "  terminate_vllm",
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli finalize-quality-task \\',
            '    --group-plan "$GROUP_PLAN" \\',
            '    --task-index "$TASK_INDEX" \\',
            '    --exit-code "$exit_code" \\',
            '    --generation-started "$GENERATION_STARTED"',
            "}",
            "trap cleanup EXIT",
            "",
            'mkdir -p "$(dirname "$STATUS_PATH")" "$(dirname "$VLLM_STDOUT")" "$(dirname "$QUALITY_STDOUT")"',
            ': >"$VLLM_STDOUT"',
            ': >"$VLLM_STDERR"',
            ': >"$QUALITY_STDOUT"',
            ': >"$QUALITY_STDERR"',
            'rm -rf "$RESULT_DIR"',
            'mkdir -p "$RESULT_DIR"',
            "",
            'setsid "${VLLM_CMD[@]}" >>"$VLLM_STDOUT" 2>>"$VLLM_STDERR" &',
            "VLLM_PID=$!",
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli mark-quality-task-running --group-plan "$GROUP_PLAN" --task-index "$TASK_INDEX"'.lstrip(),
            '  "$PYTHON_BIN" -m slurm_orchestrator.cli wait-ready \\'.lstrip(),
            '    --base-url "$BASE_URL" \\'.lstrip(),
            '    --path "$READINESS_PATH" \\'.lstrip(),
            '    --timeout-s "$READINESS_TIMEOUT_S" \\'.lstrip(),
            '    --interval-s "$READINESS_INTERVAL_S" \\'.lstrip(),
            '    --pid "$VLLM_PID"'.lstrip(),
            "",
            "GENERATION_STARTED=1",
            'for cmd_name in "${!QUALITY_GENERATION_CMD_@}"; do',
            '  declare -n generation_cmd="$cmd_name"',
            '  "${generation_cmd[@]}" >>"$QUALITY_STDOUT" 2>>"$QUALITY_STDERR"',
            '  unset -n generation_cmd',
            "done",
            '"${QUALITY_SUMMARIZE_CMD[@]}" >>"$QUALITY_STDOUT" 2>>"$QUALITY_STDERR"',
            "",
        ]
    )
    return "\n".join(lines)


def mark_quality_task_running(group_plan_path: str | Path, task_index: int) -> dict[str, Any]:
    task = read_quality_group_task(group_plan_path, task_index)
    state = _load_quality_state_for_task(task)
    state["status"] = "running"
    state["attempts"] = int(state.get("attempts", 0)) + 1
    _update_quality_slurm_metadata(state, task)
    state["updated_at"] = now_utc_iso()
    _write_json(Path(str(task["status_path"])), state)
    return state


def finalize_quality_task(
    group_plan_path: str | Path,
    task_index: int,
    *,
    exit_code: int,
    generation_started: bool,
) -> dict[str, Any]:
    task = read_quality_group_task(group_plan_path, task_index)
    state = _load_quality_state_for_task(task)
    result_dir = Path(str(task["result_dir"]))
    artifacts = state.setdefault("artifacts", {})
    _fill_quality_artifacts(
        artifacts=artifacts,
        result_dir=result_dir,
        quality_stdout_log=Path(str(task["quality_stdout_log"])),
        quality_stderr_log=Path(str(task["quality_stderr_log"])),
        vllm_stdout_log=Path(str(task["vllm_stdout_log"])),
        vllm_stderr_log=Path(str(task["vllm_stderr_log"])),
    )
    if exit_code == 0 and (result_dir / "responses.jsonl").is_file() and (result_dir / "summary.json").is_file():
        state["status"] = "succeeded"
        state["last_error"] = None
    elif exit_code == 0:
        state["status"] = "failed"
        state["last_error"] = (
            "quality task exited successfully but required artifacts are missing: "
            f"responses_exists={(result_dir / 'responses.jsonl').is_file()}, "
            f"summary_exists={(result_dir / 'summary.json').is_file()}"
        )
    else:
        phase = "generation" if generation_started else "startup"
        state["status"] = "failed"
        state["last_error"] = f"Slurm quality task exited with code {exit_code} during {phase}"
    _update_quality_slurm_metadata(state, task)
    state["updated_at"] = now_utc_iso()
    _write_json(Path(str(task["status_path"])), state)
    return state


def collect_quality_run(run_root: str | Path) -> dict[str, Any]:
    plan = load_quality_run_plan(run_root)
    jobs: list[dict[str, Any]] = []
    latest_update = str(plan.get("created_at", now_utc_iso()))
    for job_entry in plan.get("jobs", []):
        status_path = Path(str(job_entry["status_path"]))
        state = _read_json_mapping(status_path)
        if state is None:
            state = dict(job_entry["initial_state"])
        _reconcile_quality_state_artifacts(state)
        jobs.append(state)
        updated_at = state.get("updated_at")
        if isinstance(updated_at, str) and updated_at > latest_update:
            latest_update = updated_at
    aggregate_state = {
        "run_id": plan.get("run_id"),
        "manifest_path": plan.get("manifest_path"),
        "status": _derive_quality_run_status(jobs),
        "created_at": plan.get("created_at"),
        "updated_at": latest_update,
        "jobs": jobs,
    }
    run_root_path = Path(str(plan["run_root"]))
    _write_json(run_root_path / "state.json", aggregate_state)
    summary = _write_quality_summary(run_root_path, aggregate_state)
    return {"run_root": str(plan["run_root"]), "summary": summary}


def serialize_quality_job(job: QualityExperimentJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "source_index": job.source_index,
        "model": job.model,
        "workload": str(job.workload),
        "workloads": [str(workload) for workload in job.workloads],
        "endpoint": job.endpoint,
        "launch": _launch_to_dict(job.launch),
        "generation": {
            "request_timeout_s": job.generation.request_timeout_s,
            "max_concurrency": job.generation.max_concurrency,
            "concurrency_source": job.generation.concurrency_source,
            "concurrency_mst_fraction": job.generation.concurrency_mst_fraction,
            "preserve_request_order": job.generation.preserve_request_order,
            "response_text_max_chars": job.generation.response_text_max_chars,
            "include_prompt_text": job.generation.include_prompt_text,
            "decoding": job.generation.decoding.to_dict(),
        },
        "hardware": job.hardware.name,
        "probe": {
            "enabled": job.probe.enabled,
            "auto_gpu_count": job.probe.auto_gpu_count,
            "activation_memory_gb": job.probe.activation_memory_gb,
            "memory_safety_factor": job.probe.memory_safety_factor,
            "kv_cache_request_count": job.probe.kv_cache_request_count,
            "default_context_tokens": job.probe.default_context_tokens,
            "model_size_overrides_b": dict(job.probe.model_size_overrides_b),
        },
        "result_dir": str(job.result_dir),
        "model_slug": job.model_slug,
        "shard_id": job.shard_id,
        "server_signature_key": job.server_signature_key,
    }


def deserialize_quality_job(payload: dict[str, Any]) -> QualityExperimentJob:
    from local_orchestrator.models import HardwareConfig, ProbeConfig

    generation = payload["generation"]
    decoding = generation["decoding"]
    return QualityExperimentJob(
        job_id=str(payload["job_id"]),
        source_index=int(payload["source_index"]),
        model=str(payload["model"]),
        workload=Path(str(payload["workload"])),
        workloads=tuple(Path(str(item)) for item in payload.get("workloads", [payload["workload"]])),
        endpoint=str(payload["endpoint"]),
        launch=_launch_from_dict(payload["launch"]),
        generation=QualityGenerationConfig(
            request_timeout_s=float(generation["request_timeout_s"]),
            max_concurrency=(
                None if generation.get("max_concurrency") is None else int(generation["max_concurrency"])
            ),
            concurrency_source=str(generation["concurrency_source"]),
            concurrency_mst_fraction=float(generation["concurrency_mst_fraction"]),
            preserve_request_order=bool(generation["preserve_request_order"]),
            response_text_max_chars=int(generation["response_text_max_chars"]),
            include_prompt_text=bool(generation["include_prompt_text"]),
            decoding=QualityDecodingConfig(
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                top_k=int(decoding["top_k"]),
                min_p=float(decoding["min_p"]),
                n=int(decoding["n"]),
                max_tokens=int(decoding["max_tokens"]),
                max_tokens_policy=str(decoding["max_tokens_policy"]),
                prompt_token_buffer=int(decoding["prompt_token_buffer"]),
                extra_body=dict(decoding.get("extra_body") or {}),
            ),
        ),
        hardware=HardwareConfig(name=str(payload["hardware"])),
        probe=ProbeConfig(**payload["probe"]),
        result_dir=Path(str(payload["result_dir"])),
        model_slug=str(payload["model_slug"]),
        shard_id=str(payload["shard_id"]),
        server_signature_key=str(payload["server_signature_key"]),
    )


def _build_quality_live_generation_commands(
    *,
    run_id: str,
    job: QualityExperimentJob,
    python_executable: str,
    base_url: str,
    output_dir: Path,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        _build_quality_live_generation_command(
            run_id=run_id,
            job=job,
            workload=workload,
            python_executable=python_executable,
            base_url=base_url,
            output_dir=_quality_shard_output_dir(output_dir, workload),
        )
        for workload in job.workloads
    )


def _build_quality_live_generation_command(
    *,
    run_id: str,
    job: QualityExperimentJob,
    workload: Path,
    python_executable: str,
    base_url: str,
    output_dir: Path,
) -> tuple[str, ...]:
    max_concurrency = _resolved_quality_concurrency(job.generation)
    decoding = job.generation.decoding
    command = [
        python_executable,
        "-m",
        "output_quality_profiler.cli",
        "run-live-generation",
        "--job-id",
        job.job_id,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--workload",
        str(workload),
        "--model",
        job.model,
        "--base-url",
        base_url,
        "--endpoint",
        job.endpoint,
        "--request-timeout-s",
        str(job.generation.request_timeout_s),
        "--max-concurrency",
        str(max_concurrency),
        "--response-text-max-chars",
        str(job.generation.response_text_max_chars),
        "--temperature",
        str(decoding.temperature),
        "--top-p",
        str(decoding.top_p),
        "--top-k",
        str(decoding.top_k),
        "--min-p",
        str(decoding.min_p),
        "--n",
        str(decoding.n),
        "--max-tokens",
        str(decoding.max_tokens),
        "--max-tokens-policy",
        decoding.max_tokens_policy,
        "--prompt-token-buffer",
        str(decoding.prompt_token_buffer),
        "--extra-body-json",
        json.dumps(decoding.extra_body, sort_keys=True),
        "--force",
    ]
    if job.launch.max_model_len is not None:
        command.extend(["--serving-max-model-len", str(job.launch.max_model_len)])
    return tuple(command)


def _build_quality_summarize_command(
    *,
    run_id: str,
    job: QualityExperimentJob,
    python_executable: str,
    output_dir: Path,
) -> tuple[str, ...]:
    command = [
        python_executable,
        "-m",
        "output_quality_profiler.cli",
        "summarize-live-generation",
        "--job-id",
        job.job_id,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--model",
        job.model,
    ]
    for workload in job.workloads:
        command.extend(["--shard-output-dir", str(_quality_shard_output_dir(output_dir, workload))])
    return tuple(command)


def _quality_shard_output_dir(output_dir: Path, workload: Path) -> Path:
    return output_dir / "shards" / _dataset_slug(workload)


def _resolved_quality_concurrency(generation: QualityGenerationConfig) -> int:
    if generation.max_concurrency is not None:
        return generation.max_concurrency
    raise ValueError(
        "quality Slurm jobs require generation.max_concurrency to be resolved "
        "before submission; set it explicitly or generate it from MST results"
    )


def _quality_job_state(
    *,
    job: QualityExperimentJob,
    run_id: str,
    status_path: Path,
    group_key: str,
    plan_index: int,
    group_task_index: int,
    base_port: int,
    base_url: str,
    vllm_stdout_log: Path,
    vllm_stderr_log: Path,
    quality_stdout_log: Path,
    quality_stderr_log: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "job_id": job.job_id,
        "status": "planned",
        "model": job.model,
        "workload": str(job.workload),
        "workloads": [str(workload) for workload in job.workloads],
        "endpoint": job.endpoint,
        "result_dir": str(job.result_dir),
        "server_signature_key": job.server_signature_key,
        "gpu_count": job.launch.gpu_count,
        "tensor_parallel_size": job.launch.tensor_parallel_size,
        "max_model_len": job.launch.max_model_len,
        "generation": {
            "request_timeout_s": job.generation.request_timeout_s,
            "max_concurrency": _resolved_quality_concurrency(job.generation),
            "concurrency_source": job.generation.concurrency_source,
            "concurrency_mst_fraction": job.generation.concurrency_mst_fraction,
            "response_text_max_chars": job.generation.response_text_max_chars,
            "decoding": job.generation.decoding.to_dict(),
        },
        "attempts": 0,
        "last_error": None,
        "base_url": base_url,
        "artifacts": {
            "responses_jsonl": None,
            "failed_requests_jsonl": None,
            "summary_json": None,
            "quality_stdout_log": str(quality_stdout_log),
            "quality_stderr_log": str(quality_stderr_log),
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


def _load_quality_state_for_task(task: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(str(task["status_path"]))
    payload = _read_json_mapping(status_path)
    if payload is not None:
        return payload
    initial_state = task.get("initial_state")
    if not isinstance(initial_state, dict):
        raise RuntimeError(f"quality task is missing initial state: {task.get('job_id')}")
    return dict(initial_state)


def _update_quality_slurm_metadata(state: dict[str, Any], task: dict[str, Any]) -> None:
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
        }
    )


def _fill_quality_artifacts(
    *,
    artifacts: dict[str, Any],
    result_dir: Path,
    quality_stdout_log: Path,
    quality_stderr_log: Path,
    vllm_stdout_log: Path,
    vllm_stderr_log: Path,
) -> None:
    artifacts["responses_jsonl"] = str(result_dir / "responses.jsonl") if (result_dir / "responses.jsonl").is_file() else None
    artifacts["failed_requests_jsonl"] = (
        str(result_dir / "failed_requests.jsonl") if (result_dir / "failed_requests.jsonl").is_file() else None
    )
    artifacts["summary_json"] = str(result_dir / "summary.json") if (result_dir / "summary.json").is_file() else None
    artifacts["quality_stdout_log"] = str(quality_stdout_log)
    artifacts["quality_stderr_log"] = str(quality_stderr_log)
    artifacts["vllm_stdout_log"] = str(vllm_stdout_log)
    artifacts["vllm_stderr_log"] = str(vllm_stderr_log)


def _reconcile_quality_state_artifacts(state: dict[str, Any]) -> None:
    result_dir = Path(str(state.get("result_dir", "")))
    if not result_dir:
        return
    artifacts = state.setdefault("artifacts", {})
    if (result_dir / "responses.jsonl").is_file() and (result_dir / "summary.json").is_file():
        artifacts["responses_jsonl"] = str(result_dir / "responses.jsonl")
        artifacts["summary_json"] = str(result_dir / "summary.json")
        failed = result_dir / "failed_requests.jsonl"
        artifacts["failed_requests_jsonl"] = str(failed) if failed.is_file() else None
        state["status"] = "succeeded"
        state["last_error"] = None


def _derive_quality_run_status(jobs: list[dict[str, Any]]) -> str:
    statuses = {str(job.get("status", "planned")) for job in jobs}
    if "running" in statuses or "planned" in statuses:
        return "running"
    if "failed" in statuses:
        return "failed"
    if statuses <= {"succeeded", "skipped"}:
        return "succeeded"
    return "planned"


def _write_quality_summary(run_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    jobs = list(state.get("jobs", []))
    counts = dict(sorted({status: sum(1 for job in jobs if job.get("status") == status) for status in {str(job.get("status")) for job in jobs}}.items()))
    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    for job in jobs:
        summary_path = (job.get("artifacts") or {}).get("summary_json")
        if isinstance(summary_path, str) and summary_path:
            payload = _read_json_mapping(Path(summary_path))
            if payload is not None:
                total_requests += int(payload.get("total_requests", 0))
                successful_requests += int(payload.get("successful_requests", 0))
                failed_requests += int(payload.get("failed_requests", 0))
    summary = {
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "counts": counts,
        "job_count": len(jobs),
        "aggregate": {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
        },
        "jobs": jobs,
    }
    _write_json(run_root / "summary.json", summary)
    lines = [
        f"# Quality Run {state.get('run_id')}",
        "",
        f"- Status: {state.get('status')}",
        f"- Jobs: {len(jobs)}",
        f"- Requests: {total_requests}",
        f"- Successful requests: {successful_requests}",
        f"- Failed requests: {failed_requests}",
    ]
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _quality_group_key(job: QualityExperimentJob) -> str:
    return f"gpu{job.launch.gpu_count}"


def _quality_job_name(run_id: str, group_key: str) -> str:
    prefix = f"quality-{run_id}-{group_key}"
    return prefix if len(prefix) <= 120 else prefix[:120]


def _launch_to_dict(launch: LaunchConfig) -> dict[str, Any]:
    return {
        "template": None if launch.template is None else list(launch.template),
        "executable": launch.executable,
        "extra_args": list(launch.extra_args),
        "env": dict(launch.env),
        "tensor_parallel_size": launch.tensor_parallel_size,
        "gpu_count": launch.gpu_count,
        "dtype": launch.dtype,
        "quantization": launch.quantization,
        "tokenizer_mode": launch.tokenizer_mode,
        "gpu_memory_utilization": launch.gpu_memory_utilization,
        "max_model_len": launch.max_model_len,
        "max_num_seqs": launch.max_num_seqs,
        "max_num_batched_tokens": launch.max_num_batched_tokens,
        "host": launch.host,
        "readiness_path": launch.readiness_path,
        "readiness_timeout_s": launch.readiness_timeout_s,
        "readiness_interval_s": launch.readiness_interval_s,
    }


def _launch_from_dict(payload: dict[str, Any]) -> LaunchConfig:
    return LaunchConfig(
        template=None if payload.get("template") is None else tuple(payload["template"]),
        executable=str(payload.get("executable", "vllm")),
        extra_args=tuple(payload.get("extra_args", ())),
        env=dict(payload.get("env", {})),
        tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
        gpu_count=int(payload.get("gpu_count", 1)),
        dtype=payload.get("dtype"),
        quantization=payload.get("quantization"),
        tokenizer_mode=payload.get("tokenizer_mode"),
        gpu_memory_utilization=payload.get("gpu_memory_utilization"),
        max_model_len=payload.get("max_model_len"),
        max_num_seqs=payload.get("max_num_seqs"),
        max_num_batched_tokens=payload.get("max_num_batched_tokens"),
        host=str(payload.get("host", "127.0.0.1")),
        readiness_path=str(payload.get("readiness_path", "/v1/models")),
        readiness_timeout_s=float(payload.get("readiness_timeout_s", 300.0)),
        readiness_interval_s=float(payload.get("readiness_interval_s", 2.0)),
    )
