from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .models import ActiveServer, ExpandedExperimentJob, SearchExecutionResult


class MSTAdapterError(RuntimeError):
    pass


def build_search_command(
    *,
    job: ExpandedExperimentJob,
    base_url: str,
    metrics_url: str | None = None,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    resolved_python = python_executable or sys.executable
    resolved_metrics_url = metrics_url or f"{base_url.rstrip('/')}/metrics"
    command: list[str] = [
        resolved_python,
        "-m",
        "llm_mst_finder.cli",
        "search",
        "--search-id",
        job.experiment_id,
        "--search-mode",
        job.search.search_mode,
        "--output-dir",
        str(job.result_dir),
        "--base-url",
        base_url,
        "--endpoint",
        job.endpoint,
        "--model",
        job.model,
        "--workload",
        str(job.workload),
        "--trial-min-duration-s",
        f"{job.search.trial_min_duration_s}",
        "--rate-precision",
        f"{job.search.rate_precision}",
        "--initial-request-rate",
        f"{job.search.initial_request_rate}",
        "--max-binary-steps",
        str(job.search.max_binary_steps),
        "--max-bracket-trials",
        str(job.search.max_bracket_trials),
        "--closed-loop-initial-concurrency",
        str(job.search.closed_loop_initial_concurrency),
        "--closed-loop-min-trials",
        str(job.search.closed_loop_min_trials),
        "--max-closed-loop-concurrency",
        str(job.search.max_closed_loop_concurrency),
        "--closed-loop-plateau-relative-gain",
        f"{job.search.closed_loop_plateau_relative_gain}",
        "--metrics-url",
        resolved_metrics_url,
        "--metrics-interval-s",
        f"{job.search.metrics_interval_s}",
        "--window-s",
        f"{job.search.window_s}",
        "--ttft-slo-ms",
        "none" if job.search.ttft_slo_ms is None else f"{job.search.ttft_slo_ms}",
        "--tpot-slo-ms",
        "none" if job.search.tpot_slo_ms is None else f"{job.search.tpot_slo_ms}",
        "--ttft-slo-field",
        job.search.ttft_slo_field,
        "--tpot-slo-field",
        job.search.tpot_slo_field,
    ]
    if job.search.trial_max_duration_s is not None:
        command.extend(["--trial-max-duration-s", f"{job.search.trial_max_duration_s}"])
    if job.search.final_confirmation_duration_s is not None:
        command.extend(
            ["--final-confirmation-duration-s", f"{job.search.final_confirmation_duration_s}"]
        )
    if job.search.max_request_rate is not None:
        command.extend(["--max-request-rate", f"{job.search.max_request_rate}"])
    if job.launch.max_num_seqs is not None:
        command.extend(["--max-num-seqs", f"{job.launch.max_num_seqs}"])
    elif job.search.max_num_seqs is not None:
        command.extend(["--max-num-seqs", f"{job.search.max_num_seqs}"])
    if job.launch.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", f"{job.launch.max_num_batched_tokens}"])
    elif job.search.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", f"{job.search.max_num_batched_tokens}"])
    if job.server_metadata_file is not None:
        command.extend(["--server-metadata-file", str(job.server_metadata_file)])
    return tuple(command)


def build_report_command(
    *,
    job: ExpandedExperimentJob,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    resolved_python = python_executable or sys.executable
    return (
        resolved_python,
        "-m",
        "llm_mst_finder.cli",
        "report",
        "--result-dir",
        str(job.result_dir),
    )


class MSTSearchAdapter:
    def __init__(
        self,
        *,
        python_executable: str | None = None,
        profiler_root: Path | None = None,
        run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._python_executable = python_executable or sys.executable
        self._profiler_root = profiler_root or Path(__file__).resolve().parents[1]
        self._run_command = run_command or _default_run_command

    def invoke(
        self,
        *,
        job: ExpandedExperimentJob,
        server: ActiveServer,
        logs_dir: Path,
    ) -> SearchExecutionResult:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / f"{job.experiment_id}.mst.stdout.log"
        stderr_log = logs_dir / f"{job.experiment_id}.mst.stderr.log"

        if job.result_dir.exists():
            shutil.rmtree(job.result_dir)
        job.result_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        profiler_root_str = str(self._profiler_root)
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{profiler_root_str}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = profiler_root_str

        commands: list[tuple[str, ...]] = []
        search_cmd = self._build_search_command(job=job, server=server)
        commands.append(search_cmd)
        report_cmd = self._build_report_command(job=job)
        commands.append(report_cmd)

        command_stdout_blocks: list[str] = []
        command_stderr_blocks: list[str] = []
        for command in commands:
            result = self._run_command(
                command,
                env=env,
                cwd=str(self._profiler_root.parent),
            )
            command_text = " ".join(command)
            command_stdout_blocks.append(f"$ {command_text}\n{result.stdout}")
            command_stderr_blocks.append(f"$ {command_text}\n{result.stderr}")
            if result.returncode != 0:
                self._write_log(stdout_log, "\n\n".join(command_stdout_blocks))
                self._write_log(stderr_log, "\n\n".join(command_stderr_blocks))
                return SearchExecutionResult(
                    success=False,
                    return_code=result.returncode,
                    commands=tuple(commands),
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    search_trace_path=None,
                    final_report_json_path=None,
                    final_report_md_path=None,
                    error=(
                        f"MST command failed with exit code {result.returncode}: "
                        f"{command_text}"
                    ),
                )

        self._write_log(stdout_log, "\n\n".join(command_stdout_blocks))
        self._write_log(stderr_log, "\n\n".join(command_stderr_blocks))

        search_trace = job.result_dir / "search_trace.json"
        final_report_json = job.result_dir / "final_report.json"
        final_report_md = job.result_dir / "final_report.md"

        if not search_trace.is_file() or not final_report_json.is_file():
            return SearchExecutionResult(
                success=False,
                return_code=0,
                commands=tuple(commands),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                search_trace_path=search_trace if search_trace.is_file() else None,
                final_report_json_path=final_report_json if final_report_json.is_file() else None,
                final_report_md_path=final_report_md if final_report_md.is_file() else None,
                error=(
                    "MST search completed without required artifacts: "
                    f"search_trace_exists={search_trace.is_file()}, "
                    f"final_report_json_exists={final_report_json.is_file()}"
                ),
            )

        return SearchExecutionResult(
            success=True,
            return_code=0,
            commands=tuple(commands),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            search_trace_path=search_trace,
            final_report_json_path=final_report_json,
            final_report_md_path=final_report_md if final_report_md.is_file() else None,
            error=None,
        )

    def _build_search_command(self, *, job: ExpandedExperimentJob, server: ActiveServer) -> tuple[str, ...]:
        return build_search_command(
            job=job,
            base_url=server.base_url,
            metrics_url=f"{server.base_url}/metrics",
            python_executable=self._python_executable,
        )

    def _build_report_command(self, *, job: ExpandedExperimentJob) -> tuple[str, ...]:
        return build_report_command(
            job=job,
            python_executable=self._python_executable,
        )

    @staticmethod
    def _write_log(path: Path, content: str) -> None:
        path.write_text(content + "\n", encoding="utf-8")


def _default_run_command(
    command: tuple[str, ...],
    *,
    env: dict[str, str],
    cwd: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
