from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess

from local_orchestrator.models import (
    ActiveServer,
    ExpandedExperimentJob,
    HardwareConfig,
    LaunchConfig,
    SearchConfig,
)
from local_orchestrator.mst_adapter import MSTSearchAdapter


class _FakeProcess:
    def poll(self):
        return None


def _make_job(tmp_path: Path) -> ExpandedExperimentJob:
    workload = tmp_path / "workload.yaml"
    workload.write_text("name: stub\n", encoding="utf-8")
    return ExpandedExperimentJob(
        experiment_id="job-a",
        source_index=0,
        model="model-a",
        workload=workload,
        endpoint="/v1/chat/completions",
        launch=LaunchConfig(),
        search=SearchConfig(),
        hardware=HardwareConfig(),
        probe=None,
        result_dir=tmp_path / "results" / "job-a",
        model_slug="model-a",
        dataset_slug="workload",
        server_config_slug="server-a",
        server_signature_key="sig-a",
        server_metadata_file=None,
    )


def _make_server(tmp_path: Path) -> ActiveServer:
    return ActiveServer(
        reuse_key="sig-a",
        runtime_signature="runtime-a",
        model="model-a",
        endpoint="/v1/chat/completions",
        gpu_id=0,
        gpu_ids=(0,),
        base_port=8300,
        metrics_port=9300,
        command=("fake",),
        base_url="http://127.0.0.1:8300",
        stdout_log=tmp_path / "vllm.stdout.log",
        stderr_log=tmp_path / "vllm.stderr.log",
        process=_FakeProcess(),
        stdout_handle=StringIO(),
        stderr_handle=StringIO(),
    )


def test_mst_adapter_cleans_existing_result_dir_before_search(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    server = _make_server(tmp_path)
    job.result_dir.mkdir(parents=True)
    stale_trace = job.result_dir / "search_trace.json"
    stale_trace.write_text("stale\n", encoding="utf-8")

    def run_command(command, *, env, cwd):
        del env, cwd
        if "search" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "search_trace.json").write_text("{}\n", encoding="utf-8")
        if "report" in command:
            result_dir = Path(command[command.index("--result-dir") + 1])
            (result_dir / "final_report.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    adapter = MSTSearchAdapter(
        python_executable="python",
        profiler_root=Path(__file__).resolve().parents[2] / "profiler",
        run_command=run_command,
    )
    result = adapter.invoke(job=job, server=server, logs_dir=tmp_path / "logs")

    assert result.success is True
    assert stale_trace.read_text(encoding="utf-8") == "{}\n"
