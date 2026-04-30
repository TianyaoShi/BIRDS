from __future__ import annotations

from dataclasses import replace
import json
from io import StringIO
from pathlib import Path
import time

from local_orchestrator.models import (
    ActiveServer,
    ExpandedExperimentJob,
    HardwareConfig,
    LaunchConfig,
    RetryPolicy,
    RunConfig,
    ResourceProbeResult,
    SearchConfig,
    SearchExecutionResult,
)
from local_orchestrator.resources import GPULeaseManager, PortAllocator
from local_orchestrator.scheduler import OrchestratorScheduler
from local_orchestrator.state_store import RunStateStore


class _FakeProcess:
    def __init__(self) -> None:
        self._return_code: int | None = None

    def poll(self):
        return self._return_code

    def terminate(self) -> None:
        self._return_code = 0

    def kill(self) -> None:
        self._return_code = -9

    def wait(self, timeout=None):
        del timeout
        return self._return_code


class StubLifecycle:
    def __init__(self, *, startup_failures: int = 0) -> None:
        self.startup_failures = startup_failures
        self._active_server: ActiveServer | None = None

    @property
    def active_server(self) -> ActiveServer | None:
        return self._active_server

    def ensure_server(
        self,
        *,
        job: ExpandedExperimentJob,
        gpu_ids: tuple[int, ...],
        ports,
        runtime_signature: str,
        logs_dir,
        force_restart: bool = False,
    ) -> ActiveServer:
        del force_restart
        if self.startup_failures > 0:
            self.startup_failures -= 1
            raise RuntimeError("startup failed")
        if self._active_server is not None and self._active_server.runtime_signature == runtime_signature:
            return self._active_server
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        process = _FakeProcess()
        server = ActiveServer(
            reuse_key=job.server_signature_key,
            runtime_signature=runtime_signature,
            model=job.model,
            endpoint=job.endpoint,
            gpu_id=gpu_ids[0],
            gpu_ids=gpu_ids,
            base_port=ports.base_port,
            metrics_port=ports.metrics_port,
            command=("fake",),
            base_url=f"http://127.0.0.1:{ports.base_port}",
            stdout_log=logs_dir / "stub.stdout.log",
            stderr_log=logs_dir / "stub.stderr.log",
            process=process,
            stdout_handle=StringIO(),
            stderr_handle=StringIO(),
        )
        self._active_server = server
        return server

    def is_ready(self, server: ActiveServer, *, timeout_s: float = 2.0) -> bool:
        del timeout_s
        return server.process.poll() is None

    def stop_active_server(self, *, reason: str) -> None:
        del reason
        if self._active_server is None:
            return
        self._active_server.process.terminate()
        self._active_server = None

    def shutdown(self) -> None:
        self.stop_active_server(reason="shutdown")


class StubAdapter:
    def __init__(self, outcomes: dict[str, list[bool]]) -> None:
        self.outcomes = {key: list(values) for key, values in outcomes.items()}
        self.calls: list[str] = []

    def invoke(self, *, job: ExpandedExperimentJob, server: ActiveServer, logs_dir: Path) -> SearchExecutionResult:
        del server
        self.calls.append(job.experiment_id)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = logs_dir / f"{job.experiment_id}.stdout.log"
        stderr_log = logs_dir / f"{job.experiment_id}.stderr.log"
        stdout_log.write_text("stdout\n", encoding="utf-8")
        stderr_log.write_text("stderr\n", encoding="utf-8")

        remaining = self.outcomes.get(job.experiment_id, [True])
        success = remaining.pop(0) if remaining else True
        self.outcomes[job.experiment_id] = remaining

        if success:
            job.result_dir.mkdir(parents=True, exist_ok=True)
            search_trace = job.result_dir / "search_trace.json"
            final_report = job.result_dir / "final_report.json"
            search_trace.write_text("{}\n", encoding="utf-8")
            final_report.write_text("{}\n", encoding="utf-8")
            return SearchExecutionResult(
                success=True,
                return_code=0,
                commands=(("fake-search",),),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                search_trace_path=search_trace,
                final_report_json_path=final_report,
                final_report_md_path=None,
                error=None,
            )

        return SearchExecutionResult(
            success=False,
            return_code=1,
            commands=(("fake-search",),),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            search_trace_path=None,
            final_report_json_path=None,
            final_report_md_path=None,
            error="simulated search failure",
        )


def _make_job(
    tmp_path: Path,
    *,
    experiment_id: str,
    signature: str,
    launch: LaunchConfig | None = None,
    probe: ResourceProbeResult | None = None,
) -> ExpandedExperimentJob:
    workload = tmp_path / f"{experiment_id}.yaml"
    workload.write_text("name: stub\n", encoding="utf-8")
    return ExpandedExperimentJob(
        experiment_id=experiment_id,
        source_index=0,
        model="google/gemma-4-E4B-it",
        workload=workload,
        endpoint="/v1/chat/completions",
        launch=launch or LaunchConfig(),
        search=SearchConfig(),
        hardware=HardwareConfig(),
        probe=probe,
        result_dir=tmp_path / "results" / experiment_id,
        model_slug="gemma-4-e4b-it",
        dataset_slug="dataset",
        server_config_slug="server-abc",
        server_signature_key=signature,
        server_metadata_file=None,
    )


def test_scheduler_retries_and_marks_terminal_states(tmp_path: Path) -> None:
    jobs = [
        _make_job(tmp_path, experiment_id="job-success-after-retry", signature="sig-a"),
        _make_job(tmp_path, experiment_id="job-fails", signature="sig-b"),
    ]
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0, 1, 2, 3),
        max_active_gpus=3,
        retry=RetryPolicy(startup_attempts=2, search_attempts=2),
    )

    lifecycle = StubLifecycle(startup_failures=1)
    adapter = StubAdapter(
        {
            "job-success-after-retry": [False, True],
            "job-fails": [False, False],
        }
    )
    state_store = RunStateStore(tmp_path / "run")
    state = state_store.initialize_new(
        run_id="run-1",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=jobs,
    )

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=3),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
    )
    summary = scheduler.run(jobs=jobs, state=state, resume=False)

    assert summary["counts"]["succeeded"] == 1
    assert summary["counts"]["failed"] == 1

    final_state = state_store.load()
    success_job = state_store.find_job(final_state, "job-success-after-retry")
    failed_job = state_store.find_job(final_state, "job-fails")

    assert success_job["status"] == "succeeded"
    assert int(success_job["attempts"]["search"]) == 2
    assert int(success_job["attempts"]["startup"]) >= 2

    assert failed_job["status"] == "failed"
    assert int(failed_job["attempts"]["search"]) == 2


def test_scheduler_resume_reconciles_existing_artifacts(tmp_path: Path) -> None:
    job = _make_job(tmp_path, experiment_id="job-existing", signature="sig-a")
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0, 1, 2, 3),
        max_active_gpus=3,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )

    lifecycle = StubLifecycle(startup_failures=0)
    adapter = StubAdapter({"job-existing": [True]})
    state_store = RunStateStore(tmp_path / "resume-run")
    state = state_store.initialize_new(
        run_id="run-resume",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[job],
    )

    state_job = state_store.find_job(state, "job-existing")
    state_job["status"] = "running"
    state_store.save(state)

    job.result_dir.mkdir(parents=True, exist_ok=True)
    (job.result_dir / "search_trace.json").write_text("{}\n", encoding="utf-8")
    (job.result_dir / "final_report.json").write_text("{}\n", encoding="utf-8")

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=3),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
    )
    summary = scheduler.run(jobs=[job], state=state_store.load(), resume=True)

    assert summary["counts"]["succeeded"] == 1
    assert adapter.calls == []


def test_scheduler_resume_does_not_reconcile_failed_job_from_existing_artifacts(tmp_path: Path) -> None:
    job = _make_job(tmp_path, experiment_id="job-failed-existing", signature="sig-a")
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0,),
        max_active_gpus=1,
        keep_one_gpu_spare=False,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )
    lifecycle = StubLifecycle(startup_failures=0)
    adapter = StubAdapter({"job-failed-existing": [True]})
    state_store = RunStateStore(tmp_path / "resume-failed-existing-run")
    state = state_store.initialize_new(
        run_id="run-resume-failed-existing",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[job],
    )

    state_job = state_store.find_job(state, "job-failed-existing")
    state_job["status"] = "failed"
    state_job["last_error"] = "old report failure"
    state_store.save(state)

    job.result_dir.mkdir(parents=True, exist_ok=True)
    (job.result_dir / "search_trace.json").write_text("{}\n", encoding="utf-8")
    (job.result_dir / "final_report.json").write_text("{}\n", encoding="utf-8")

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=1),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
    )
    summary = scheduler.run(jobs=[job], state=state_store.load(), resume=True)

    assert summary["counts"]["succeeded"] == 1
    assert adapter.calls == ["job-failed-existing"]


def test_scheduler_resume_refreshes_non_succeeded_job_plan(tmp_path: Path) -> None:
    original_job = _make_job(
        tmp_path,
        experiment_id="job-refresh",
        signature="sig-old",
        launch=LaunchConfig(max_model_len=32768),
    )
    refreshed_job = _make_job(
        tmp_path,
        experiment_id="job-refresh",
        signature="sig-new",
        launch=LaunchConfig(max_model_len=4096),
    )
    refreshed_job = replace(
        refreshed_job,
        result_dir=tmp_path / "results" / "job-refresh-new",
        server_config_slug="server-new",
        server_signature_key="sig-new",
    )

    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0,),
        max_active_gpus=1,
        keep_one_gpu_spare=False,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )
    lifecycle = StubLifecycle(startup_failures=0)
    adapter = StubAdapter({"job-refresh": [True]})
    state_store = RunStateStore(tmp_path / "resume-refresh-run")
    state = state_store.initialize_new(
        run_id="run-resume-refresh",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[original_job],
    )
    state_job = state_store.find_job(state, "job-refresh")
    state_job["status"] = "failed"
    state_job["last_error"] = "old failure"
    state_job["attempts"] = {"startup": 2, "search": 2}
    state_store.save(state)

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=1),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
    )
    summary = scheduler.run(jobs=[refreshed_job], state=state_store.load(), resume=True)

    assert summary["counts"]["succeeded"] == 1
    assert adapter.calls == ["job-refresh"]

    final_job_state = state_store.find_job(state_store.load(), "job-refresh")
    assert final_job_state["result_dir"] == str(refreshed_job.result_dir)
    assert final_job_state["server_signature_key"] == "sig-new"
    assert final_job_state["max_model_len"] == 4096
    assert int(final_job_state["attempts"]["startup"]) == 3
    assert int(final_job_state["attempts"]["search"]) == 3


def test_scheduler_parallel_uses_multiple_slots_and_gpu_ids(tmp_path: Path) -> None:
    class RecordingLifecycle(StubLifecycle):
        def __init__(self) -> None:
            super().__init__(startup_failures=0)
            self.ensure_calls: list[tuple[str, int]] = []

        def ensure_server(
            self,
            *,
            job: ExpandedExperimentJob,
            gpu_ids: tuple[int, ...],
            ports,
            runtime_signature: str,
            logs_dir,
            force_restart: bool = False,
        ) -> ActiveServer:
            self.ensure_calls.append((job.experiment_id, gpu_ids[0]))
            return super().ensure_server(
                job=job,
                gpu_ids=gpu_ids,
                ports=ports,
                runtime_signature=runtime_signature,
                logs_dir=logs_dir,
                force_restart=force_restart,
            )

    class DelayedSuccessAdapter(StubAdapter):
        def invoke(self, *, job: ExpandedExperimentJob, server: ActiveServer, logs_dir: Path) -> SearchExecutionResult:
            time.sleep(0.05)
            return super().invoke(job=job, server=server, logs_dir=logs_dir)

    jobs = [
        _make_job(tmp_path, experiment_id="job-a", signature="sig-a"),
        _make_job(tmp_path, experiment_id="job-b", signature="sig-b"),
    ]
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0, 1, 2, 3),
        max_active_gpus=2,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )

    base_lifecycle = RecordingLifecycle()
    lifecycles: list[RecordingLifecycle] = [base_lifecycle]

    def lifecycle_factory() -> RecordingLifecycle:
        lifecycle = RecordingLifecycle()
        lifecycles.append(lifecycle)
        return lifecycle

    adapter = DelayedSuccessAdapter({"job-a": [True], "job-b": [True]})
    state_store = RunStateStore(tmp_path / "parallel-run")
    state = state_store.initialize_new(
        run_id="run-parallel",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=jobs,
    )

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=2),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=base_lifecycle,
        adapter=adapter,
        state_store=state_store,
        lifecycle_factory=lifecycle_factory,
    )
    summary = scheduler.run(jobs=jobs, state=state, resume=False)

    assert summary["counts"]["succeeded"] == 2
    used_gpu_ids = sorted(
        {
            gpu_id
            for lifecycle in lifecycles
            for _, gpu_id in lifecycle.ensure_calls
        }
    )
    assert used_gpu_ids == [0, 1]


def test_scheduler_force_rerun_resets_and_reexecutes(tmp_path: Path) -> None:
    job = _make_job(tmp_path, experiment_id="job-force", signature="sig-force")
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0, 1, 2, 3),
        max_active_gpus=1,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )

    lifecycle = StubLifecycle(startup_failures=0)
    adapter = StubAdapter({"job-force": [True]})
    state_store = RunStateStore(tmp_path / "force-run")
    state = state_store.initialize_new(
        run_id="run-force",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[job],
    )

    job.result_dir.mkdir(parents=True, exist_ok=True)
    marker_path = job.result_dir / "old-marker.txt"
    marker_path.write_text("old-result\n", encoding="utf-8")
    search_trace_path = job.result_dir / "search_trace.json"
    search_trace_path.write_text("{}\n", encoding="utf-8")
    final_report_path = job.result_dir / "final_report.json"
    final_report_path.write_text("{}\n", encoding="utf-8")

    prior_job_state = state_store.find_job(state, "job-force")
    prior_job_state["status"] = "succeeded"
    prior_job_state["result_dir"] = str(tmp_path / "results" / "old-signature")
    old_signature_dir = Path(prior_job_state["result_dir"])
    old_signature_dir.mkdir(parents=True, exist_ok=True)
    (old_signature_dir / "old-trace.json").write_text("{}\n", encoding="utf-8")
    prior_job_state["attempts"] = {"startup": 7, "search": 7}
    prior_job_state["artifacts"] = {
        "search_trace": str(search_trace_path),
        "final_report_json": str(final_report_path),
        "final_report_md": None,
        "stdout_log": None,
        "stderr_log": None,
    }
    state_store.save(state)

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=1),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=lifecycle,
        adapter=adapter,
        state_store=state_store,
    )
    summary = scheduler.run(jobs=[job], state=state_store.load(), resume=True, force=True)

    assert summary["counts"]["succeeded"] == 1
    assert adapter.calls == ["job-force"]

    final_state = state_store.load()
    final_job_state = state_store.find_job(final_state, "job-force")
    assert final_job_state["status"] == "succeeded"
    assert final_job_state["result_dir"] == str(job.result_dir)
    assert int(final_job_state["attempts"]["startup"]) == 1
    assert int(final_job_state["attempts"]["search"]) == 1
    assert marker_path.exists() is False
    assert old_signature_dir.exists() is False


def test_scheduler_does_not_block_on_probe_gpu_estimate(tmp_path: Path) -> None:
    probe = ResourceProbeResult(
        hardware_name="l40",
        gpu_memory_gb=48,
        model_params_b=70,
        estimated_weight_gb=130.0,
        estimated_activation_gb=4.0,
        estimated_kv_cache_gb=8.0,
        estimated_required_gb=170.0,
        usable_memory_per_gpu_gb=43.2,
        required_gpu_count=4,
        context_tokens=4096,
        warnings=(),
    )
    job = _make_job(
        tmp_path,
        experiment_id="job-preflight",
        signature="sig-preflight",
        launch=LaunchConfig(gpu_count=1, tensor_parallel_size=1),
        probe=probe,
    )
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0, 1, 2, 3),
        max_active_gpus=4,
        keep_one_gpu_spare=False,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )
    adapter = StubAdapter({"job-preflight": [True]})
    state_store = RunStateStore(tmp_path / "preflight-run")
    state = state_store.initialize_new(
        run_id="run-preflight",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[job],
    )

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=4),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=StubLifecycle(),
        adapter=adapter,
        state_store=state_store,
    )
    summary = scheduler.run(jobs=[job], state=state, resume=False)

    assert summary["counts"]["succeeded"] == 1
    assert adapter.calls == ["job-preflight"]
    final_job = state_store.find_job(state_store.load(), "job-preflight")
    assert final_job["last_error"] is None


def test_scheduler_marks_adapter_exceptions_failed(tmp_path: Path) -> None:
    class RaisingAdapter:
        def invoke(self, *, job: ExpandedExperimentJob, server: ActiveServer, logs_dir: Path):
            del job, server, logs_dir
            raise RuntimeError("adapter exploded")

    job = _make_job(tmp_path, experiment_id="job-raises", signature="sig-raises")
    run_config = RunConfig(
        output_root=tmp_path / "orchestrator-runs",
        allowed_gpu_ids=(0,),
        max_active_gpus=1,
        keep_one_gpu_spare=False,
        retry=RetryPolicy(startup_attempts=1, search_attempts=1),
    )
    state_store = RunStateStore(tmp_path / "exception-run")
    state = state_store.initialize_new(
        run_id="run-exception",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[job],
    )

    scheduler = OrchestratorScheduler(
        run_config=run_config,
        gpu_manager=GPULeaseManager(allowed_gpu_ids=run_config.allowed_gpu_ids, max_active_gpus=1),
        port_allocator=PortAllocator(base_port_start=8000, base_port_end=8010, metrics_port_offset=1000),
        lifecycle=StubLifecycle(),
        adapter=RaisingAdapter(),
        state_store=state_store,
    )
    summary = scheduler.run(jobs=[job], state=state, resume=False)

    final_job = state_store.find_job(state_store.load(), "job-raises")
    assert summary["counts"]["failed"] == 1
    assert final_job["status"] == "failed"
    assert "adapter exploded" in final_job["last_error"]


def test_state_store_summary_includes_search_result_aggregates(tmp_path: Path) -> None:
    succeeded = _make_job(tmp_path, experiment_id="job-success", signature="sig-a")
    failed = _make_job(tmp_path, experiment_id="job-failed", signature="sig-b")

    state_store = RunStateStore(tmp_path / "summary-run")
    state = state_store.initialize_new(
        run_id="run-summary",
        manifest_path=tmp_path / "manifest.yaml",
        jobs=[succeeded, failed],
    )

    succeeded.result_dir.mkdir(parents=True, exist_ok=True)
    search_trace_payload = {
        "result": {
            "termination_reason": "confirmed_stable",
            "bottleneck_class": "decode_bandwidth",
            "max_no_drift_request_rate": 9.25,
            "max_slo_satisfying_request_rate": 8.75,
        }
    }
    search_trace_path = succeeded.result_dir / "search_trace.json"
    search_trace_path.write_text(json.dumps(search_trace_payload) + "\n", encoding="utf-8")
    final_report_json_path = succeeded.result_dir / "final_report.json"
    final_report_json_path.write_text("{}\n", encoding="utf-8")

    succeeded_state = state_store.find_job(state, "job-success")
    succeeded_state["status"] = "succeeded"
    succeeded_state["attempts"] = {"startup": 1, "search": 1}
    succeeded_state["artifacts"] = {
        "search_trace": str(search_trace_path),
        "final_report_json": str(final_report_json_path),
        "final_report_md": None,
        "stdout_log": None,
        "stderr_log": None,
    }

    failed_state = state_store.find_job(state, "job-failed")
    failed_state["status"] = "failed"
    failed_state["last_error"] = "simulated failure"
    failed_state["attempts"] = {"startup": 2, "search": 2}
    state_store.save(state)

    summary = state_store.write_summary_files(state)
    assert summary["counts"]["succeeded"] == 1
    assert summary["counts"]["failed"] == 1
    assert summary["aggregate"]["termination_reason_counts"] == {"confirmed_stable": 1}
    assert summary["aggregate"]["bottleneck_class_counts"] == {"decode_bandwidth": 1}
    assert summary["aggregate"]["max_no_drift_request_rate"] == {
        "min": 9.25,
        "max": 9.25,
        "mean": 9.25,
    }
    assert summary["aggregate"]["max_slo_satisfying_request_rate"] == {
        "min": 8.75,
        "max": 8.75,
        "mean": 8.75,
    }
    assert summary["aggregate"]["failed_jobs"] == [
        {"experiment_id": "job-failed", "error": "simulated failure"}
    ]

    markdown = state_store.summary_md_path.read_text(encoding="utf-8")
    assert "confirmed_stable" in markdown
    assert "decode_bandwidth" in markdown
