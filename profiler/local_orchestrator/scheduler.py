from __future__ import annotations

from typing import Any, Protocol

from .models import ActiveServer, ExpandedExperimentJob, GPULease, PortReservation, RunConfig
from .state_store import RunStateStore
from .utils import runtime_server_signature


class LifecycleProtocol(Protocol):
    @property
    def active_server(self) -> ActiveServer | None:
        ...

    def ensure_server(
        self,
        *,
        job: ExpandedExperimentJob,
        gpu_id: int,
        ports: PortReservation,
        runtime_signature: str,
        logs_dir,
        force_restart: bool = False,
    ) -> ActiveServer:
        ...

    def is_ready(self, server: ActiveServer, *, timeout_s: float = 2.0) -> bool:
        ...

    def stop_active_server(self, *, reason: str) -> None:
        ...

    def shutdown(self) -> None:
        ...


class AdapterProtocol(Protocol):
    def invoke(self, *, job: ExpandedExperimentJob, server: ActiveServer, logs_dir):
        ...


class GPULeaseManagerProtocol(Protocol):
    def acquire(self) -> GPULease:
        ...

    def release(self, lease: GPULease) -> None:
        ...


class PortAllocatorProtocol(Protocol):
    def reserve(self) -> PortReservation:
        ...

    def release(self, reservation: PortReservation) -> None:
        ...


class OrchestratorScheduler:
    def __init__(
        self,
        *,
        run_config: RunConfig,
        gpu_manager: GPULeaseManagerProtocol,
        port_allocator: PortAllocatorProtocol,
        lifecycle: LifecycleProtocol,
        adapter: AdapterProtocol,
        state_store: RunStateStore,
    ) -> None:
        self._run_config = run_config
        self._gpu_manager = gpu_manager
        self._port_allocator = port_allocator
        self._lifecycle = lifecycle
        self._adapter = adapter
        self._state_store = state_store

        self._active_lease: GPULease | None = None
        self._active_ports: PortReservation | None = None
        self._active_reuse_key: str | None = None

    def run(
        self,
        *,
        jobs: list[ExpandedExperimentJob],
        state: dict[str, Any],
        resume: bool,
    ) -> dict[str, Any]:
        if resume:
            self._state_store.reconcile_jobs(state)

        try:
            for job in jobs:
                job_state = self._state_store.find_job(state, job.experiment_id)
                if job_state.get("status") in {"succeeded", "skipped"}:
                    continue
                self._run_single_job(job=job, state=state)
            state["status"] = "completed"
            self._state_store.save(state)
            return self._state_store.write_summary_files(state)
        finally:
            self._release_active_server(reason="scheduler_shutdown")
            self._lifecycle.shutdown()

    def _run_single_job(self, *, job: ExpandedExperimentJob, state: dict[str, Any]) -> None:
        last_error: str | None = None

        for search_attempt in range(1, self._run_config.retry.search_attempts + 1):
            self._state_store.increment_attempt(state, job.experiment_id, kind="search")
            self._state_store.set_job_status(state, experiment_id=job.experiment_id, status="running")
            self._state_store.append_event(
                state,
                event_type="search_attempt",
                experiment_id=job.experiment_id,
                payload={"attempt": search_attempt},
            )

            try:
                server = self._ensure_server_with_startup_retries(
                    job=job,
                    state=state,
                )
            except Exception as exc:
                last_error = f"startup failed before search attempt {search_attempt}: {exc}"
                self._state_store.append_event(
                    state,
                    event_type="startup_failed",
                    experiment_id=job.experiment_id,
                    payload={"attempt": search_attempt, "error": str(exc)},
                )
                self._release_active_server(reason="startup_failed")
                continue

            result = self._adapter.invoke(
                job=job,
                server=server,
                logs_dir=self._state_store.logs_dir,
            )
            if result.success:
                self._state_store.mark_job_succeeded(state, experiment_id=job.experiment_id, result=result)
                self._state_store.append_event(
                    state,
                    event_type="job_succeeded",
                    experiment_id=job.experiment_id,
                    payload={"attempt": search_attempt, "result_dir": str(job.result_dir)},
                )
                return

            last_error = result.error or f"search command failed with exit code {result.return_code}"
            self._state_store.append_event(
                state,
                event_type="search_failed",
                experiment_id=job.experiment_id,
                payload={
                    "attempt": search_attempt,
                    "return_code": result.return_code,
                    "error": last_error,
                },
            )
            self._release_active_server(reason="search_retry_restart")

        self._state_store.mark_job_failed(
            state,
            experiment_id=job.experiment_id,
            error=last_error or "search failed after retries",
        )
        self._state_store.append_event(
            state,
            event_type="job_failed",
            experiment_id=job.experiment_id,
            payload={"error": last_error or "search failed after retries"},
        )

    def _ensure_server_with_startup_retries(
        self,
        *,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
    ) -> ActiveServer:
        last_exception: Exception | None = None
        for startup_attempt in range(1, self._run_config.retry.startup_attempts + 1):
            self._state_store.increment_attempt(state, job.experiment_id, kind="startup")
            self._state_store.append_event(
                state,
                event_type="startup_attempt",
                experiment_id=job.experiment_id,
                payload={"attempt": startup_attempt},
            )
            try:
                return self._ensure_server(
                    job=job,
                    force_restart=startup_attempt > 1,
                )
            except Exception as exc:
                last_exception = exc
                self._state_store.append_event(
                    state,
                    event_type="startup_attempt_failed",
                    experiment_id=job.experiment_id,
                    payload={"attempt": startup_attempt, "error": str(exc)},
                )
                self._release_active_server(reason="startup_attempt_failed")

        if last_exception is None:
            raise RuntimeError("startup retries exhausted")
        raise RuntimeError(str(last_exception))

    def _ensure_server(
        self,
        *,
        job: ExpandedExperimentJob,
        force_restart: bool,
    ) -> ActiveServer:
        if force_restart:
            self._release_active_server(reason="force_restart")

        active_server = self._lifecycle.active_server
        if active_server is not None and self._active_reuse_key == job.server_signature_key:
            if self._lifecycle.is_ready(active_server):
                return active_server
            self._release_active_server(reason="active_server_unhealthy")

        if self._active_reuse_key is not None and self._active_reuse_key != job.server_signature_key:
            self._release_active_server(reason="signature_mismatch")

        if self._active_lease is None:
            self._active_lease = self._gpu_manager.acquire()
        if self._active_ports is None:
            self._active_ports = self._port_allocator.reserve()

        runtime_signature = runtime_server_signature(
            server_signature_key=job.server_signature_key,
            gpu_id=self._active_lease.gpu_id,
            base_port=self._active_ports.base_port,
            metrics_port=self._active_ports.metrics_port,
        )

        server = self._lifecycle.ensure_server(
            job=job,
            gpu_id=self._active_lease.gpu_id,
            ports=self._active_ports,
            runtime_signature=runtime_signature,
            logs_dir=self._state_store.logs_dir,
            force_restart=False,
        )
        self._active_reuse_key = job.server_signature_key
        return server

    def _release_active_server(self, *, reason: str) -> None:
        self._lifecycle.stop_active_server(reason=reason)
        if self._active_lease is not None:
            self._gpu_manager.release(self._active_lease)
            self._active_lease = None
        if self._active_ports is not None:
            self._port_allocator.release(self._active_ports)
            self._active_ports = None
        self._active_reuse_key = None
