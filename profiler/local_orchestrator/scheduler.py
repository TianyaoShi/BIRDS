from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
import shutil
from threading import Lock, Thread
from typing import Any, Callable, Protocol

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
        gpu_ids: tuple[int, ...],
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
    def acquire(self, gpu_count: int = 1) -> GPULease:
        ...

    def release(self, lease: GPULease) -> None:
        ...

    def snapshot(self) -> dict[str, object]:
        ...


class PortAllocatorProtocol(Protocol):
    def reserve(self) -> PortReservation:
        ...

    def release(self, reservation: PortReservation) -> None:
        ...


@dataclass(slots=True)
class _WorkerSlot:
    slot_index: int
    lease: GPULease
    ports: PortReservation
    lifecycle: LifecycleProtocol
    active_reuse_key: str | None = None


@dataclass(slots=True)
class _RunningJob:
    thread: Thread
    slot: _WorkerSlot
    job: ExpandedExperimentJob


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
        lifecycle_factory: Callable[[], LifecycleProtocol] | None = None,
    ) -> None:
        self._run_config = run_config
        self._gpu_manager = gpu_manager
        self._port_allocator = port_allocator
        self._lifecycle = lifecycle
        self._adapter = adapter
        self._state_store = state_store
        self._lifecycle_factory = lifecycle_factory
        self._state_lock = Lock()

        self._active_lease: GPULease | None = None
        self._active_ports: PortReservation | None = None
        self._active_reuse_key: str | None = None

    def run(
        self,
        *,
        jobs: list[ExpandedExperimentJob],
        state: dict[str, Any],
        resume: bool,
        force: bool = False,
    ) -> dict[str, Any]:
        if resume:
            self._with_state_lock(lambda: self._state_store.reconcile_jobs(state))

        pending_jobs = self._collect_pending_jobs(jobs=jobs, state=state, resume=resume, force=force)

        try:
            if self._should_run_parallel(pending_jobs):
                self._run_parallel_jobs(jobs=pending_jobs, state=state)
            else:
                for job in pending_jobs:
                    self._run_single_job(job=job, state=state)
            with self._state_lock:
                state["status"] = "completed"
                self._state_store.save(state)
                return self._state_store.write_summary_files(state)
        finally:
            self._release_active_server(reason="scheduler_shutdown")
            self._lifecycle.shutdown()

    def _collect_pending_jobs(
        self,
        *,
        jobs: list[ExpandedExperimentJob],
        state: dict[str, Any],
        resume: bool,
        force: bool,
    ) -> list[ExpandedExperimentJob]:
        pending_jobs: list[ExpandedExperimentJob] = []
        with self._state_lock:
            for job in jobs:
                job_state = self._state_store.find_job(state, job.experiment_id)
                prior_status = str(job_state.get("status", "planned"))
                if force:
                    self._prepare_job_for_force_rerun(
                        job=job,
                        state=state,
                        prior_status=prior_status,
                    )
                    pending_jobs.append(job)
                    continue
                if job_state.get("status") in {"succeeded", "skipped"}:
                    continue
                if resume:
                    self._refresh_job_plan_for_resume(job=job, state=state, prior_status=prior_status)
                pending_jobs.append(job)
        return pending_jobs

    def _refresh_job_plan_for_resume(
        self,
        *,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
        prior_status: str,
    ) -> None:
        self._state_store.refresh_job_plan(state, job=job)
        self._state_store.append_event(
            state,
            event_type="job_plan_refreshed_for_resume",
            experiment_id=job.experiment_id,
            payload={
                "prior_status": prior_status,
                "result_dir": str(job.result_dir),
            },
        )

    def _prepare_job_for_force_rerun(
        self,
        *,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
        prior_status: str,
    ) -> None:
        state_job = self._state_store.find_job(state, job.experiment_id)
        prior_result_dir_raw = state_job.get("result_dir")
        if isinstance(prior_result_dir_raw, str):
            prior_result_dir = Path(prior_result_dir_raw)
            if prior_result_dir != job.result_dir and prior_result_dir.exists():
                shutil.rmtree(prior_result_dir)
        if job.result_dir.exists():
            shutil.rmtree(job.result_dir)
        self._state_store.refresh_job_plan(state, job=job)
        self._state_store.reset_job_for_rerun(state, experiment_id=job.experiment_id)
        self._state_store.append_event(
            state,
            event_type="job_reset_for_force_rerun",
            experiment_id=job.experiment_id,
            payload={
                "prior_status": prior_status,
                "result_dir": str(job.result_dir),
            },
        )

    def _should_run_parallel(self, pending_jobs: list[ExpandedExperimentJob]) -> bool:
        return (
            self._lifecycle_factory is not None
            and self._run_config.max_active_gpus > 1
            and len(pending_jobs) > 1
        )

    def _run_parallel_jobs(self, *, jobs: list[ExpandedExperimentJob], state: dict[str, Any]) -> None:
        if self._lifecycle_factory is None:
            raise RuntimeError("parallel execution requires lifecycle_factory")

        pending_jobs = self._filter_preflight_valid_jobs(jobs=jobs, state=state)
        completion_queue: Queue[tuple[_WorkerSlot, ExpandedExperimentJob, Exception | None]] = Queue()
        running_jobs: list[_RunningJob] = []
        worker_failures: list[Exception] = []
        next_slot_index = 0

        try:
            while pending_jobs or running_jobs:
                launched_any = False
                while True:
                    free_gpu_count = len(self._gpu_manager.snapshot()["free_gpu_ids"])
                    next_job_index = self._next_schedulable_job_index(
                        pending_jobs,
                        free_gpu_count=free_gpu_count,
                    )
                    if next_job_index is None:
                        break

                    job = pending_jobs.pop(next_job_index)
                    slot = self._build_job_slot(slot_index=next_slot_index, gpu_count=job.launch.gpu_count)
                    next_slot_index += 1
                    thread = Thread(
                        target=self._run_job_on_dynamic_slot,
                        args=(job, state, slot, completion_queue),
                        name=f"orchestrator-worker-{slot.slot_index}",
                        daemon=True,
                    )
                    running_jobs.append(_RunningJob(thread=thread, slot=slot, job=job))
                    thread.start()
                    launched_any = True

                if not running_jobs:
                    failed_jobs = pending_jobs
                    pending_jobs = []
                    for job in failed_jobs:
                        preflight_error = self._preflight_error(job)
                        error = preflight_error or (
                            f"job requires gpu_count={job.launch.gpu_count}, "
                            f"but no schedulable GPU lease is available"
                        )
                        self._mark_job_failed(state, experiment_id=job.experiment_id, error=error)
                        self._append_event(
                            state,
                            event_type="job_failed_preflight",
                            experiment_id=job.experiment_id,
                            payload={"error": error},
                        )
                    break

                if launched_any:
                    continue

                slot, job, exc = completion_queue.get()
                matching_running_job = next(
                    (
                        running_job
                        for running_job in running_jobs
                        if running_job.slot is slot and running_job.job is job
                    ),
                    None,
                )
                if matching_running_job is not None:
                    matching_running_job.thread.join()
                    running_jobs.remove(matching_running_job)
                self._release_slot(slot, reason="parallel_job_finished")
                if exc is not None:
                    worker_failures.append(exc)
        finally:
            for running_job in list(running_jobs):
                running_job.thread.join()
                self._release_slot(running_job.slot, reason="parallel_scheduler_shutdown")
                running_jobs.remove(running_job)

        if worker_failures:
            first_failure = worker_failures[0]
            raise RuntimeError(f"parallel scheduler worker failed: {first_failure}") from first_failure

    def _filter_preflight_valid_jobs(
        self,
        *,
        jobs: list[ExpandedExperimentJob],
        state: dict[str, Any],
    ) -> list[ExpandedExperimentJob]:
        valid_jobs: list[ExpandedExperimentJob] = []
        for job in jobs:
            preflight_error = self._preflight_error(job)
            if preflight_error is None:
                valid_jobs.append(job)
                continue
            self._mark_job_failed(state, experiment_id=job.experiment_id, error=preflight_error)
            self._append_event(
                state,
                event_type="job_failed_preflight",
                experiment_id=job.experiment_id,
                payload={"error": preflight_error},
            )
        return valid_jobs

    def _next_schedulable_job_index(
        self,
        pending_jobs: list[ExpandedExperimentJob],
        *,
        free_gpu_count: int,
    ) -> int | None:
        best_index: int | None = None
        best_gpu_count = -1
        for index, job in enumerate(pending_jobs):
            if job.launch.gpu_count > self._run_config.max_active_gpus:
                continue
            if job.launch.gpu_count > free_gpu_count:
                continue
            if job.launch.gpu_count > best_gpu_count:
                best_index = index
                best_gpu_count = job.launch.gpu_count
        return best_index

    def _run_job_on_dynamic_slot(
        self,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
        slot: _WorkerSlot,
        completion_queue: Queue[tuple[_WorkerSlot, ExpandedExperimentJob, Exception | None]],
    ) -> None:
        failure: Exception | None = None
        try:
            self._run_single_job_on_slot(job=job, state=state, slot=slot)
        except Exception as exc:  # pragma: no cover - defensive safety
            failure = exc
        finally:
            completion_queue.put((slot, job, failure))

    def _build_job_slot(self, *, slot_index: int, gpu_count: int) -> _WorkerSlot:
        if self._lifecycle_factory is None:
            raise RuntimeError("parallel execution requires lifecycle_factory")

        lease: GPULease | None = None
        ports: PortReservation | None = None
        try:
            lease = self._gpu_manager.acquire(gpu_count)
            ports = self._port_allocator.reserve()
            lifecycle = self._lifecycle if slot_index == 0 else self._lifecycle_factory()
            return _WorkerSlot(
                slot_index=slot_index,
                lease=lease,
                ports=ports,
                lifecycle=lifecycle,
            )
        except Exception:
            if ports is not None:
                self._port_allocator.release(ports)
            if lease is not None:
                self._gpu_manager.release(lease)
            raise

    def _run_single_job(self, *, job: ExpandedExperimentJob, state: dict[str, Any]) -> None:
        preflight_error = self._preflight_error(job)
        if preflight_error is not None:
            self._mark_job_failed(state, experiment_id=job.experiment_id, error=preflight_error)
            self._append_event(
                state,
                event_type="job_failed_preflight",
                experiment_id=job.experiment_id,
                payload={"error": preflight_error},
            )
            return

        last_error: str | None = None

        for search_attempt in range(1, self._run_config.retry.search_attempts + 1):
            self._increment_attempt(state, experiment_id=job.experiment_id, kind="search")
            self._set_job_status(state, experiment_id=job.experiment_id, status="running")
            self._append_event(
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
                self._append_event(
                    state,
                    event_type="startup_failed",
                    experiment_id=job.experiment_id,
                    payload={"attempt": search_attempt, "error": str(exc)},
                )
                self._release_active_server(reason="startup_failed")
                continue

            try:
                result = self._adapter.invoke(
                    job=job,
                    server=server,
                    logs_dir=self._state_store.logs_dir,
                )
            except Exception as exc:
                last_error = f"search adapter raised before completing attempt {search_attempt}: {exc}"
                self._append_event(
                    state,
                    event_type="search_failed",
                    experiment_id=job.experiment_id,
                    payload={"attempt": search_attempt, "error": last_error},
                )
                self._release_active_server(reason="search_exception")
                continue
            if result.success:
                self._mark_job_succeeded(state, experiment_id=job.experiment_id, result=result)
                self._append_event(
                    state,
                    event_type="job_succeeded",
                    experiment_id=job.experiment_id,
                    payload={"attempt": search_attempt, "result_dir": str(job.result_dir)},
                )
                return

            last_error = result.error or f"search command failed with exit code {result.return_code}"
            self._append_event(
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

        self._mark_job_failed(
            state,
            experiment_id=job.experiment_id,
            error=last_error or "search failed after retries",
        )
        self._append_event(
            state,
            event_type="job_failed",
            experiment_id=job.experiment_id,
            payload={"error": last_error or "search failed after retries"},
        )

    def _run_single_job_on_slot(
        self,
        *,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
        slot: _WorkerSlot,
    ) -> None:
        preflight_error = self._preflight_error(job)
        if preflight_error is not None:
            self._mark_job_failed(state, experiment_id=job.experiment_id, error=preflight_error)
            self._append_event(
                state,
                event_type="job_failed_preflight",
                experiment_id=job.experiment_id,
                payload={"error": preflight_error, "slot_index": slot.slot_index},
            )
            return

        last_error: str | None = None

        for search_attempt in range(1, self._run_config.retry.search_attempts + 1):
            self._increment_attempt(state, experiment_id=job.experiment_id, kind="search")
            self._set_job_status(state, experiment_id=job.experiment_id, status="running")
            self._append_event(
                state,
                event_type="search_attempt",
                experiment_id=job.experiment_id,
                payload={"attempt": search_attempt, "slot_index": slot.slot_index},
            )

            try:
                server = self._ensure_server_with_startup_retries_on_slot(
                    job=job,
                    state=state,
                    slot=slot,
                )
            except Exception as exc:
                last_error = f"startup failed before search attempt {search_attempt}: {exc}"
                self._append_event(
                    state,
                    event_type="startup_failed",
                    experiment_id=job.experiment_id,
                    payload={
                        "attempt": search_attempt,
                        "slot_index": slot.slot_index,
                        "error": str(exc),
                    },
                )
                self._release_slot_server(slot, reason="startup_failed")
                continue

            try:
                result = self._adapter.invoke(
                    job=job,
                    server=server,
                    logs_dir=self._state_store.logs_dir,
                )
            except Exception as exc:
                last_error = f"search adapter raised before completing attempt {search_attempt}: {exc}"
                self._append_event(
                    state,
                    event_type="search_failed",
                    experiment_id=job.experiment_id,
                    payload={
                        "attempt": search_attempt,
                        "slot_index": slot.slot_index,
                        "error": last_error,
                    },
                )
                self._release_slot_server(slot, reason="search_exception")
                continue
            if result.success:
                self._mark_job_succeeded(state, experiment_id=job.experiment_id, result=result)
                self._append_event(
                    state,
                    event_type="job_succeeded",
                    experiment_id=job.experiment_id,
                    payload={
                        "attempt": search_attempt,
                        "slot_index": slot.slot_index,
                        "result_dir": str(job.result_dir),
                    },
                )
                return

            last_error = result.error or f"search command failed with exit code {result.return_code}"
            self._append_event(
                state,
                event_type="search_failed",
                experiment_id=job.experiment_id,
                payload={
                    "attempt": search_attempt,
                    "slot_index": slot.slot_index,
                    "return_code": result.return_code,
                    "error": last_error,
                },
            )
            self._release_slot_server(slot, reason="search_retry_restart")

        self._mark_job_failed(
            state,
            experiment_id=job.experiment_id,
            error=last_error or "search failed after retries",
        )
        self._append_event(
            state,
            event_type="job_failed",
            experiment_id=job.experiment_id,
            payload={"error": last_error or "search failed after retries", "slot_index": slot.slot_index},
        )

    def _ensure_server_with_startup_retries(
        self,
        *,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
    ) -> ActiveServer:
        last_exception: Exception | None = None
        for startup_attempt in range(1, self._run_config.retry.startup_attempts + 1):
            self._increment_attempt(state, experiment_id=job.experiment_id, kind="startup")
            self._append_event(
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
                self._append_event(
                    state,
                    event_type="startup_attempt_failed",
                    experiment_id=job.experiment_id,
                    payload={"attempt": startup_attempt, "error": str(exc)},
                )
                self._release_active_server(reason="startup_attempt_failed")

        if last_exception is None:
            raise RuntimeError("startup retries exhausted")
        raise RuntimeError(str(last_exception))

    def _ensure_server_with_startup_retries_on_slot(
        self,
        *,
        job: ExpandedExperimentJob,
        state: dict[str, Any],
        slot: _WorkerSlot,
    ) -> ActiveServer:
        last_exception: Exception | None = None
        for startup_attempt in range(1, self._run_config.retry.startup_attempts + 1):
            self._increment_attempt(state, experiment_id=job.experiment_id, kind="startup")
            self._append_event(
                state,
                event_type="startup_attempt",
                experiment_id=job.experiment_id,
                payload={"attempt": startup_attempt, "slot_index": slot.slot_index},
            )
            try:
                return self._ensure_server_on_slot(
                    job=job,
                    slot=slot,
                    force_restart=startup_attempt > 1,
                )
            except Exception as exc:
                last_exception = exc
                self._append_event(
                    state,
                    event_type="startup_attempt_failed",
                    experiment_id=job.experiment_id,
                    payload={
                        "attempt": startup_attempt,
                        "slot_index": slot.slot_index,
                        "error": str(exc),
                    },
                )
                self._release_slot_server(slot, reason="startup_attempt_failed")

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
            self._active_lease = self._gpu_manager.acquire(job.launch.gpu_count)
        if self._active_ports is None:
            self._active_ports = self._port_allocator.reserve()

        runtime_signature = runtime_server_signature(
            server_signature_key=job.server_signature_key,
            gpu_ids=self._active_lease.gpu_ids,
            base_port=self._active_ports.base_port,
            metrics_port=self._active_ports.metrics_port,
        )

        server = self._lifecycle.ensure_server(
            job=job,
            gpu_ids=self._active_lease.gpu_ids,
            ports=self._active_ports,
            runtime_signature=runtime_signature,
            logs_dir=self._state_store.logs_dir,
            force_restart=False,
        )
        self._active_reuse_key = job.server_signature_key
        return server

    def _ensure_server_on_slot(
        self,
        *,
        job: ExpandedExperimentJob,
        slot: _WorkerSlot,
        force_restart: bool,
    ) -> ActiveServer:
        if force_restart:
            self._release_slot_server(slot, reason="force_restart")

        active_server = slot.lifecycle.active_server
        if active_server is not None and slot.active_reuse_key == job.server_signature_key:
            if slot.lifecycle.is_ready(active_server):
                return active_server
            self._release_slot_server(slot, reason="active_server_unhealthy")

        if slot.active_reuse_key is not None and slot.active_reuse_key != job.server_signature_key:
            self._release_slot_server(slot, reason="signature_mismatch")

        runtime_signature = runtime_server_signature(
            server_signature_key=job.server_signature_key,
            gpu_ids=slot.lease.gpu_ids,
            base_port=slot.ports.base_port,
            metrics_port=slot.ports.metrics_port,
        )

        server = slot.lifecycle.ensure_server(
            job=job,
            gpu_ids=slot.lease.gpu_ids,
            ports=slot.ports,
            runtime_signature=runtime_signature,
            logs_dir=self._state_store.logs_dir,
            force_restart=False,
        )
        slot.active_reuse_key = job.server_signature_key
        return server

    def _preflight_error(self, job: ExpandedExperimentJob) -> str | None:
        if job.launch.gpu_count > self._run_config.max_active_gpus:
            return (
                f"job requires gpu_count={job.launch.gpu_count}, "
                f"but run.max_active_gpus={self._run_config.max_active_gpus}"
            )
        return None

    def _release_active_server(self, *, reason: str) -> None:
        self._lifecycle.stop_active_server(reason=reason)
        if self._active_lease is not None:
            self._gpu_manager.release(self._active_lease)
            self._active_lease = None
        if self._active_ports is not None:
            self._port_allocator.release(self._active_ports)
            self._active_ports = None
        self._active_reuse_key = None

    def _release_slot_server(self, slot: _WorkerSlot, *, reason: str) -> None:
        slot.lifecycle.stop_active_server(reason=reason)
        slot.active_reuse_key = None

    def _release_slot(self, slot: _WorkerSlot, *, reason: str) -> None:
        self._release_slot_server(slot, reason=reason)
        self._port_allocator.release(slot.ports)
        self._gpu_manager.release(slot.lease)
        if slot.lifecycle is not self._lifecycle:
            slot.lifecycle.shutdown()

    def _with_state_lock(self, callback: Callable[[], Any]) -> Any:
        with self._state_lock:
            return callback()

    def _increment_attempt(self, state: dict[str, Any], *, experiment_id: str, kind: str) -> None:
        with self._state_lock:
            self._state_store.increment_attempt(state, experiment_id, kind=kind)

    def _set_job_status(self, state: dict[str, Any], *, experiment_id: str, status: str) -> None:
        with self._state_lock:
            self._state_store.set_job_status(state, experiment_id=experiment_id, status=status)

    def _append_event(
        self,
        state: dict[str, Any],
        *,
        event_type: str,
        experiment_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        with self._state_lock:
            self._state_store.append_event(
                state,
                event_type=event_type,
                experiment_id=experiment_id,
                payload=payload,
            )

    def _mark_job_succeeded(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        result,
    ) -> None:
        with self._state_lock:
            self._state_store.mark_job_succeeded(
                state,
                experiment_id=experiment_id,
                result=result,
            )

    def _mark_job_failed(
        self,
        state: dict[str, Any],
        *,
        experiment_id: str,
        error: str,
    ) -> None:
        with self._state_lock:
            self._state_store.mark_job_failed(
                state,
                experiment_id=experiment_id,
                error=error,
            )
