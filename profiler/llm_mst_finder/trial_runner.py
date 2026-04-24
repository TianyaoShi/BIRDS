from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .loadgen import ClosedLoopLoadGenerator, OpenLoopLoadGenerator, RequestSource
from .records import RequestRecord, SampleRequest, ServerMetricSample, TrialConfig, TrialSummary
from .request_client import RequestClient
from .vllm_compat import calculate_benchmark_metrics


class MetricsPoller(Protocol):
    async def run(
        self,
        *,
        output_path: Path,
        stop_event: asyncio.Event,
        trial_id: str,
    ) -> Sequence[ServerMetricSample]:
        ...


class WindowAggregator(Protocol):
    def write_outputs(
        self,
        *,
        trial_id: str,
        request_records: Sequence[RequestRecord],
        server_metrics: Sequence[ServerMetricSample],
        output_path: Path,
    ) -> int:
        ...


@dataclass(frozen=True, slots=True)
class TrialArtifacts:
    output_dir: Path
    request_records_path: Path
    summary_path: Path
    server_metrics_path: Path | None
    windows_path: Path | None


@dataclass(frozen=True, slots=True)
class TrialRunResult:
    config: TrialConfig
    summary: TrialSummary
    request_records: list[RequestRecord]
    server_metrics: list[ServerMetricSample]
    artifacts: TrialArtifacts


class TrialRunner:
    def __init__(
        self,
        request_client: RequestClient,
        *,
        metrics_poller: MetricsPoller | None = None,
        window_aggregator: WindowAggregator | None = None,
        time_fn=time.perf_counter,
    ) -> None:
        self._request_client = request_client
        self._metrics_poller = metrics_poller
        self._window_aggregator = window_aggregator
        self._time_fn = time_fn

    async def run_trial(
        self,
        config: TrialConfig,
        *,
        request_source: RequestSource,
        output_dir: str | Path,
    ) -> TrialRunResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        request_records_path = output_dir / "request_records.jsonl"
        summary_path = output_dir / "summary.json"
        server_metrics_path = output_dir / "server_metrics.jsonl" if self._metrics_poller else None
        windows_path = output_dir / "windows.csv" if self._window_aggregator else None
        for managed_path in (request_records_path, summary_path, server_metrics_path, windows_path):
            if managed_path is not None and managed_path.exists():
                raise FileExistsError(f"refusing to overwrite existing artifact {managed_path}")

        await self._request_client.open()
        records: list[RequestRecord] = []
        record_lock = asyncio.Lock()
        outstanding = 0
        max_observed_outstanding = 0
        abort_reason: str | None = None
        stop_event = asyncio.Event()

        async def send_and_record(
            request_index: int,
            scheduled_send_ts: float,
            sample: SampleRequest,
        ) -> RequestRecord:
            nonlocal outstanding
            try:
                record = await self._request_client.send_request(
                    sample,
                    request_id=f"{config.trial_id}-{request_index:06d}",
                    trial_id=config.trial_id,
                    scheduled_send_ts=scheduled_send_ts,
                )
                async with record_lock:
                    records.append(record)
                    self._append_jsonl(request_records_path, record.to_dict())
                return record
            finally:
                outstanding -= 1

        def start_dispatch(scheduled_request) -> asyncio.Task[RequestRecord]:
            nonlocal outstanding, max_observed_outstanding
            outstanding += 1
            max_observed_outstanding = max(max_observed_outstanding, outstanding)
            return asyncio.create_task(
                send_and_record(
                    scheduled_request.request_index,
                    scheduled_request.scheduled_send_ts,
                    scheduled_request.sample,
                )
            )

        def should_abort() -> bool:
            nonlocal abort_reason
            if config.safety_max_outstanding is None:
                return False
            if outstanding >= config.safety_max_outstanding:
                abort_reason = (
                    "outstanding requests reached safety_max_outstanding="
                    f"{config.safety_max_outstanding}"
                )
                return True
            return False

        metrics_task: asyncio.Task[Sequence[ServerMetricSample]] | None = None
        if self._metrics_poller is not None:
            assert server_metrics_path is not None
            metrics_task = asyncio.create_task(
                self._metrics_poller.run(
                    output_path=server_metrics_path,
                    stop_event=stop_event,
                    trial_id=config.trial_id,
                )
            )

        run_start_ts = self._time_fn()
        try:
            if config.mode == "open-loop":
                generator = OpenLoopLoadGenerator(
                    request_rate=config.request_rate,
                    burstiness=config.burstiness,
                    time_fn=self._time_fn,
                )
            else:
                generator = ClosedLoopLoadGenerator(
                    concurrency=config.concurrency,
                    think_time_s=config.think_time_s,
                    time_fn=self._time_fn,
                )
            await generator.run(
                request_source,
                start_dispatch,
                duration_s=config.duration_s,
                start_ts=run_start_ts,
                should_abort=should_abort,
            )
        finally:
            stop_event.set()

        if outstanding != 0:
            raise RuntimeError(f"outstanding request accounting ended at {outstanding}, expected 0")

        server_metrics: list[ServerMetricSample] = []
        if metrics_task is not None:
            server_metrics = list(await metrics_task)

        if not records:
            raise RuntimeError("trial produced no request records")

        records.sort(key=lambda record: record.scheduled_send_ts)
        run_end_ts = max(
            record.end_ts if record.end_ts is not None else record.actual_send_ts
            for record in records
            if (record.end_ts is not None or record.actual_send_ts is not None)
        )
        if run_end_ts is None:
            raise RuntimeError("trial run_end_ts could not be determined")
        wall_time_s = run_end_ts - run_start_ts
        if wall_time_s <= 0:
            raise RuntimeError(f"wall_time_s must be positive, got {wall_time_s}")

        benchmark_metrics = calculate_benchmark_metrics(records, wall_time_s)
        scheduling_delays = [
            record.scheduling_delay_s
            for record in records
            if record.scheduling_delay_s is not None
        ]
        successful_requests = sum(1 for record in records if record.success)
        failed_requests = len(records) - successful_requests
        summary = TrialSummary(
            trial_id=config.trial_id,
            mode=config.mode,
            status="aborted_safety" if abort_reason else "completed",
            requested_request_rate=config.request_rate,
            requested_concurrency=config.concurrency,
            target_duration_s=config.duration_s,
            wall_time_s=wall_time_s,
            started_requests=len(records),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            actual_send_rate=len(records) / wall_time_s,
            successful_completion_rate=successful_requests / wall_time_s,
            error_rate=failed_requests / len(records),
            mean_scheduling_delay_s=(
                sum(scheduling_delays) / len(scheduling_delays) if scheduling_delays else None
            ),
            max_scheduling_delay_s=max(scheduling_delays) if scheduling_delays else None,
            max_observed_outstanding=max_observed_outstanding,
            metrics_sample_count=len(server_metrics),
            abort_reason=abort_reason,
            benchmark_metrics=benchmark_metrics,
            metadata=config.metadata,
        )
        self._write_json(summary_path, {"config": config.to_dict(), "summary": summary.to_dict()})

        if self._window_aggregator is not None:
            assert windows_path is not None
            self._window_aggregator.write_outputs(
                trial_id=config.trial_id,
                request_records=records,
                server_metrics=server_metrics,
                output_path=windows_path,
            )

        artifacts = TrialArtifacts(
            output_dir=output_dir,
            request_records_path=request_records_path,
            summary_path=summary_path,
            server_metrics_path=server_metrics_path,
            windows_path=windows_path,
        )
        return TrialRunResult(
            config=config,
            summary=summary,
            request_records=records,
            server_metrics=server_metrics,
            artifacts=artifacts,
        )

    @staticmethod
    def _append_jsonl(path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
