from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import fields
from math import floor, isfinite
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .records import RequestRecord, ServerMetricSample, WindowSummary

_PREEMPTION_COUNTER_NAMES = (
    "vllm:num_preemptions",
    "vllm_num_preemptions",
    "vllm:num_preemptions_total",
    "vllm_num_preemptions_total",
    "vllm:request_preemptions_total",
    "vllm_request_preemptions_total",
    "vllm:requests_preempted_total",
    "vllm_requests_preempted_total",
)

_NON_CAPACITY_FAILURE_CLASSES = frozenset(
    {
        "model_server_harmony_stream_error",
    }
)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = floor(position)
    upper = lower if position.is_integer() else lower + 1
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _to_ms(values_s: Iterable[float]) -> list[float]:
    return [value * 1000.0 for value in values_s]


def _assign_window(relative_ts: float, window_s: float, window_count: int) -> int:
    if not isfinite(relative_ts):
        raise ValueError(f"timestamp must be finite, got {relative_ts!r}")
    if relative_ts < 0:
        raise ValueError(f"timestamp must be within the trial timebase, got {relative_ts!r}")
    index = floor(relative_ts / window_s)
    if index >= window_count:
        raise ValueError(
            f"timestamp {relative_ts!r} falls outside windowed trial range with {window_count} windows"
        )
    return index


def _extract_raw_counter(sample: ServerMetricSample, metric_names: Sequence[str]) -> float | None:
    matched_values: list[float] = []
    for metric_name in metric_names:
        raw_series = sample.raw.get(metric_name)
        if raw_series is None:
            continue
        if not isinstance(raw_series, list):
            raise ValueError(f"raw metric series for {metric_name!r} must be a list")
        for item in raw_series:
            if not isinstance(item, dict) or "value" not in item:
                raise ValueError(f"raw metric series for {metric_name!r} must contain dict values")
            value = item["value"]
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"raw metric value for {metric_name!r} must be finite")
            matched_values.append(float(value))
    if not matched_values:
        return None
    return sum(matched_values)


class FixedWindowAggregator:
    def __init__(self, *, window_s: float = 10.0) -> None:
        if not isfinite(window_s) or window_s <= 0:
            raise ValueError(f"window_s must be a positive finite value, got {window_s!r}")
        self._window_s = window_s

    @property
    def window_s(self) -> float:
        return self._window_s

    def summarize(
        self,
        *,
        trial_id: str,
        request_records: Sequence[RequestRecord],
        server_metrics: Sequence[ServerMetricSample],
    ) -> list[WindowSummary]:
        if not trial_id:
            raise ValueError("trial_id must be non-empty")
        if not request_records:
            raise ValueError("request_records must be non-empty")

        validated_records = list(request_records)
        validated_metrics = sorted(server_metrics, key=lambda sample: sample.ts)
        for record in validated_records:
            if record.trial_id != trial_id:
                raise ValueError(
                    f"request record {record.request_id!r} trial_id={record.trial_id!r} "
                    f"does not match {trial_id!r}"
                )
            if record.actual_send_ts is None:
                raise ValueError(
                    f"request record {record.request_id!r} requires actual_send_ts for windowing"
                )
            if record.end_ts is None:
                raise ValueError(f"request record {record.request_id!r} requires end_ts for windowing")

        trial_start_ts = min(
            min(record.actual_send_ts for record in validated_records if record.actual_send_ts is not None),
            min((sample.ts for sample in validated_metrics), default=float("inf")),
        )
        if not isfinite(trial_start_ts):
            raise ValueError("trial_start_ts could not be determined")

        trial_end_ts = max(
            max(record.end_ts for record in validated_records if record.end_ts is not None),
            max((sample.ts for sample in validated_metrics), default=float("-inf")),
        )
        if not isfinite(trial_end_ts) or trial_end_ts < trial_start_ts:
            raise ValueError(
                f"trial_end_ts must be finite and >= trial_start_ts, got {trial_end_ts!r}"
            )

        trial_duration_s = trial_end_ts - trial_start_ts
        window_count = floor(trial_duration_s / self._window_s) + 1

        arrivals_by_window = [0] * window_count
        completions_by_window = [0] * window_count
        failures_by_window = [0] * window_count
        ttft_by_window: list[list[float]] = [[] for _ in range(window_count)]
        tpot_by_window: list[list[float]] = [[] for _ in range(window_count)]
        itl_by_window: list[list[float]] = [[] for _ in range(window_count)]
        e2e_by_window: list[list[float]] = [[] for _ in range(window_count)]
        prompt_len_by_window: list[list[float]] = [[] for _ in range(window_count)]
        expected_output_len_by_window: list[list[float]] = [[] for _ in range(window_count)]
        actual_output_len_by_window: list[list[float]] = [[] for _ in range(window_count)]
        outstanding_events: dict[float, int] = defaultdict(int)

        for record in validated_records:
            if _is_ignored_for_capacity_stability(record):
                continue
            assert record.actual_send_ts is not None
            assert record.end_ts is not None
            send_relative_s = record.actual_send_ts - trial_start_ts
            end_relative_s = record.end_ts - trial_start_ts
            arrival_window = _assign_window(send_relative_s, self._window_s, window_count)
            terminal_window = _assign_window(end_relative_s, self._window_s, window_count)

            arrivals_by_window[arrival_window] += 1
            if record.success:
                completions_by_window[terminal_window] += 1
            else:
                failures_by_window[terminal_window] += 1

            outstanding_events[send_relative_s] += 1
            outstanding_events[end_relative_s] -= 1

            if record.ttft_s is not None:
                ttft_by_window[arrival_window].append(record.ttft_s * 1000.0)
            if record.tpot_s is not None:
                tpot_by_window[arrival_window].append(record.tpot_s * 1000.0)
            if record.e2e_s is not None:
                e2e_by_window[arrival_window].append(record.e2e_s * 1000.0)
            if record.itl_s:
                itl_by_window[arrival_window].extend(_to_ms(record.itl_s))
            prompt_len_by_window[arrival_window].append(float(record.prompt_len))
            expected_output_len_by_window[arrival_window].append(float(record.expected_output_len))
            if record.actual_output_len is not None:
                actual_output_len_by_window[arrival_window].append(float(record.actual_output_len))

        metric_windows: list[list[ServerMetricSample]] = [[] for _ in range(window_count)]
        for sample in validated_metrics:
            relative_ts = sample.ts - trial_start_ts
            metric_windows[_assign_window(relative_ts, self._window_s, window_count)].append(sample)

        prompt_deltas = self._counter_deltas(
            metrics=validated_metrics,
            trial_start_ts=trial_start_ts,
            window_count=window_count,
            accessor=lambda sample: sample.prompt_tokens_total,
        )
        generation_deltas = self._counter_deltas(
            metrics=validated_metrics,
            trial_start_ts=trial_start_ts,
            window_count=window_count,
            accessor=lambda sample: sample.generation_tokens_total,
        )
        preemption_deltas = self._counter_deltas(
            metrics=validated_metrics,
            trial_start_ts=trial_start_ts,
            window_count=window_count,
            accessor=lambda sample: _extract_raw_counter(sample, _PREEMPTION_COUNTER_NAMES),
        )

        outstanding_stats = self._outstanding_stats(
            outstanding_events=outstanding_events,
            window_count=window_count,
        )

        windows: list[WindowSummary] = []
        for window_idx in range(window_count):
            start_s = window_idx * self._window_s
            end_s = start_s + self._window_s
            completions = completions_by_window[window_idx]
            failures = failures_by_window[window_idx]
            terminal_events = completions + failures
            prompt_delta = None if prompt_deltas is None else prompt_deltas[window_idx]
            generation_delta = None if generation_deltas is None else generation_deltas[window_idx]
            preemption_delta = None if preemption_deltas is None else preemption_deltas[window_idx]
            total_delta = (
                None
                if prompt_delta is None or generation_delta is None
                else prompt_delta + generation_delta
            )

            samples = metric_windows[window_idx]
            windows.append(
                WindowSummary(
                    trial_id=trial_id,
                    window_idx=window_idx,
                    start_s=start_s,
                    end_s=end_s,
                    arrivals=arrivals_by_window[window_idx],
                    completions=completions,
                    failures=failures,
                    arrival_rate=arrivals_by_window[window_idx] / self._window_s,
                    completion_rate=completions / self._window_s,
                    error_rate=(failures / terminal_events) if terminal_events else 0.0,
                    outstanding_start=outstanding_stats[window_idx]["start"],
                    outstanding_end=outstanding_stats[window_idx]["end"],
                    outstanding_mean=outstanding_stats[window_idx]["mean"],
                    outstanding_slope=(
                        (outstanding_stats[window_idx]["end"] - outstanding_stats[window_idx]["start"])
                        / self._window_s
                    ),
                    ttft_p50_ms=_percentile(ttft_by_window[window_idx], 50.0),
                    ttft_p90_ms=_percentile(ttft_by_window[window_idx], 90.0),
                    ttft_p99_ms=_percentile(ttft_by_window[window_idx], 99.0),
                    tpot_p50_ms=_percentile(tpot_by_window[window_idx], 50.0),
                    tpot_p90_ms=_percentile(tpot_by_window[window_idx], 90.0),
                    tpot_p99_ms=_percentile(tpot_by_window[window_idx], 99.0),
                    itl_p90_ms=_percentile(itl_by_window[window_idx], 90.0),
                    e2e_p90_ms=_percentile(e2e_by_window[window_idx], 90.0),
                    e2e_p99_ms=_percentile(e2e_by_window[window_idx], 99.0),
                    prompt_tok_s=None if prompt_delta is None else prompt_delta / self._window_s,
                    generation_tok_s=(
                        None if generation_delta is None else generation_delta / self._window_s
                    ),
                    total_tok_s=None if total_delta is None else total_delta / self._window_s,
                    num_running_mean=_mean(
                        [sample.num_running for sample in samples if sample.num_running is not None]
                    ),
                    num_waiting_mean=_mean(
                        [sample.num_waiting for sample in samples if sample.num_waiting is not None]
                    ),
                    num_swapped_mean=_mean(
                        [sample.num_swapped for sample in samples if sample.num_swapped is not None]
                    ),
                    kv_cache_usage_mean=_mean(
                        [
                            sample.kv_cache_usage
                            for sample in samples
                            if sample.kv_cache_usage is not None
                        ]
                    ),
                    kv_cache_usage_max=max(
                        [
                            sample.kv_cache_usage
                            for sample in samples
                            if sample.kv_cache_usage is not None
                        ],
                        default=None,
                    ),
                    preemptions_delta=preemption_delta,
                    prompt_len_mean=_mean(prompt_len_by_window[window_idx]),
                    expected_output_len_mean=_mean(expected_output_len_by_window[window_idx]),
                    actual_output_len_mean=_mean(actual_output_len_by_window[window_idx]),
                )
            )
        return windows

    def write_outputs(
        self,
        *,
        trial_id: str,
        request_records: Sequence[RequestRecord],
        server_metrics: Sequence[ServerMetricSample],
        output_path: Path,
    ) -> int:
        windows = self.summarize(
            trial_id=trial_id,
            request_records=request_records,
            server_metrics=server_metrics,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [field.name for field in fields(WindowSummary)]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for window in windows:
                writer.writerow(window.to_dict())
        return len(windows)

    def _counter_deltas(
        self,
        *,
        metrics: Sequence[ServerMetricSample],
        trial_start_ts: float,
        window_count: int,
        accessor: Callable[[ServerMetricSample], float | None],
    ) -> list[float] | None:
        deltas = [0.0] * window_count
        previous_value: float | None = None
        observed_samples = 0
        for sample in metrics:
            value = accessor(sample)
            if value is None:
                continue
            if not isfinite(value):
                raise ValueError(f"counter metric value must be finite, got {value!r}")
            observed_samples += 1
            if previous_value is not None:
                if value < previous_value:
                    raise ValueError(
                        f"counter metric decreased from {previous_value!r} to {value!r}"
                    )
                relative_ts = sample.ts - trial_start_ts
                window_idx = _assign_window(relative_ts, self._window_s, window_count)
                deltas[window_idx] += value - previous_value
            previous_value = value
        if observed_samples < 2:
            return None
        return deltas

    def _outstanding_stats(
        self,
        *,
        outstanding_events: dict[float, int],
        window_count: int,
    ) -> list[dict[str, float | int]]:
        aggregated_events = sorted(outstanding_events.items())
        outstanding = 0
        event_index = 0
        current_time = 0.0
        stats: list[dict[str, float | int]] = []

        for window_idx in range(window_count):
            start_s = window_idx * self._window_s
            end_s = start_s + self._window_s
            if current_time < start_s:
                current_time = start_s
            outstanding_start = outstanding
            area = 0.0
            while event_index < len(aggregated_events):
                event_ts, delta = aggregated_events[event_index]
                if event_ts < start_s:
                    raise ValueError(
                        f"outstanding event at {event_ts!r} precedes the active window start {start_s!r}"
                    )
                if event_ts >= end_s:
                    break
                area += outstanding * (event_ts - current_time)
                outstanding += delta
                if outstanding < 0:
                    raise ValueError("outstanding request count became negative")
                current_time = event_ts
                event_index += 1
            area += outstanding * (end_s - current_time)
            stats.append(
                {
                    "start": outstanding_start,
                    "end": outstanding,
                    "mean": area / self._window_s,
                }
            )
            current_time = end_s

        if outstanding != 0:
            raise ValueError(f"outstanding request count ended at {outstanding}, expected 0")
        return stats


def _is_ignored_for_capacity_stability(record: RequestRecord) -> bool:
    if record.success:
        return False
    failure_class = record.metadata.get("failure_class")
    return isinstance(failure_class, str) and failure_class in _NON_CAPACITY_FAILURE_CLASSES


__all__ = ["FixedWindowAggregator"]
