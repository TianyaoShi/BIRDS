# Implement windowed aggregation
## Goal

Convert per-request records and server metric samples into time-window summaries.

## Files
```
windowing.py
Windowing policy
```
Use fixed-width windows:
```
window_s = 10 by default
```
Use arrival-time-based windows for request metrics:
```
window_id = floor((actual_send_ts - trial_start_ts) / window_s)
```
For server metrics, average/gauge-summarize within each window and compute counter deltas.

## Window summary schema
```Python
@dataclass
class WindowSummary:
    trial_id: str
    window_idx: int
    start_s: float
    end_s: float

    arrivals: int
    completions: int
    failures: int
    arrival_rate: float
    completion_rate: float
    error_rate: float

    outstanding_start: int
    outstanding_end: int
    outstanding_mean: float
    outstanding_slope: float

    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    ttft_p99_ms: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None
    tpot_p99_ms: float | None
    itl_p90_ms: float | None
    e2e_p90_ms: float | None
    e2e_p99_ms: float | None

    prompt_tok_s: float | None
    generation_tok_s: float | None
    total_tok_s: float | None

    num_running_mean: float | None
    num_waiting_mean: float | None
    num_swapped_mean: float | None
    kv_cache_usage_mean: float | None
    kv_cache_usage_max: float | None
    preemptions_delta: float | None
```
## Derived quantities

Compute:
```
arrival_rate = arrivals / window_s
completion_rate = completions / window_s
outstanding = cumulative_arrivals - cumulative_completions - cumulative_failures
queue drift = slope(outstanding over windows)
TTFT drift = slope(p90 TTFT over windows)
TPOT drift = slope(p90 TPOT over windows)
```

## Local consistency constraints

+ Import shared dataclasses from `records.py`; do not redefine schemas locally.
+ Reject negative durations, inconsistent timestamps, nonpositive `window_s`, and records outside the trial timebase.
+ Empty windows are allowed, but their rates and missing percentiles must be explicit and deterministic.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
