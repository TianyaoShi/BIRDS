# Implement stability classifier
## Goal

Given window summaries, classify one trial as:

```
stable
unstable
slo_violation
uncertain
aborted_safety
```

This classifier assumes the trial is workload-valid. Context-length failures from the serving API are not stability evidence. They should be caught before dispatch by the workload/model compatibility step; if discovered in saved request records, trial analysis should report `trial_validity=invalid_workload` and skip stability classification or return low-confidence `uncertain` with an explicit invalid-workload reason.
## Files
```
stability.py
```
## Inputs
```YAML
stability:
  warmup_windows: 2
  min_eval_windows: 4
  completion_arrival_tolerance: 0.05
  max_positive_backlog_slope: 0.10
  min_backlog_growth_for_hard_pressure: 2.0
  min_backlog_relative_increase: 0.10
  backlog_trend_alpha: 0.05
  min_waiting_queue_mean_for_pressure: 1.0
  min_waiting_queue_active_fraction: 0.5
  max_error_rate: 0.03
  ttft_slo_ms: 2000
  tpot_slo_ms: 80
  drift_test:
    method: theil_sen
    min_relative_increase: 0.20
```

## Core rules
### Stable

Classify as stable if, after warmup:
```
no robust outstanding backlog trend
TTFT p90/p99 has no positive drift
TPOT p90/p99 has no positive drift
error rate <= threshold
preemption rate acceptable
SLOs satisfied
```

`completion_rate / arrival_rate < 1 - tolerance` is supporting evidence only. Do not make it an independent unstable decision criterion.

### Unsustainable

Classify as unstable if any of these hold:

```
robust outstanding backlog trend plus throughput/capacity confirmation
material sustained waiting queue pressure
TTFT p90 or queue time has positive drift with server/backlog pressure
TPOT p90 has positive drift with server/backlog pressure
preemptions increase repeatedly
safety outstanding cap is reached
```
### SLO violation

If the queue is stationary but TTFT/TPOT exceeds SLO:
```
slo_violation
```
This matters because the system may be stable but not acceptable for interactive use.

## Statistical method

Use robust slopes rather than ordinary least squares:
```
SciPy Theil-Sen slope for TTFT, TPOT, outstanding, num_waiting
Mann-Kendall trend test for outstanding backlog pressure
```

For outstanding backlog pressure, require all of:

```
Theil-Sen slope > max_positive_backlog_slope
Mann-Kendall p < backlog_trend_alpha
fitted end-to-end relative increase >= min_backlog_relative_increase
fitted end-to-end delta >= min_backlog_growth_for_hard_pressure
```

Do not use `outstanding requests grew across consecutive windows` or any equivalent consecutive-increase shortcut. It is not robust under stochastic open-loop arrivals.

`scipy` is a required dependency. Use `scipy.stats.theilslopes` and `scipy.stats.kendalltau`; do not add a local statistical fallback.

If required request/window fields are missing or internally inconsistent, raise. If optional server-side evidence is missing, classify from available evidence, lower confidence, and include that limitation in `reasons`.

Do not classify model context-limit validation failures as `unstable`, `slo_violation`, KV saturation, or scheduler overload. Those failures mean the workload is invalid for the selected model configuration.

## Return object
```Python
@dataclass
class StabilityResult:
    status: Literal["stable", "unstable", "slo_violation", "uncertain", "aborted_safety"]
    confidence: Literal["high", "medium", "low"]
    reasons: list[str]
    key_metrics: dict[str, float]
```

## Local consistency constraints

+ Use `WindowSummary` from `records.py` or `windowing.py`; do not invent a duplicate schema.
+ `uncertain` is for insufficient measurement evidence, not for hidden exceptions.
+ Treat context-limit/server validation errors as invalid workload metadata supplied by trial analysis, not as queueing drift.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
