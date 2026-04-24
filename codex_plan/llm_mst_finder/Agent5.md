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
  completion_arrival_tolerance: 0.03
  max_positive_backlog_slope: 0.05
  max_error_rate: 0.01
  ttft_slo_ms: 2000
  tpot_slo_ms: 80
  e2e_slo_ms: null
  drift_test:
    method: theil_sen
    min_relative_increase: 0.20
```

## Core rules
### Stable

Classify as stable if, after warmup:
```
completion_rate / arrival_rate >= 1 - tolerance
outstanding slope <= threshold
TTFT p90/p99 has no positive drift
TPOT p90/p99 has no positive drift
error rate <= threshold
preemption rate acceptable
SLOs satisfied
```
### Unsustainable

Classify as unstable if any of these hold:

```
completion_rate persistently < arrival_rate
outstanding_end grows over consecutive windows
num_waiting grows over consecutive windows
TTFT p90 or queue time has positive drift
TPOT p90 has positive drift
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
Theil-Sen slope for TTFT, TPOT, outstanding, num_waiting
Mann-Kendall trend test optional
```
Do not overfit the first version. A robust monotonic trend rule is enough.

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
