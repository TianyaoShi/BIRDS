# Stability Judgement Correction

## Status

Implemented in `profiler/llm_mst_finder/stability.py`.

The previous classifier treated this reason as capacity pressure:

```text
outstanding requests grew across consecutive windows
```

That rule is removed. It was too sensitive to normal open-loop stochastic arrival variance and caused false unstable or uncertain decisions on trials that visually and statistically behaved as stable.

## Triggering Evidence

The false-positive behavior was observed in these Qwen3-4B-Thinking trials:

```text
results/mst/qwen-qwen3-4b-thinking-2507/live-sharegpt-workload-context/server-f57f96ac974f/trials/trial_008_openloop_r5_25322
results/mst/qwen-qwen3-4b-thinking-2507/live-sharegpt-workload-context/server-f57f96ac974f/trials/trial_009_openloop_r5_25322
```

The outstanding-request plots and `windows.csv` show oscillation and high variance, not a robust monotonic backlog increase. Re-analysis after the fix classifies:

```text
trial_008_openloop_r5_25322: stable, medium confidence
trial_009_openloop_r5_25322: stable, high confidence
```

Across existing result traces, most events carrying the removed consecutive-increase reason reclassify away from unstable/uncertain under the corrected policy.

## Current Backlog Trend Rule

Outstanding-request backlog pressure is now recognized only when all of these hold:

```text
Theil-Sen slope > max_positive_backlog_slope
Mann-Kendall p < backlog_trend_alpha
fitted end-to-end relative increase >= min_backlog_relative_increase
fitted end-to-end delta >= min_backlog_growth_for_hard_pressure
```

Defaults:

```yaml
max_positive_backlog_slope: 0.10
backlog_trend_alpha: 0.05
min_backlog_relative_increase: 0.10
min_backlog_growth_for_hard_pressure: 2.0
```

The classifier records these key metrics when available:

```text
outstanding_end_slope_per_s
outstanding_end_delta
outstanding_end_relative_increase
outstanding_end_mann_kendall_p
outstanding_end_slope_ci_low
outstanding_end_slope_ci_high
```

## SciPy Dependency

`scipy` is a hard dependency for this policy. The implementation uses:

```python
scipy.stats.theilslopes
scipy.stats.kendalltau
```

There is no local fallback implementation. If SciPy is missing, import should fail immediately in the shared uv environment.

## Completion/Arrival Ratio

`completion_arrival_ratio < 1 - completion_arrival_tolerance` is no longer independent capacity-pressure evidence. It is retained as supporting evidence and attached only when robust outstanding backlog pressure is already present.

Reason: a short trial can end after a burst of arrivals or long-output samples. In that case, completion lag can be a sampling artifact rather than proof that service rate is below arrival rate.

## SLO Priority

TTFT/TPOT SLO violations remain top-priority classification evidence. If a trial violates the configured SLO, the status should be `slo_violation` even if other instability evidence is also present.
