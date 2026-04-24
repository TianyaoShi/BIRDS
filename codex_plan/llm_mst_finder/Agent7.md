# Implement search controller
## Goal

Minimize trial-and-test time while estimating max sustainable rate.

## Files
```
search.py
Search modes
```
Support:

```
closed-loop
open-loop
hybrid
```

Recommended default:
```
hybrid
```
## Hybrid algorithm
### Phase 1: closed-loop scouting

Run concurrency sweep:

N = 1, 2, 4, 8, 16, 32, ...

Stop when:
```
request throughput plateaus
or TPOT SLO violation
or KV/preemption wall
or configured max N reached
```
This gives approximate capacity:
```
closed_loop_req_s_peak
closed_loop_output_tok_s_peak
```
### Phase 2: open-loop bracketing

Use closed-loop estimate as starting point.

Example:
```
lambda_start = 0.6 * closed_loop_req_s_peak
lambda_high_candidate = 1.2 * closed_loop_req_s_peak
```
Run open-loop trials.

Find:
```
lambda_low = highest stable/SLO-satisfying rate
lambda_high = lowest unstable/SLO-violating rate
```
If no closed-loop estimate is available, use exponential search:

$$0, 2\lambda_0, 4\lambda_0, ...$$
### Phase 3: binary search

While:
```
(lambda_high - lambda_low) / lambda_low > precision
```
run:
```
lambda_mid = (lambda_low + lambda_high) / 2
```
Update bounds based on stability result.

### Phase 4: final confirmation

Run longer trial at:
```
lambda_low
```
and optionally:
```
0.95 * lambda_low
```
Return:
```
max_no_drift_rate
max_slo_rate
```
### Pseudocode
```Python
def hybrid_search(config):
    closed = run_closed_loop_sweep(config)

    bracket = find_open_loop_bracket(
        start_rate=0.6 * closed.peak_req_s,
        high_rate=1.2 * closed.peak_req_s,
    )

    while relative_width(bracket) > config.rate_precision:
        rate = midpoint(bracket)
        trial = run_open_loop_trial(rate)
        result = classify_stability(trial)

        if result.status == "stable":
            bracket.low = rate
        elif result.status in {"unstable", "slo_violation", "aborted_safety"}:
            bracket.high = rate
        else:
            trial = extend_or_repeat(rate)
            update_bracket_conservatively(trial)

    confirmation = run_open_loop_trial(
        rate=bracket.low,
        duration=config.final_confirmation_duration_s,
    )

    return FinalResult(...)
```

## Example
You may inspect older binary-search code if it exists locally, but do not copy health-check, restart, broad-retry, or defensive fallback machinery into this package.

## Local consistency constraints

+ Search must call the shared `TrialRunner`, `StabilityClassifier`, and `BottleneckClassifier` contracts.
+ Unknown or contradictory trial statuses should update bounds conservatively or request a repeat, but implementation errors should raise.
+ Every tested rate and classification must be written to `search_trace.json`.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
