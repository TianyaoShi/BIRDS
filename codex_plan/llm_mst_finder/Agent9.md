# Implement plotting and reporting
## Goal

Generate useful artifacts.

## Files

```
plotting.py
reporting.py
```
## Required plots

For each trial:

```
arrival rate vs completion rate over time
outstanding requests over time
p50/p90/p99 TTFT over windows
p50/p90/p99 TPOT over windows
output tok/s over time
KV-cache usage over time
running/waiting/swapped requests over time
```

For search:

```
tested request rate vs classification
request rate vs p90 TTFT
request rate vs p90 TPOT
request rate vs output tok/s
request rate vs queue drift
```

For config sweep:
```
max sustainable req/s by config
max output tok/s by config
bottleneck class by config
```
## Report sections

final_report.md should include:

1. Workload definition
2. Server configuration
3. Search trace
4. Closed-loop scouting result
5. Open-loop stability boundary
6. Max no-drift request rate
7. Max SLO-satisfying request rate
8. Bottleneck diagnosis
9. Recommended next config sweep
10. Limitations

## Local consistency constraints

+ Reports consume saved artifacts; they should not rerun trials or mutate search results.
+ Missing plots should fail report generation unless plots are explicitly disabled.
+ Markdown and JSON reports must agree on headline rates, bottleneck class, confidence, and trial statuses.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
