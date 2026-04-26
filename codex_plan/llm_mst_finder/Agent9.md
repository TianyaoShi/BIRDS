# Implement Reporting And Result Comparison
## Goal

Generate useful artifacts from saved MST finder outputs. Reports consume existing trial/search/analysis artifacts; they do not rerun trials, mutate search results, launch servers, or perform server configuration search.

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

For externally orchestrated result comparison:
```
max sustainable req/s by result directory
max output tok/s by result directory
bottleneck class by result directory
server metadata comparison table
```
## Report sections

final_report.md should include:

1. Workload definition
2. Workload/model context compatibility summary
3. Server configuration
4. Search trace
5. Closed-loop scouting result
6. Open-loop stability boundary
7. Max no-drift request rate
8. Max SLO-satisfying request rate
9. Bottleneck diagnosis
10. Recommended next orchestration action
11. Limitations

## Local consistency constraints

+ Reports consume saved artifacts; they should not rerun trials or mutate search results.
+ Missing plots should fail report generation unless plots are explicitly disabled.
+ Markdown and JSON reports must agree on headline rates, bottleneck class, confidence, and trial statuses.
+ Reports must include `context_policy`, tokenizer source, max model length, and skipped/truncated sample counts. If a trial was `invalid_workload`, make that prominent and do not present its rate as an overload boundary.
+ Result comparison is metadata validation and presentation only. Do not add `config-sweep`, server restart, health-check, or GPU orchestration logic.
+ Multiple result directories are comparable only when workload identity, context policy, SLOs, duration, search settings, model id, and declared server metadata are present. If comparability cannot be proven, fail report generation or mark the comparison as invalid explicitly; do not rank incomparable runs.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
