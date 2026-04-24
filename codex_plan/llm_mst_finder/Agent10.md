# Testing plan
## Unit tests
### `test_windowing.py`
Synthetic request records where:
```
arrivals = completions
outstanding flat
```
should produce stable windows.

Synthetic overloaded records where:
```
arrivals > completions
```
should produce positive outstanding slope.

### `test_stability.py`

Create fixture windows for:
```
stable
TTFT drift only
TPOT drift
KV preemption
SLO violation without drift
scheduler cap
```

Expected outputs:
```
stable -> stable
TTFT drift -> unstable
TPOT drift -> unstable
preemption -> unstable
SLO violation -> slo_violation
scheduler cap -> unstable + bottleneck scheduler_cap
```

### `test_search.py`

Mock run_trial(rate) with threshold behavior:
```
stable if rate <= 10
unstable if rate > 10
```
Verify binary search returns approximately 10.

### `test_prometheus_parser.py`

Feed simplified Prometheus text and verify extraction of:
```
num_requests_running
num_requests_waiting
kv_cache_usage_perc
generation_tokens_total
request_success_total
```

## Local test commands

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m compileall profiler/llm_mst_finder tests/llm_mst_finder
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m pytest tests/llm_mst_finder -q
```

## Local consistency constraints

+ Unit tests must use synthetic records, synthetic metrics text, and local temporary files.
+ Do not require a live vLLM server, GPU, NVML, Hugging Face download, or network access in default tests.
+ Add live-server smoke tests only behind explicit opt-in.
+ Tests should assert fail-fast behavior for malformed configs and invalid records; do not encode defensive placeholder behavior as expected output.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
