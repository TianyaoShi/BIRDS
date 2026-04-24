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

### `test_model_context.py`

Create fixture samples and fake tokenizers for:
```
fits context
over limit with over_limit=fail
over limit with over_limit=skip_sample
over limit with over_limit=truncate_prompt
missing max_model_len
missing model-compatible tokenizer
```

Expected outputs:
```
fits context -> unchanged samples
fail -> raises before trial dispatch
skip_sample -> removes only explicit over-limit samples and records counts/source indexes
truncate_prompt -> shortens prompt and records counts/source indexes
missing metaconfig -> raises
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
+ Tests should assert context-limit failures are invalid workload conditions, not stability or bottleneck evidence.
+ Follow `Rules.md` for `/local/scratch/a/shi676/.venv`, `PYTHONPATH`, and fail-fast behavior.
