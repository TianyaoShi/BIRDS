# Critical Design And Execution Rules

These rules override individual agent plans when they conflict.

## Local Environment

- Use the shared uv-managed environment at `/local/scratch/a/shi676/.venv`.
- Run Python commands with `/local/scratch/a/shi676/.venv/bin/python`.
- Run package commands with `PYTHONPATH=/local/scratch/a/shi676/arr26/profiler`.
- Do not create a second virtual environment inside the repo.
- Do not install packages globally. If a dependency is truly missing, use:

```bash
uv pip install --python /local/scratch/a/shi676/.venv/bin/python PACKAGE
```

- Default unit tests must be offline and synthetic. They must not require a live vLLM server, GPU, NVML, Hugging Face download, or external network access.
- Use `tmp_path` or another test-local directory for generated artifacts. Do not write persistent results during unit tests.
- Before handing off a code slice, run the narrow tests for that slice and then:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m compileall profiler/llm_mst_finder tests/llm_mst_finder
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m pytest tests/llm_mst_finder -q
```

## Package Layout

- Implement the import package at `profiler/llm_mst_finder/`.
- Tests live in `tests/llm_mst_finder/`.
- Use `llm_mst_finder.cli` as the only CLI module.
- Keep `profiler/benchmark_serving.py`, `profiler/backend_request_func.py`, and `profiler/gpu_monitor.py` as reference or adapter sources. Do not turn the old benchmark script into the core runtime.

## Fail-Fast Implementation Style

- Prefer explicit exceptions over defensive fallback behavior.
- Do not add broad `except Exception` blocks that convert programmer errors into successful outputs.
- Do not fabricate placeholder request outputs after internal failures. Request-level HTTP/API failures may be recorded as failed `RequestRecord`s; schema, parsing, scheduling, or implementation errors should raise.
- Do not retry, restart, health-check, or auto-recover the vLLM server in v1.
- Do not silently ignore missing required config fields, malformed workload YAML, invalid metric names, or inconsistent timestamps.
- Do not silently clamp request rates, durations, concurrency, token lengths, or SLO thresholds. Reject invalid values.
- Do not silently throttle open-loop traffic. If client scheduling delay or a safety outstanding cap prevents the configured arrival rate, classify the trial as `client_limited` or `aborted_safety`.
- Optional subsystems must be explicit. If GPU monitoring is not requested, skip it. If it is requested and unavailable, fail immediately.
- Missing optional Prometheus metrics may be represented as `None`, but the analysis must add a reason and lower confidence when those metrics affect classification.

## Measurement Rules

1. Open-loop rate must not be silently throttled.

Bad:

```text
request-rate = 20 req/s
max_concurrency = 64
actual sent rate silently drops to 12 req/s
reported stable at 20 req/s
```

Correct:

```text
configured rate = 20 req/s
actual sent rate measured separately
if actual sent rate falls below configured rate due to client limits, classify trial invalid/client-limited
```

2. Stability is time-series stationarity, not aggregate p99.

A trial with acceptable aggregate p99 can still be unstable if late-window TTFT is much worse than early-window TTFT.

Required:

```text
windowed TTFT
windowed TPOT
windowed outstanding queue
windowed server waiting/running/swapped requests
```

3. Report both request/s and token/s.

For LLMs, request/s alone is insufficient. Always report:

```text
request throughput
successful request throughput
prompt tokens/s
generation tokens/s
total tokens/s
mean/median/p90/p99 prompt length
mean/median/p90/p99 output length
```

4. Treat `max_num_seqs` and `max_num_batched_tokens` as first-class profiling variables.

The final result should always say:

```text
max sustainable rate under this serving configuration
```

not:

```text
absolute max sustainable rate of the model/GPU
```

unless a configuration sweep was performed.
