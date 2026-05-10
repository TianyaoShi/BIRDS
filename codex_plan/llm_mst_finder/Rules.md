# Critical Design And Execution Rules

These rules override individual agent plans when they conflict.

## Agent Continuation Discipline

- At the start of every new dispatch, resumed task, or continuation after another agent has changed files, re-read this `Rules.md` before editing.
- Also re-read `Implementation_order.md`, `OVERVIEW.md`, and the relevant `AgentN.md` brief for the current scope.
- Treat this file as live. If a rule changed since your last work session, follow the newest version and mention the rule-relevant adjustment in your final handoff.
- Do not rely on memory of earlier prompts when it conflicts with these docs.

## Local Environment

- Use the shared uv-managed environment at `/local/scratch/a/shi676/.venv`.
- Run Python commands with `/local/scratch/a/shi676/.venv/bin/python`.
- Run package commands with `PYTHONPATH=/local/scratch/a/shi676/arr26/profiler`.
- Do not create a second virtual environment inside the repo.
- Do not install packages globally. If a dependency is truly missing, use:

```bash
uv pip install --python /local/scratch/a/shi676/.venv/bin/python PACKAGE
```

- `scipy` is required for MST stability trend statistics. The stability classifier should fail immediately if `scipy.stats` is unavailable; do not add a local statistical fallback.

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
- Do not silently skip or truncate real workload samples. Context-limit handling must be controlled by an explicit `context_policy`.
- Do not use workload sampling tokenizers such as `whitespace` as proof that a prompt fits a production model context. Context validation must use the serving/model-compatible tokenizer.
- `whitespace` and other toy tokenizers are allowed only in synthetic/offline unit tests or clearly labeled example workloads. They must not be used as production context validators for JSONL, ShareGPT, or any live profiling run.
- If model max context length or a compatible tokenizer cannot be determined for a real dataset run, fail before trial dispatch unless the user explicitly disables validation and accepts the invalid-workload risk in recorded metadata.
- Chat dataset conversation semantics must be explicit. `single_turn`, `multi_turn_prefix`, and `session_replay` are distinct workload identities and must be recorded in metadata/reports.
- `session_replay` must preserve request order within each conversation/session. Do not shuffle turns inside one `session_id`; interleave different sessions instead.
- Do not claim prefix-cache reuse from session replay by assumption alone. The workload may preserve reuse opportunity, but actual reuse is server-state dependent unless directly measured.

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

Comparisons across server configurations may be reported only when those result directories were produced by an external orchestrator and carry enough metadata to prove they are comparable. MST finder itself must not claim to have searched server configuration space.

5. Separate invalid workload samples from throughput instability.

Context-length errors such as:

```text
prompt_tokens + requested_output_tokens > max_model_len
```

mean the workload is invalid for the selected model configuration. They must not be counted as evidence of overload, SLO drift, KV saturation, or scheduler bottleneck. The trial should be rejected before dispatch when possible. If discovered from server responses, analysis should mark the trial as `invalid_workload` and exclude it from search decisions.
