# Implementation Order

Use this order for dispatch and integration. It intentionally collapses some generated design into fewer integration milestones, while preserving the `Agent*.md` files as scoped briefs.

For every dispatch or continuation, use `codex_plan/llm_mst_finder/Dispatch_prompt_template.md`. Agents must re-read `Rules.md` from disk each time they resume work, because these rules are expected to evolve as live profiling exposes new project constraints.

## Milestone 0 - Package Skeleton And Contracts

Owner: lead or Agent 1 before parallel work starts.

Deliver:

```text
profiler/llm_mst_finder/__init__.py
profiler/llm_mst_finder/records.py
profiler/llm_mst_finder/cli.py
tests/llm_mst_finder/
```

Define dataclasses and stable function boundaries first. Other agents must import these contracts instead of inventing local duplicates.

## Milestone 1 - Single Trial Runner

Agents: 1, 2, 3, 4.

Deliver:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli run-trial --mode open-loop --request-rate X

PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli run-trial --mode closed-loop --concurrency N
```

Must save:

```text
request_records.jsonl
server_metrics.jsonl
windows.csv
summary.json
```

No adaptive search yet. If a live server is unavailable, unit tests should still cover the load generator, record serialization, parser, and window aggregation with synthetic data.

## Milestone 1.5 - Workload/Model Context Compatibility

Agents: 2 with Agent 1 integration. The lead agent owns final integrated test review.

Deliver:

```text
profiler/llm_mst_finder/model_context.py
context_policy support in workload/config loading
pre-trial context validation in CLI/search/trial setup
```

Required behavior:

```text
prompt_tokens + requested_output_tokens <= max_model_len
```

must be checked with a serving/model-compatible tokenizer before real dataset trials. Default policy is `fail`. Explicit policies may be:

```YAML
context_policy:
  max_model_len: 4096
  tokenizer_source: vllm_model_config
  tokenizer: meta-llama/Llama-2-7b-chat-hf
  over_limit: fail | skip_sample | truncate_prompt
  truncation_side: left
```

`skip_sample` and `truncate_prompt` must record counts and affected source indexes in metadata. Silent skip/truncation is forbidden. If the server still returns context-length validation failures, the trial analysis marks `trial_validity=invalid_workload` and search excludes that trial from bounds.

## Milestone 2 - Trial Analysis

Agents: 5 and 6.

Deliver:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli analyze --trial-dir results/trial_001
```

Must output:

```text
trial_validity: valid | invalid_workload | client_limited | metrics_invalid
stable | unstable | slo_violation | uncertain | aborted_safety
reasons
bottleneck_class
confidence
```

## Milestone 3 - Hybrid Search

Agent: 7.

Deliver:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli search --search-mode hybrid
```

Must output final max sustainable rate under the supplied workload and server configuration.

## Milestone 4 - Reporting And Result Comparison

Agent: 9.

Deliver:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli report --result-dir results/run_001
```

Reporting may compare multiple completed result directories when they were produced by an external orchestrator, but MST finder does not own server configuration search. There is no `config-sweep` module or CLI in this package. Server restarts, GPU allocation, dataset/model scheduling, and non-default vLLM configuration experiments belong to an upper orchestration layer.

## Milestone 5 - Test Hardening

Owner: each implementation agent for its own slice, plus lead integration review. Do not dispatch a separate Agent 10.

Required checks:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m compileall profiler/llm_mst_finder tests/llm_mst_finder
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m pytest tests/llm_mst_finder -q
```

Add live-server smoke tests only behind an explicit opt-in marker or environment variable.

## Redundancy Removed From Generated Plan

- Use `run-trial`, `analyze`, `search`, and `report` subcommands only. Do not add separate `profile_once`, `profile_search`, `analyze_trial`, or `config-sweep` executables.
- Keep `TrialRunner` as the shared orchestrator for open-loop and closed-loop trials.
- Keep dataclasses in `records.py`; do not redefine request/window/result schemas in each module.
- Keep Prometheus text parsing inside `metrics_polling.py` unless it becomes large enough to justify `prometheus_parser.py`.
- Treat plotting as report support, not a separate runtime path.
- Keep server launching/restarting out of v1.

## Agent Model And Reasoning Recommendations

| Agent | Scope | Suggested model | Reasoning effort |
| --- | --- | --- | --- |
| Agent 1 | Async client, open/closed loadgen, records adapter | `gpt-5.3-codex` | high |
| Agent 2 | Workload YAML, deterministic sampling, model context validation | `gpt-5.3-codex` | high |
| Agent 3 | Prometheus polling and metric normalization | `gpt-5.3-codex` | medium |
| Agent 4 | Fixed-window aggregation and CSV/summary output | `gpt-5.3-codex` | high |
| Agent 5 | Stability classification and trend rules | `GPT-5.5` | high |
| Agent 6 | Bottleneck classification and evidence strings | `GPT-5.5` | high |
| Agent 7 | Hybrid search controller and convergence tests | `GPT-5.5` | high |
| Agent 8 | Retired: metadata-only config sweep is out of scope | do not dispatch | n/a |
| Agent 9 | Markdown/JSON reporting, plots, and optional comparison of externally produced result dirs | `gpt-5.4` | medium |
| Agent 10 | Retired: lead agent owns final integrated testing | do not dispatch | n/a |

Use the lead agent for final integration review with `GPT-5.5` at high reasoning effort. Reserve `xhigh` for a focused debugging pass only if stability/search behavior is contradictory after tests.
