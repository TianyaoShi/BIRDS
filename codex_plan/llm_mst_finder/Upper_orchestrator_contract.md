# MST Finder Contract For Upper Orchestrator Agents

`llm_mst_finder` estimates the max sustainable request rate for **one live OpenAI-compatible endpoint under one fixed serving configuration**. The upper orchestrator owns server lifecycle, GPU allocation, model/workload selection, and server-config search.

## Package Scope

MST finder owns:

- workload sampling and context validation;
- open-loop / closed-loop trial execution;
- optional vLLM Prometheus `/metrics` polling;
- fixed-window aggregation;
- stability and bottleneck classification;
- adaptive request-rate search;
- JSON/Markdown reporting.

MST finder does **not**:

- start, stop, restart, or health-check vLLM;
- mutate `max_num_seqs`, `max_num_batched_tokens`, TP, quantization, model, or GPU allocation;
- search serving configuration space;
- persist response text in normal trial artifacts.

Interpret every result as:

```text
max sustainable rate for this workload + model + endpoint + SLO policy + fixed serving config
```

## Environment

Use the shared env:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli ...
```

Do not create another venv.

## Required Inputs

### Endpoint

Required:

- `--base-url`, e.g. `http://127.0.0.1:8000`
- `--endpoint`, `/v1/completions` or `/v1/chat/completions`
- `--model`, exact served model id
- `--workload`, workload YAML path
- `--output-dir`, result directory

Use `/v1/chat/completions` for chat/instruct semantic quality checks. `/v1/completions` is still valid for raw throughput profiling.

### Workload YAML

Supported dataset types:

- `synthetic-fixed`
- `synthetic-distribution`
- `jsonl`
- `sharegpt`
- `hf`

Real datasets (`jsonl`, `sharegpt`, `hf`) require `context_policy`.

Minimal real workload:

```yaml
name: workload-name
dataset:
  type: sharegpt        # or jsonl / hf
  path: /path/to/data
tokenizer: google/gemma-4-E4B-it
sampling:
  seed: 1
  num_requests: 3000
  conversation_mode: single_turn
  prompt_len:
    mode: from_dataset
  output_len:
    mode: from_dataset
request:
  stream: true
  temperature: 0.0
context_policy:
  max_model_len: 131072
  tokenizer_source: vllm_model_config
  over_limit: fail      # fail | skip_sample | truncate_prompt
```

HF example:

```yaml
dataset:
  type: hf
  path: allenai/WildChat
  split: train
  conversation_field: conversation
```

If `over_limit` is `skip_sample` or `truncate_prompt`, skipped/truncated counts are recorded in metadata. Silent skip/truncate is not allowed.

### Chat Conversation Workloads

For ShareGPT/WildChat-style datasets, conversation semantics are workload identity. Supported modes are:

- `single_turn`: first valid user/human -> assistant/gpt pair, independent requests.
- `multi_turn_prefix`: independent prefix -> next assistant requests, useful for longer-prompt studies.
- `session_replay`: realistic chatbot workload. Emit all valid assistant-target turns, preserve order within a session, and interleave sessions.

Recommended realistic chat config:

```yaml
sampling:
  conversation_mode: session_replay
  turn_selection: all_valid
  include_assistant_history: true
  min_prompt_turns: 1
  max_prompt_turns: 16
traffic:
  session_ordering: preserve_within_session
  session_interleaving: shuffled_sessions
  per_session_think_time_s: 0
```

The orchestrator should treat different `conversation_mode` values as different workloads. `session_replay` preserves prefix-cache reuse opportunity but does not guarantee cache hits; cache behavior remains a serving/runtime property.

### SLO Policy

Only TTFT and TPOT SLOs exist:

```bash
--ttft-slo-ms 2000
--tpot-slo-ms 80
--ttft-slo-field ttft_p90_ms|ttft_p99_ms|ttft_p50_ms
--tpot-slo-field tpot_p90_ms|tpot_p99_ms|tpot_p50_ms
```

Use `none` to disable:

```bash
--tpot-slo-ms none
```

There is no E2E SLO interface. Do not add one in the orchestrator.

### Server Metadata

Pass serving config metadata whenever known:

```bash
--max-num-seqs 256
--max-num-batched-tokens 2048
```

or:

```bash
--server-metadata-file server_metadata.json
```

This improves bottleneck classification. MST finder records metadata but never changes server config.

### Metrics

Prefer:

```bash
--metrics-url http://127.0.0.1:8000/metrics
--metrics-interval-s 1
```

Useful metrics include running/waiting/swapped requests, KV cache usage, preemptions, and prompt/generation token counters. Missing optional metrics lower confidence when relevant.

## Main Commands

Single trial:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli run-trial \
  --trial-id TRIAL_ID \
  --mode open-loop \
  --request-rate 10 \
  --duration-s 90 \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model MODEL \
  --workload WORKLOAD.yaml \
  --metrics-url http://127.0.0.1:8000/metrics \
  --output-dir results/trial_r10
```

Adaptive search:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli search \
  --search-id SEARCH_ID \
  --search-mode open-loop \
  --output-dir results/mst/MODEL_WORKLOAD_SERVER \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model MODEL \
  --workload WORKLOAD.yaml \
  --trial-min-duration-s 90 \
  --trial-max-duration-s 180 \
  --final-confirmation-duration-s 180 \
  --rate-precision 0.05 \
  --initial-request-rate 5 \
  --max-request-rate 50 \
  --metrics-url http://127.0.0.1:8000/metrics \
  --max-num-seqs 256 \
  --max-num-batched-tokens 2048
```

Analyze / report:

```bash
... -m llm_mst_finder.cli analyze --trial-dir results/trial_r10
... -m llm_mst_finder.cli report --result-dir results/mst/MODEL_WORKLOAD_SERVER
```

## Output Layout

Trial directory:

```text
trial_xxx/
  request_records.jsonl
  server_metrics.jsonl       # if metrics-url configured
  windows.csv
  summary.json
  analysis.json              # after analyze/search
```

Search directory:

```text
RESULT_DIR/
  search_trace.json
  trials/
    trial_000_...
    trial_001_...
  final_report.json          # after report
  final_report.md            # after report
  plots/                     # if plots enabled
```

## Key Output Formats

### `summary.json`

Top-level shape:

```json
{
  "config": { "trial_id": "...", "metadata": {} },
  "summary": {
    "status": "completed|aborted_safety",
    "requested_request_rate": 10.0,
    "actual_send_rate": 9.98,
    "successful_completion_rate": 9.7,
    "error_rate": 0.0,
    "max_observed_outstanding": 123,
    "benchmark_metrics": {
      "request_throughput": 10.0,
      "prompt_token_throughput": 1000.0,
      "generation_token_throughput": 800.0,
      "prompt_length_summary": {},
      "output_length_summary": {}
    }
  }
}
```

`config.metadata` contains workload metadata, context validation report, stability policy, and supplied server metadata.

### `request_records.jsonl`

One JSON object per request. Important fields:

- `success`, `error`
- `scheduled_send_ts`, `actual_send_ts`, `first_token_ts`, `end_ts`
- `prompt_len`, `expected_output_len`, `actual_output_len`
- `ttft_s`, `tpot_s`, `e2e_s`, `itl_s`
- `metadata` with workload source fields

Response text is intentionally absent.

### `windows.csv`

Fixed-window fields include:

- arrivals, completions, failures;
- arrival/completion/error rates;
- outstanding start/end/mean/slope;
- TTFT/TPOT percentiles;
- prompt/generation/total token throughput;
- server running/waiting/swapped means;
- KV cache usage;
- preemption deltas;
- prompt/output length means.

### `analysis.json`

Shape:

```json
{
  "trial_id": "trial_001_openloop_r10",
  "trial_validity": "valid|invalid_workload|client_limited|metrics_invalid",
  "validity_reasons": [],
  "stability": {
    "status": "stable|unstable|slo_violation|uncertain|aborted_safety",
    "confidence": "high|medium|low",
    "reasons": [],
    "key_metrics": {}
  },
  "bottleneck": {
    "bottleneck_class": "scheduler_cap|prefill_compute|decode_bandwidth|kv_cache|slo_limited|mixed|unknown|client_limited",
    "confidence": "high|medium|low",
    "evidence": []
  }
}
```

`invalid_workload`, `client_limited`, and `metrics_invalid` trials must not be used for search decisions.

### `search_trace.json`

Important fields:

- `config`: search config and metadata;
- `events`: trial purpose, rate/concurrency, summary, analysis;
- `bounds`: final low/high open-loop rates;
- `result`: final search result.

Key `result` fields:

```json
{
  "max_no_drift_request_rate": 9.0625,
  "max_slo_satisfying_request_rate": 9.0625,
  "rate_precision": 0.05,
  "confirmation_trial_id": "trial_007_openloop_r9.0625",
  "termination_reason": "confirmed_stable|scheduler_config_limited",
  "bottleneck_class": "decode_bandwidth",
  "confidence": "medium",
  "reasons": []
}
```

## Search Semantics

- `stable`: updates low bound.
- `unstable`, `slo_violation`, `aborted_safety`: update high bound.
- `uncertain`: rerun same rate once with extended duration.
- still `uncertain`: conservative high bound if a stable low bound exists; otherwise convergence error.

Raw outstanding-request movement alone is not decisive instability. Backlog pressure requires the stability classifier's robust trend gate: SciPy Theil-Sen slope above threshold, Mann-Kendall `p < backlog_trend_alpha`, fitted relative increase above threshold, and fitted absolute delta above threshold. The removed `outstanding requests grew across consecutive windows` reason must not be used by orchestrators as overload evidence.

Completion/arrival lag is supporting evidence only. It should not by itself cause an orchestrator to conclude the server is overloaded, because short open-loop trials can end after stochastic arrival bursts or long-output sample segments.

TTFT/TPOT SLO violations remain top-priority decision evidence. A trial can be queue-stationary and still fail as `slo_violation`.

`termination_reason=scheduler_config_limited` means the orchestrator should consider rerunning with different server scheduler settings, then launch a new MST search for that new serving configuration.

## Orchestrator Responsibilities

The orchestrator should:

- launch/stop vLLM and allocate GPUs externally;
- keep result dirs separated by model, workload, SLO policy, and server config;
- pass server metadata into MST finder;
- compare only compatible result directories;
- treat fail-fast errors as signals to fix orchestration inputs;
- rerun MST finder after changing server config, model, dataset, or SLO policy.

Recommended directory pattern:

```text
results/mst/{model_slug}/{dataset_slug}/{server_config_slug}/
```

## Common Failure Modes

- `invalid_workload`: fix context policy/tokenizer/max model length/workload sampling.
- `client_limited`: client did not issue configured load; increase client capacity or lower tested rate.
- `metrics_invalid`: metrics were configured but missing/inconsistent.
- `scheduler_config_limited`: adjust scheduler-related server settings externally.
- repeated `uncertain`: increase duration, improve metrics, or use a more representative workload sample.
