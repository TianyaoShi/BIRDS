# LLM MST Finder

`llm_mst_finder` estimates the maximum sustainable external request rate for one live OpenAI-compatible vLLM endpoint under one fixed serving configuration.

The result should be read as:

```text
max sustainable rate for this workload + model + endpoint + SLO policy + server config
```

It is not a vLLM launcher, GPU scheduler, or server-configuration searcher. An upper orchestrator should own GPU allocation, vLLM lifecycle, model selection, workload selection, and reruns with different serving flags such as `max_num_seqs` or `max_num_batched_tokens`.

## Environment

Use the shared uv-managed environment:

```bash
export PYTHONPATH=/local/scratch/a/shi676/arr26/profiler
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli --help
```

`scipy` is required for stability trend statistics. The package intentionally fails if `scipy.stats` is unavailable.

## What It Does

The package provides:

- workload loading and context validation;
- open-loop and closed-loop load generation;
- request JSONL artifact recording;
- optional vLLM Prometheus `/metrics` polling;
- fixed-window aggregation;
- stability and bottleneck classification;
- adaptive search over request rate;
- JSON/Markdown reports and plots.

Closed-loop trials are scouting only. The reported MST is based on open-loop request-rate trials.

## Required Inputs

For live profiling, specify:

- `--base-url`, for example `http://127.0.0.1:8000`
- `--endpoint`, usually `/v1/chat/completions` or `/v1/completions`
- `--model`, exact served model id
- `--workload`, workload YAML path
- `--output-dir`, artifact directory

Pass metrics and server metadata when available:

```bash
--metrics-url http://127.0.0.1:8000/metrics
--metrics-interval-s 1
--max-num-seqs 256
--max-num-batched-tokens 2048
```

or:

```bash
--server-metadata-file server_metadata.json
```

Server metadata improves bottleneck classification. MST finder records it but never changes server config.

## Workloads

Supported dataset types include:

- `synthetic-fixed`
- `synthetic-distribution`
- `jsonl`
- `sharegpt`
- `hf`

Real datasets require context validation. Minimal ShareGPT-style example:

```yaml
name: live_sharegpt
dataset:
  type: sharegpt
  path: /path/to/ShareGPT.json
tokenizer: Qwen/Qwen3-8B
sampling:
  seed: 1
  num_requests: 3000
  prompt_len:
    mode: from_dataset
  output_len:
    mode: from_dataset
request:
  stream: true
  temperature: 0.0
context_policy:
  max_model_len: 32768
  tokenizer_source: vllm_model_config
  over_limit: fail
```

Reasoning datasets with only final-answer labels should not use
`output_len.mode: from_dataset`, because the final answer length is not the
reasoning trace length. Use a natural-stop cap instead:

```yaml
sampling:
  output_len:
    mode: natural_until_eos
    max_tokens: 2048
request:
  ignore_eos: false
```

This sends `max_tokens: 2048` as a safety cap while allowing the model to stop
on EOS. Context validation reserves the same cap against `max_model_len`.

HF example:

```yaml
dataset:
  type: hf
  path: allenai/WildChat
  split: train
  conversation_field: conversation
```

HF datasets are sampled with deterministic reservoir sampling over the streamed split, using `sampling.seed`. This avoids the previous prefix bias from taking the first streamed rows. Add `dataset.max_scan_rows` only when you intentionally want to cap the scan and accept uniformity over the scanned prefix rather than the full split.

For conversation-style HF rows, the current adapter uses the first user/human turn as the prompt and the first assistant/gpt turn as the reference completion length. It does not serialize full multi-round chat history into the request prompt.

Context policy rules:

- `prompt_tokens + requested_output_tokens` must fit the selected model context.
- Real workload validation must use the serving/model-compatible tokenizer.
- `over_limit: fail` is the default and safest policy.
- `skip_sample` and `truncate_prompt` are allowed only when explicitly configured and are recorded in metadata.
- `whitespace` tokenization is rejected; workloads must name a real tokenizer or an explicit local test tokenizer.
- If context failures are discovered from server responses, analysis marks the trial `invalid_workload` and excludes it from search decisions.

## SLO Policy

Only TTFT and TPOT SLOs are supported:

```bash
--ttft-slo-ms 2000
--tpot-slo-ms 80
--ttft-slo-field ttft_p90_ms
--tpot-slo-field tpot_p90_ms
```

Use `none` to disable a threshold:

```bash
--tpot-slo-ms none
```

TTFT has two policy modes:

- `--ttft-slo-mode static`: the default. It uses `--ttft-slo-ms` exactly as
  before.
- `--ttft-slo-mode length_scaled`: request-level LongBench profile policy. It
  uses `metadata.profile` and `prompt_len` from materialized LongBench bucket
  workloads, then checks the configured TTFT percentile as a normalized
  threshold ratio.

LongBench bucketized workloads can also use profile-specific static fallbacks:

```bash
--ttft-slo-mode static --longbench-ttft-static-preset default
--ttft-slo-mode static --longbench-ttft-static-preset tight
--ttft-slo-mode static --longbench-ttft-static-preset relaxed
```

The preset TTFT seconds are ordered by profile
`long_output_summarization`, `medium_output_summarization`,
`medium_answer_rag_qa`, `short_answer_document_qa`:

- default: `35 / 30 / 20 / 15`
- tight: `25 / 22 / 15 / 10`
- relaxed: `45 / 40 / 30 / 20`

There is no E2E SLO interface.

## Commands

Run one open-loop trial:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli run-trial \
  --trial-id trial_r10 \
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

Run adaptive search:

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

Analyze and report:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli analyze \
  --trial-dir results/trial_r10

PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli report \
  --result-dir results/mst/MODEL_WORKLOAD_SERVER
```

Inspect a workload before profiling:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m llm_mst_finder.cli inspect-workload \
  --workload experiments/workloads/wildchat_hf.yaml \
  --model google/gemma-4-E4B-it \
  --sample-size 4096 \
  --output results/mst/wildchat_inspection.json
```

This reports sampled prompt/output token length distributions, HF scan/usable/skipped counts, tokenizer/model-context metadata, and suggested search-duration overrides.

## Output Artifacts

Single trial:

```text
trial_xxx/
  request_records.jsonl
  server_metrics.jsonl       # if metrics-url configured
  windows.csv
  summary.json
  analysis.json              # after analyze/search
```

Search:

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

Normal trial artifacts do not persist response text.

## Key Result Fields

`analysis.json`:

```json
{
  "trial_validity": "valid|invalid_workload|client_limited|metrics_invalid",
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

`search_trace.json` result:

```json
{
  "max_no_drift_request_rate": 9.0625,
  "max_slo_satisfying_request_rate": 9.0625,
  "rate_precision": 0.05,
  "confirmation_trial_id": "trial_007_openloop_r9_0625",
  "termination_reason": "confirmed_stable",
  "bottleneck_class": "decode_bandwidth",
  "confidence": "medium",
  "reasons": []
}
```

Important termination reasons:

- `confirmed_stable`: selected low-bound rate passed final confirmation.
- `scheduler_config_limited`: high-bound evidence points to server scheduler limits; external orchestration should adjust vLLM config and rerun.
- `max_request_rate_limited`: configured `--max-request-rate` prevented finding an unstable high bound.
- `no_confirmed_stable_open_loop_rate`: confirmation rejected the lowest known open-loop stable rate.
- `closed_loop_only`: closed-loop mode was used without open-loop MST search.

## Stability Semantics

Decision mapping:

- `stable`: updates low search bound.
- `unstable`, `slo_violation`, `aborted_safety`: update high search bound.
- `uncertain`: rerun same rate once with extended duration.
- repeated `uncertain`: conservative high bound if a stable low bound exists; otherwise convergence error.

SLO violations are top-priority evidence. A stationary server can still fail the configured TTFT/TPOT SLO.

Outstanding-request backlog pressure uses a robust trend gate. All conditions must hold:

```text
SciPy Theil-Sen slope > max_positive_backlog_slope
Mann-Kendall p < backlog_trend_alpha
fitted relative increase >= min_backlog_relative_increase
fitted absolute delta >= min_backlog_growth_for_hard_pressure
```

The old `outstanding requests grew across consecutive windows` rule is intentionally removed. It is not robust under stochastic open-loop arrivals.

`completion_arrival_ratio < 1 - completion_arrival_tolerance` is supporting evidence only, not an independent instability trigger.

## Caveats

- MST finder assumes the server is already live and compatible with the requested endpoint/model.
- It does not start, stop, restart, health-check, or mutate vLLM.
- It does not search model/server configuration space.
- Missing optional Prometheus metrics may lower confidence and reduce bottleneck specificity.
- Comparisons are meaningful only across compatible workload, model, SLO, and server-config metadata.
- Open-loop rate must not be silently throttled. If the client cannot issue configured load, the trial should be `client_limited` or `aborted_safety`, not stable.
- Result quality depends on representative workload sampling and sufficient trial duration.
