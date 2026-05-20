# Output Quality Profiler

`output_quality_profiler` is the response-text quality layer for ShareGPT and
WildChat style datasets that do not have ground-truth labels or canonical
reference responses.

The V1 contract is intentionally fixed:

- sample 10,000 requests, split exactly 5,000 ShareGPT and 5,000 WildChat;
- shard into 10 sequential workload shards after stratified sampling;
- bucket prompts as `short <100`, `medium 100-512`, and `long >512` tokens;
- include full prompt text in response artifacts;
- generate one response per model with `temperature=0.6`, `top_p=0.95`,
  `top_k=20`, `min_p=0.0`, `n=1`, and a 32,768 token cap adjusted downward by
  model context minus prompt length plus buffer;
- use moderate client concurrency derived from 40% of existing MST results
  unless an explicit concurrency is supplied.

This package must not run MST search, latency reporting, or energy monitoring.
Its authoritative generation artifact is `responses.jsonl`.

Current commands:

```bash
PYTHONPATH=profiler:. python -m output_quality_profiler.cli validate-materialization \
  --config experiments/quality/sharegpt_wildchat_10k.yaml

PYTHONPATH=profiler:. python -m output_quality_profiler.cli dry-run \
  --manifest experiments/quality/run.yaml

PYTHONPATH=profiler:. python -m output_quality_profiler.cli run-live-generation \
  --job-id shard_000 \
  --output-dir results/quality/live-shard-000 \
  --workload experiments/quality/sharegpt_wildchat_10k/workload_yamls/shard_000.yaml \
  --model <model-name> \
  --base-url http://127.0.0.1:<port> \
  --endpoint /v1/chat/completions \
  --request-timeout-s 120 \
  --max-concurrency <reviewed-concurrency> \
  --response-text-max-chars 8192 \
  --force
```

Judge batch construction writes OpenAI Batch JSONL plus a manifest that records
the randomized A/B assignment:

```bash
PYTHONPATH=profiler:. python -m output_quality_profiler.cli build-judge-batch \
  --responses-root results/quality/<run_id>/responses \
  --reference-model-slug meta-llama-llama-3-1-8b-instruct \
  --candidate-model-slug qwen-qwen3-8b \
  --judge-template codex_plan/output_quality_profiler/llm_as_a_judge_template.md \
  --output-dir results/quality/<run_id>/judge_batches/<batch_name> \
  --evaluator-model gpt-4.1-nano \
  --max-comparisons 100

PYTHONPATH=profiler:. python -m output_quality_profiler.cli aggregate-judge-results \
  --batch-manifest results/quality/<run_id>/judge_batches/<batch_name>/batch_manifest.json \
  --judge-results results/quality/<run_id>/judge_batches/<batch_name>/<batch_output>.jsonl \
  --output-dir results/quality/<run_id>/scores/<batch_name>
```

A/B order is randomized during judge JSONL composition and inverted during
aggregation. Aggregation also reports scores by candidate position. If those
position breakdowns or a larger audit confirm judge position bias, run
position-swap judging for the affected comparison set and aggregate the paired
judgments before reporting final scores.
