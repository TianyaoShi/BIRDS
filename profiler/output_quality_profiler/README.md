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

## Ground-Truth Benchmark Compatibility

Phase F adds a two-step path for benchmarks with ground truth:

1. select missing `(model, benchmark)` pairs from
   `experiments/quality/benchmark-scores-1.xlsx`;
2. collect responses with the same Slurm quality generation path, then score
   saved response JSONL with benchmark adapters.

Supported targets are `SuperGPQA`, `SuperGPQA-hard`, `RepoBench`,
`CrossCodeEval`, and `LongBench-v1-covered`. `SuperGPQA` uses the full 26,528
record materialization for models missing public SuperGPQA scores.
`SuperGPQA-hard` uses the 7,049 hard-question subset for every model so it
matches the MST/energy profiling boundary. `RepoBench`, `CrossCodeEval`, and
the detailed LongBench-v1-covered target are selected for every eligible model
because those arranged score columns do not provide consistent substitutes for
the original benchmark/evaluator contract. Code targets exclude `gpt-oss`
models, and LongBench excludes Llama-2 models. LongBench is reported as a
covered-task subset, not full LongBench v1.

Do not use the latency/energy LongBench materializations under
`experiments/longbench_workloads/materialization/*_qwen3_8b`: those shards use
repeat expansion for profiling. Do not use the `*_expanded_qwen3_8b` variants
either because they include external expansion rows without benchmark reference
answers. The benchmark registry only accepts no-repeat original-task
materializations under `experiments/longbench_workloads/benchmark_original/`
and checks each materialization report before building response manifests.

```bash
PYTHONPATH=profiler:. python -m output_quality_profiler.cli select-missing-benchmark-scores \
  --scorebook experiments/quality/benchmark-scores-1.xlsx \
  --output-dir results/quality/<run_id>/benchmark_selection

PYTHONPATH=profiler:. python -m output_quality_profiler.cli build-benchmark-generation-manifest \
  --missing-plan results/quality/<run_id>/benchmark_selection/missing_scores.json \
  --base-manifest experiments/quality/h100_sharegpt_wildchat_10k_responses.yaml \
  --output experiments/quality/<run_id>_benchmark_responses.yaml \
  --include-benchmark CrossCodeEval

PYTHONPATH=profiler:. python -m output_quality_profiler.cli score-benchmark-responses \
  --benchmark CrossCodeEval \
  --responses-root results/quality/<run_id>/responses/<model-or-job-dir> \
  --output-dir results/quality/<run_id>/benchmark_scores/crosscodeeval/<model-or-job-dir>
```

The current adapters are intentionally lightweight compatibility adapters for
SuperGPQA and RepoBench/CrossCodeEval: SuperGPQA uses answer-label extraction,
and RepoBench/CrossCodeEval report code-completion exact-match/similarity
metrics. LongBench-v1-covered uses the original LongBench v1 task-to-metric
mapping over the covered tasks and resolves answer lists from
`data/raw/longbench` by `longbench_dataset` and `longbench_row_index`. The
LongBench adapter prefers the original `rouge` and `jieba` packages when they
are installed, and records dependency status in `score.json`.
