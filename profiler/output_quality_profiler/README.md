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
  --job-id smoke \
  --output-dir results/quality/live-smoke \
  --workload experiments/quality/smoke_mock/workload_yamls/shard_000.yaml \
  --model mock-quality-model \
  --base-url http://127.0.0.1:9900 \
  --endpoint /v1/chat/completions \
  --request-timeout-s 120 \
  --max-concurrency 1 \
  --response-text-max-chars 8192 \
  --force
```

Slurm smoke submission uses the mock OpenAI-compatible server:

```bash
PYTHONPATH=profiler:. python -m slurm_orchestrator.cli quality-submit \
  --manifest experiments/quality/slurm_quality_smoke_mock.yaml \
  --run-id quality-smoke-mock-000

PYTHONPATH=profiler:. python -m slurm_orchestrator.cli quality-collect \
  --run-root results/quality/quality-smoke-mock-000 \
  --no-sync-results
```

The smoke path is for checking Slurm task wiring only. Real quality runs should
replace the raw mock launch template with normal vLLM launch settings and use
materialized ShareGPT/WildChat shards.
