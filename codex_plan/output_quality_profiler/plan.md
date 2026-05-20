# Output Quality Profiler

Status: Draft updated on 2026-05-20 for chat-quality judging and benchmark
compatibility evaluation.

## Goal

Add a response-quality profiling layer with two closely related modes:

1. Chat quality profiling for datasets that do not have ground truth labels or
   canonical reference response text, starting with ShareGPT and WildChat.
2. Benchmark compatibility profiling for datasets that do have ground truth and
   original evaluation metrics, starting with SuperGPQA, RepoBench/CrossCodeEval,
   and the LongBench v1 tasks already covered by existing materialization.

For chat-quality datasets, the profiler should generate one response per model
for a fixed, stratified request set, then prepare LLM-as-a-judge comparison
batches against a selected reference model and compute the win/tie adjusted
score:

```text
Q_chat(theta) = Pr[y_theta > y_base] + 0.5 Pr[y_theta ~ y_base]
```

For ground-truth benchmarks, the profiler should reuse the same response
collection infrastructure, then run lightweight compatibility adapters around
the benchmark's original scoring semantics. This layer is not a latency,
throughput, stability, or energy profiler. It may reuse serving lifecycle,
manifest expansion, workload loading, request construction, and Slurm/local
scheduling code, but its authoritative artifact is model response text plus the
evaluation records derived from that text.

## Non-Goals

- No MST search, SLO classification, window aggregation, or throughput report.
- No GPU power collection or energy accounting.
- No automatic judge API execution in the first implementation unless the API
  provider contract is finalized.
- No heavy benchmark harness dependency such as LLMCompass in the response
  collection path.
- No full reimplementation of unsupported benchmark suites. Compatibility
  adapters should target only the benchmarks and tasks explicitly in scope.
- No claim that a covered-task subset is the full benchmark when the materialized
  task set is partial, especially for LongBench v1.
- No model-server configuration sweep beyond the same model/workload/launch
  matrix style already supported by `local_orchestrator`.

## Existing Reuse Points

- `dataset_workload_materializer` already defines the offline dataset boundary:
  raw artifacts in, JSONL shards plus workload YAMLs out.
- `llm_mst_finder.workload` already loads `sharegpt`, streamed HF datasets such
  as `allenai/WildChat`, and materialized `jsonl` workloads into
  `SampleRequest` objects with prompt length metadata.
- `llm_mst_finder.request_client.RequestClient` already builds
  OpenAI-compatible request payloads and can capture response text when
  `LLM_MST_FINDER_CAPTURE_RESPONSE_TEXT=1`; the quality profiler should reuse
  the payload/client behavior but write a response-only record shape.
- `local_orchestrator.manifest` and `local_orchestrator.matrix` already parse
  strict experiment manifests, apply launch/search overrides, probe model memory
  needs, and produce deterministic `ExpandedExperimentJob` values.
- `local_orchestrator.lifecycle`, `resources`, and `scheduler` already own local
  GPU/port leasing and vLLM startup.
- `slurm_orchestrator` already reuses expanded local jobs to materialize Slurm
  array payloads and per-job state files.

## Main Use Case

Protocol V1:

1. Sample 10,000 total prompts from ShareGPT and WildChat with a 50/50 source
   split.
2. Stratify by `(source, prompt_length_bucket)`, where buckets are
   `short`, `medium`, and `long`.
3. Divide the stratified population into 10 deterministic shards.
4. Generate one response per model per request under fixed decoding settings.
5. Select one reference/base model.
6. For each candidate model, encode response pairs from one or more shards as
   long JSON batch prompts for an LLM-as-a-judge provider.
7. Randomize A/B order per comparison item while preserving enough metadata to
   invert judge labels back to candidate/base outcomes.
8. Parse judge results and compute win, tie, loss, invalid, and
   `Q_chat(theta)`.

The fixed decoding settings are part of the V1 contract. The judge prompt
template remains late-bound until the provider contract is finalized. The
package contract should carry decoding settings and judge-template paths as
manifest fields and copied artifacts so later changes are auditable.

V1 decoding contract:

- `temperature: 0.6`
- `top_p: 0.95`
- `top_k: 20`
- `min_p: 0.0`
- `n: 1`
- `max_tokens: 32768`, adjusted downward at runtime when the model context
  cannot support `prompt_tokens + 32768 + prompt_token_buffer`
- `prompt_token_buffer: 128`

## Recommended Package Layout

Add a new package instead of extending MST or energy modules directly:

```text
profiler/output_quality_profiler/
  __init__.py
  cli.py
  models.py
  manifest.py
  materialization.py
  generation.py
  benchmark_selection.py
  benchmark_adapters/
    __init__.py
    base.py
    supergpqa.py
    code_completion.py
    longbench_v1.py
  benchmark_reporting.py
  judge_batches.py
  scoring.py
  reporting.py
  orchestrator_adapter.py
```

Rationale:

- Quality profiling consumes similar manifests and serving infrastructure, but
  its experiment semantics and artifact schema are different from MST.
- Response generation, judge-batch creation, judge-result parsing, and scoring
  should be independently rerunnable.
- Ground-truth benchmark scoring should be independently rerunnable from saved
  response JSONL without restarting model serving.
- Local and Slurm execution should share the same generation job model.

## Workflow

### Phase A: Materialize Quality Requests

Command:

```bash
PYTHONPATH=profiler:. python -m output_quality_profiler.cli materialize \
  --config experiments/quality/sharegpt_wildchat_10k.yaml
```

Inputs:

- ShareGPT source path, normally through the existing `data/local/sharegpt`
  symlink convention.
- WildChat HF dataset id, split, conversation field, optional scan cap.
- Tokenizer used only for prompt-length accounting and bucket assignment.
- Source weights: ShareGPT 5,000 and WildChat 5,000 for V1.
- Prompt-length bucket policy.
- Shard count: 10.
- Seed.

Output:

```text
experiments/quality/sharegpt_wildchat_10k/
  materialization_config.yaml
  materialization_report.json
  request_manifest.json
  shards/
    shard_000.runner.jsonl
    ...
    shard_009.runner.jsonl
  workload_yamls/
    shard_000.yaml
    ...
    shard_009.yaml
```

Each JSONL row should include:

```json
{
  "prompt": "...",
  "prompt_len": 512,
  "expected_output_len": 1024,
  "metadata": {
    "request_id": "sharegpt:abc123",
    "source": "sharegpt",
    "prompt_length_bucket": "medium",
    "stratum": "sharegpt:medium",
    "source_row_index": 123,
    "session_id": "...",
    "turn_index": 1,
    "content_hash": "...",
    "shard_id": "shard_000",
    "within_shard_index": 42
  }
}
```

Keep quality-specific identifiers inside `metadata` because the existing JSONL
workload loader preserves `metadata` and ignores unrelated top-level row fields.
The quality response writer should promote common metadata fields such as
`request_id`, `source`, and `prompt_length_bucket` into its own output rows.

Generated workload YAMLs should use existing `llm_mst_finder` JSONL semantics:

```yaml
name: quality-sharegpt-wildchat-shard-000
dataset:
  type: jsonl
  path: ../shards/shard_000.runner.jsonl
sampling:
  seed: 42
  num_requests: 1000
  entry_selection: sequential
  prompt_len:
    mode: from_dataset
  output_len:
    mode: natural_until_eos
    max_tokens: 32768
request:
  stream: true
  temperature: 0.6
  top_p: 0.95
  ignore_eos: false
  extra_body:
    top_k: 20
    min_p: 0.0
    n: 1
context_policy:
  max_model_len: 32768
  tokenizer_source: vllm_model_config
  over_limit: skip_sample
```

The first implementation can reuse `llm_mst_finder.workload` loaders directly
inside `output_quality_profiler.materialization` instead of forcing the existing
`dataset_workload_materializer` to understand mixed ShareGPT/HF stratification.
If this logic later grows, move shared dataset adapters into a common helper.

### Phase B: Generate Model Responses

Quality manifests should look similar to local orchestrator manifests, but
replace `search` with `generation`:

```yaml
run:
  run_id: h100-quality-sharegpt-wildchat-000
  output_root: ../results/quality
  default_endpoint: /v1/chat/completions
  python_executable: /scratch/gautschi/shi676/BioLLM/.venv-h100/bin/python

slurm:
  partition: ai
  account: yiding
  qos: normal
  time: 04:00:00
  modules:
    - modtree/gpu
    - cuda/12.9.0
  setup_commands:
    - source /scratch/gautschi/shi676/BioLLM/helper_scripts/activate_vllm_h100.sh
  python_executable: /scratch/gautschi/shi676/BioLLM/.venv-h100/bin/python
  array_concurrency_limit: 4
  base_port: 9600

hardware:
  name: h100-80gb
  gpu_memory_gb: 80
  gpu_memory_utilization: 0.90

probe:
  enabled: true
  auto_gpu_count: false
  default_context_tokens: 4096
  activation_memory_gb: 2.0
  memory_safety_factor: 1.20
  kv_cache_request_count: 1

launch:
  executable: vllm
  tensor_parallel_size: 1
  gpu_count: 1
  dtype: float16
  gpu_memory_utilization: 0.90
  max_model_len: 32768
  readiness_timeout_s: 1200

generation:
  request_timeout_s: 21600
  concurrency_source: mst_fraction
  concurrency_mst_fraction: 0.40
  preserve_request_order: true
  response_text_max_chars: 65536
  include_prompt_text: true
  decoding:
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    n: 1
    max_tokens: 32768
    max_tokens_policy: model_context_minus_prompt_buffer
    prompt_token_buffer: 128
    extra_body: {}

experiments:
  - id: qwen3-8b-quality
    model: Qwen/Qwen3-8B
    workloads:
      - ./quality/sharegpt_wildchat_10k/workload_yamls/shard_000.yaml
      - ./quality/sharegpt_wildchat_10k/workload_yamls/shard_001.yaml
```

Implementation contract:

- Reuse local orchestrator parsing primitives where possible. Prefer extracting
  shared manifest sections (`run`, `slurm`, `hardware`, `probe`, `launch`,
  `overrides`, `experiments`) into reusable helpers rather than duplicating
  validation.
- Add `QualityGenerationConfig`, `QualityExperimentJob`, and
  `QualityRunManifest` models. They can embed or reference existing
  `LaunchConfig`, `HardwareConfig`, `ProbeConfig`, and `SlurmConfig`.
- Expand each model experiment into one concrete generation job that may own
  multiple shard workloads so the same serving instance can be reused across
  shards. Per-shard outputs must remain separately addressable under the model
  result directory.
- Reuse `VLLMLifecycleManager`, `GPULeaseManager`, `PortAllocator`, and
  `runtime_server_signature`.
- Do not invoke `llm_mst_finder.cli search` or `report`.
- Implement a `QualityGenerationAdapter` that loads the workload samples,
  dispatches each sample exactly once, captures full response text, and writes
  response artifacts.

Per-job output:

```text
results/quality/<run_id>/responses/<model_slug>/<shard_id>/
  responses.jsonl
  summary.json
  failed_requests.jsonl
```

`responses.jsonl` row contract:

```json
{
  "run_id": "h100-quality-sharegpt-wildchat-000",
  "job_id": "qwen3-8b-quality__shard_000",
  "model": "Qwen/Qwen3-8B",
  "workload": "experiments/quality/.../shard_000.yaml",
  "request_id": "sharegpt:abc123",
  "source": "sharegpt",
  "prompt_length_bucket": "medium",
  "prompt": "...",
  "response_text": "...",
  "finish_reason": null,
  "success": true,
  "error": null,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024
  },
  "metadata": {
    "shard_id": "shard_000",
    "within_shard_index": 42,
    "content_hash": "..."
  }
}
```

`summary.json` should include counts by source, bucket, success/error class,
truncation count, decoding settings digest, workload digest, model, launch
fields, and timestamps. It should not report TTFT, TPOT, throughput, or energy.

Client concurrency should be moderate and derived from existing MST results:
`quality_concurrency = floor(0.40 * mst_request_rate)` after converting the
MST rate to a practical concurrent request count for the selected model/shard
runner. A manifest may use an explicit concurrency only when MST evidence is not
available, and that override should be visible in run metadata.

### Phase C: Local and Slurm Integration

Local command set:

```bash
python -m output_quality_profiler.cli dry-run --manifest experiments/quality/run.yaml
python -m output_quality_profiler.cli run --manifest experiments/quality/run.yaml --run-id ...
python -m output_quality_profiler.cli resume --run-root results/quality/<run_id>
python -m output_quality_profiler.cli status --run-root results/quality/<run_id>
```

Slurm command set:

```bash
python -m slurm_orchestrator.cli quality-plan --manifest experiments/quality/run.yaml --run-id ...
python -m slurm_orchestrator.cli quality-submit --manifest experiments/quality/run.yaml --run-id ...
python -m slurm_orchestrator.cli quality-resume --run-root results/quality/<run_id>
python -m slurm_orchestrator.cli quality-collect --run-root results/quality/<run_id>
```

Recommended Slurm implementation:

- Mirror the existing energy Slurm path rather than overloading MST array tasks.
- Group quality generation jobs by `launch.gpu_count`.
- Each task starts one vLLM server, waits for readiness, invokes
  `python -m output_quality_profiler.cli run-live-generation`, finalizes one
  per-job state file, and exits.
- `quality-collect` aggregates per-job state into `state.json`,
  `summary.json`, and `summary.md` under the quality run root.
- Publish only compact quality summaries, judge batches, score outputs, and
  optionally response JSONL files when explicitly requested. Response JSONL may
  be large and should not silently sync to shared results.

### Phase D: Build Judge Batches

Command:

```bash
python -m output_quality_profiler.cli build-judge-batches \
  --run-root results/quality/<run_id> \
  --reference-model Qwen/Qwen3-8B \
  --candidate-model Qwen/Qwen3-14B \
  --include-shard shard_000 \
  --include-shard shard_001 \
  --judge-template experiments/quality/judge_templates/pairwise_v1.txt \
  --output-dir results/quality/<run_id>/judge_batches/qwen3-14b_vs_qwen3-8b
```

Batch artifact contract:

```text
judge_batches/<candidate>_vs_<base>/
  batch_manifest.json
  prompts/
    batch_000.jsonl
    batch_001.jsonl
  answer_key.jsonl
```

Each comparison item must include:

- request id, source, shard id, and prompt-length bucket;
- prompt text;
- base response text;
- candidate response text;
- randomized A/B assignment;
- deterministic random seed and A/B inversion metadata;
- model identifiers hidden from the judge prompt unless the template
  explicitly requests them.

`answer_key.jsonl` is the authoritative map from provider result id to
`candidate_is_a`. This makes randomized A/B order auditable and reversible.

### Phase E: Score Judge Results

Command:

```bash
python -m output_quality_profiler.cli score \
  --batch-manifest results/quality/<run_id>/judge_batches/.../batch_manifest.json \
  --judge-results results/quality/<run_id>/judge_results/...jsonl \
  --output-dir results/quality/<run_id>/scores/qwen3-14b_vs_qwen3-8b
```

Input parser V1 should accept a normalized JSONL format:

```json
{
  "comparison_id": "shard_000:000042",
  "judge_label": "A_BETTER"
}
```

Supported normalized labels:

- `A_BETTER`
- `B_BETTER`
- `TIE`
- `INVALID`

Provider-specific adapters can be added after the selected API is known.

Score output:

```text
scores/<candidate>_vs_<base>/
  score.json
  score_by_stratum.csv
  score.md
  invalid_items.jsonl
```

`score.json` should include total valid comparisons, win/tie/loss/invalid
counts, `Q_chat`, Wilson or bootstrap confidence intervals, and stratified
breakdowns by source and prompt-length bucket. Overall score should be computed
on the intended stratified sample. If a stratum has missing/invalid judge
outputs, report both raw and reweighted scores and mark the coverage gap.

### Phase F: Ground-Truth Benchmark Compatibility

This phase expands V1 without changing the response collection boundary. It is
for benchmarks where `experiments/quality/benchmark-scores-1.xlsx` shows that a
model lacks a usable public score or consistent substitute score, and where the
benchmark has ground truth plus an original metric we can reproduce or call
lightly.

Target benchmarks:

- `SuperGPQA`: problem-solving benchmark with answer labels/text in materialized
  metadata. Collect responses on the existing SuperGPQA workloads, then score
  with a SuperGPQA adapter that extracts the final answer and compares it to the
  materialized ground truth. Report overall accuracy and breakdowns by available
  subject, difficulty, and shard metadata.
- `RepoBench/CrossCodeEval`: code-completion benchmarks with target code in
  materialized metadata. Collect raw completions on the existing RepoBench and
  CrossCodeEval workloads, then score with the original metric semantics where
  they can be called as a lightweight evaluator. If an official evaluator is too
  heavy, implement a documented compatibility metric and mark it as such in the
  report.
- `LongBench v1`: long-context benchmark tasks, limited to profiles already
  covered by materialization. Collect responses on the covered tasks only, run a
  LongBench-v1 adapter using the original per-task metrics, and report the score
  as a covered-task subset rather than full LongBench v1.

Current LongBench covered materialization profiles:

- `long_output_summarization`: `gov_report`, `gov_report_e`
- `medium_output_summarization`: `multi_news`, `multi_news_e`, `qmsum`,
  `vcsum`
- `medium_answer_rag_qa`: `dureader`
- `short_answer_document_qa`: `multifieldqa_en`, `multifieldqa_en_e`,
  `multifieldqa_zh`, `qasper`, `qasper_e`

Two-step protocol:

1. Response collection: parse the score workbook, select missing
   `(model, benchmark)` pairs, map each benchmark to an existing materialized
   workload set, and submit Slurm quality generation jobs. Do not choose a
   reference model and do not build judge prompts for these runs.
2. Evaluation: read saved response JSONL plus preserved workload metadata, adapt
   it to the benchmark evaluator input format, run the evaluator, and emit score
   artifacts. Evaluation must be rerunnable without model serving.

Workbook selection contract:

- Input workbook: `experiments/quality/benchmark-scores-1.xlsx`.
- Treat a score as missing when the relevant benchmark name is absent from the
  workbook row, the corresponding score cell is empty, or the workbook marks the
  benchmark as `N/A`.
- Treat substitute scores as usable only when the benchmark name is explicitly
  accepted by a benchmark-specific normalization rule. For this expansion,
  SuperGPQA, RepoBench/CrossCodeEval, and covered LongBench v1 should be
  selected when the row only contains unrelated substitutes such as MMLU-Pro,
  EvalPlus, LiveCodeBench, MRCR, RULER, or LongBench v2.
- Emit a machine-readable missing-score plan so response collection is auditable
  and can be reviewed before submission.

Expected selection output:

```text
results/quality/<run_id>/benchmark_selection/
  missing_scores.json
  missing_scores.csv
  workbook_parse_report.json
```

`missing_scores.json` row contract:

```json
{
  "model": "Qwen/Qwen3-30B-A3B-Instruct",
  "benchmark": "RepoBench",
  "scorebook_category": "Code Completion Benchmark",
  "scorebook_benchmark_cell": null,
  "scorebook_score_cell": null,
  "reason": "target benchmark missing",
  "workload_group": "repobench_python_java_aggregate_cache_realistic",
  "workload_paths": [
    "experiments/code_workloads/repobench_python_java_aggregate_cache_realistic/workload_yamls/shard_000.yaml"
  ]
}
```

Benchmark response artifact contract:

```text
results/quality/<run_id>/benchmark_responses/<benchmark>/<model_slug>/
  aggregate_responses.jsonl
  shards/
    shard_000.responses.jsonl
    ...
  generation_summary.json
```

Each response row should retain the same response fields as chat-quality
generation and must preserve benchmark metadata from the materialized workload:

```json
{
  "model": "Qwen/Qwen3-8B",
  "benchmark": "LongBench-v1-covered",
  "dataset_task": "qasper",
  "request_id": "longbench:qasper:123",
  "prompt": "...",
  "response_text": "...",
  "success": true,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024
  },
  "metadata": {
    "ground_truth": ["..."],
    "longbench_task": "qasper",
    "longbench_profile": "short_answer_document_qa",
    "target_hash": "..."
  }
}
```

Benchmark evaluation artifact contract:

```text
results/quality/<run_id>/benchmark_scores/<benchmark>/<model_slug>/
  score.json
  score.md
  per_item.jsonl
  evaluator_inputs/
  evaluator_outputs/
```

`score.json` should include:

- model, benchmark, workload group, shard list, and response run id;
- evaluator adapter name and version;
- decoding settings digest and workload digest;
- total items, scored items, failed generations, invalid parses, and skipped
  items;
- overall score and benchmark-specific sub-scores;
- explicit `is_full_benchmark: false` when only a subset is covered.

Benchmark decoding policy:

- Do not blindly reuse the chat-quality decoding contract for benchmark scoring.
- Default to deterministic benchmark decoding (`temperature=0.0`, `top_p=1.0`,
  `n=1`) unless an original benchmark specifies otherwise.
- Record benchmark-specific `max_tokens`, stop sequences, and postprocessing
  rules in the manifest. Code-completion benchmarks especially need explicit
  stop and stripping behavior so the response text is a completion, not a chatty
  explanation.

Recommended CLI additions:

```bash
python -m output_quality_profiler.cli select-missing-benchmark-scores \
  --scorebook experiments/quality/benchmark-scores-1.xlsx \
  --output-dir results/quality/<run_id>/benchmark_selection

python -m output_quality_profiler.cli build-benchmark-generation-manifest \
  --missing-plan results/quality/<run_id>/benchmark_selection/missing_scores.json \
  --output experiments/quality/<run_id>_benchmark_responses.yaml

python -m output_quality_profiler.cli score-benchmark-responses \
  --benchmark longbench-v1-covered \
  --responses-root results/quality/<run_id>/benchmark_responses/longbench-v1-covered \
  --output-dir results/quality/<run_id>/benchmark_scores/longbench-v1-covered
```

## Stratification Contract

The V1 materializer should make the sample shape explicit:

```yaml
sampling:
  seed: 20260520
  total_requests: 10000
  shards: 10
  sources:
    - name: sharegpt
      weight: 0.5
      dataset:
        type: sharegpt
        path: ../../data/local/sharegpt/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
    - name: wildchat
      weight: 0.5
      dataset:
        type: hf
        path: allenai/WildChat
        split: train
        conversation_field: conversation
        max_scan_rows: 50000
  prompt_length_buckets:
    short:
      lt_tokens: 100
    medium:
      min_tokens: 100
      max_tokens: 512
    long:
      gt_tokens: 512
  allocation:
    source: exact
    bucket: proportional_with_minimum
    minimum_per_source_bucket: 100
```

Implementation details:

- Load each source with existing conversation semantics: first valid user prompt
  and assistant response for single-turn mode unless the config says otherwise.
- Tokenize prompts once with the configured tokenizer and record token counts.
- Assign buckets before sampling.
- Use the V1 bucket boundaries exactly: `short <100`, `medium 100-512`, and
  `long >512` prompt tokens.
- Sample deterministically within each source/bucket stratum.
- Preserve source proportions exactly at 5,000/5,000 for V1.
- Make bucket allocation deterministic and report actual counts. If a stratum
  lacks enough rows, fail unless `allow_replacement: true` is explicitly set.
- Distribute each stratum across all 10 shards to avoid shard-local source or
  length skew.
- After the 10k shards are materialized, generation must not resample. Feed all
  requests sequentially, shard by shard, for every selected model.

## Boundaries and Gaps

### Dataset materialization

Gap: existing `dataset_workload_materializer` is built around one dataset config
at a time and its current dataset list does not include mixed ShareGPT +
WildChat stratification.

Plan: implement quality-specific mixed materialization using
`llm_mst_finder.workload` dataset adapters first. Later, factor common
conversation extraction into shared helpers if materializer duplication becomes
real.

### Response capture

Gap: `RequestClient` can capture response text only as metadata on
`RequestRecord`, and it truncates by environment variable.

Plan: add a response-only client helper or adapter method that returns
`response_text`, finish/error metadata, and request metadata without requiring
latency artifact writes. Internally it can reuse `build_openai_payload`,
stream decoding helpers, endpoint detection, and HTTP session handling.

### Orchestrator model

Gap: `local_orchestrator` dataclasses require `SearchConfig` and produce
`ExpandedExperimentJob` values tied to MST result directories.

Plan: share the common manifest parsing and expansion machinery where it is
clean, but introduce `QualityExperimentJob` and a quality scheduler adapter.
Avoid pretending a quality generation run is an MST search with dummy search
fields.

### Slurm adapter

Gap: current Slurm tasks call MST search/report or energy trial commands.

Plan: add explicit quality subcommands and task shell rendering. Reuse grouping,
state collection, launch command rendering, readiness checks, and result sync
controls.

### Judge API provider

Gap: provider, exact template, rate limits, and response schema are not yet
known.

Plan: make batch construction provider-neutral. Store normalized comparison
items and answer keys now; add `submit-judge-batches` only after the provider
contract is fixed.

### Benchmark scorebook selection

Gap: `benchmark-scores-1.xlsx` is human-oriented. Benchmark names and score
cells may contain multiple entries, substitutes, empty values, or `N/A`, and the
meaning of "missing" depends on the target benchmark.

Plan: implement a strict parser that normalizes model names, benchmark aliases,
and score cells into a typed missing-score plan. Keep the raw workbook values in
the plan for auditability. Selection should be deterministic and reviewed before
Slurm submission.

### Benchmark workload registry

Gap: benchmark workloads already exist under several experiment directories, but
there is no single registry mapping a benchmark selection to the workload YAMLs
that should be served.

Plan: add a lightweight registry in the quality profiler that points at existing
materialized workload groups first:

- `SuperGPQA`: `experiments/reasoning_workloads/supergpqa_reasoning` and, when
  explicitly requested, `supergpqa_hard_reasoning`.
- `RepoBench`: `experiments/code_workloads/repobench_python_java_aggregate_cache_realistic`
  or the 8k-drop variant when the benchmark plan asks for that constraint.
- `CrossCodeEval`: `experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic`.
- `LongBench-v1-covered`: the four covered LongBench materialization profiles
  already present under `experiments/longbench_workloads/materialization`.

### Benchmark evaluator adapters

Gap: the materialized workloads preserve ground truth metadata, but there is no
common adapter layer from quality response JSONL to original benchmark evaluator
inputs and outputs.

Plan: add per-benchmark adapters with a small shared interface:

```python
class BenchmarkEvaluator:
    benchmark_name: str

    def build_evaluator_inputs(self, responses_path, output_dir): ...
    def run_evaluator(self, evaluator_inputs, output_dir): ...
    def normalize_scores(self, evaluator_outputs): ...
```

Adapters should prefer original evaluator code when it can be called without a
heavy framework install. If a local compatibility metric is used instead, the
score report must state the metric name and compatibility limitation.

### Benchmark response generation

Gap: chat-quality response generation assumes comparison-oriented chat prompts,
while benchmark scoring may require deterministic decoding, benchmark-specific
max output lengths, stop sequences, and stricter postprocessing.

Plan: reuse serving and request dispatch, but make generation config
benchmark-aware. The response contract remains response JSONL; only decoding
defaults, output directories, and evaluator metadata differ.

## Implementation Order

1. Add `output_quality_profiler` package skeleton, models, README, and CLI help.
2. Implement mixed ShareGPT/WildChat materialization with deterministic
   source/bucket/shard reports.
3. Add unit tests for bucket assignment, exact 50/50 allocation, shard balance,
   and insufficient-stratum failures.
4. Implement quality manifest parsing and matrix expansion using existing
   launch/hardware/probe/slurm dataclasses where practical.
5. Implement local response generation against an already-running server
   (`run-live-generation`) before managed local scheduling.
6. Implement managed local `run/resume/status` by reusing lifecycle, resource,
   and state-store patterns.
7. Add Slurm `quality-plan`, `quality-submit`, `quality-resume`, and
   `quality-collect`.
8. Implement judge batch construction with randomized A/B assignment and
   answer-key artifacts.
9. Implement normalized judge-result scoring and reports.
10. Add end-to-end smoke tests with a tiny synthetic/mock OpenAI-compatible
    server and 2 shards.
11. Implement workbook parsing and missing-score plan generation for
    SuperGPQA, RepoBench/CrossCodeEval, and covered LongBench v1.
12. Add a benchmark workload registry that maps each selected benchmark to
    existing materialized workload YAMLs and records whether the selection is a
    full benchmark or a covered subset.
13. Add benchmark response manifest generation that groups shards per model so
    each model reuses its serving instance across the selected workload set.
14. Implement SuperGPQA, RepoBench/CrossCodeEval, and LongBench-v1-covered
    evaluator adapters with tiny fixture tests first.
15. Add benchmark score reporting and manifest-level provenance checks so scores
    can be traced back to workbook row, workload digest, decoding config, and
    response artifact.

## Validation Plan

- Unit tests:
  - materialization allocation and bucket assignment;
  - deterministic shard layout for fixed seed;
  - manifest validation rejects unknown keys and missing decoding fields;
  - judge A/B randomization is deterministic and reversible;
  - score formula handles wins, ties, losses, invalids, and stratum weights.
  - workbook parser normalizes aliases, empty cells, `N/A`, and substitute
    benchmark names;
  - missing-score selection emits stable `(model, benchmark)` plans;
  - benchmark adapters score tiny golden examples and preserve invalid-item
    accounting.

- Integration tests with mocked server:
  - `run-live-generation` emits response JSONL with no latency/energy summary
    fields;
  - managed local run writes state, summary, logs, and per-job response files;
  - resume skips succeeded jobs and reruns failed/incomplete jobs.

- Slurm dry-run tests:
  - quality jobs group by GPU count;
  - generated sbatch scripts call quality task entry points, not MST or energy
    commands;
  - collect creates `results/quality/<run_id>/summary.json` and `summary.md`.

- Manual smoke:
  - materialize 100 ShareGPT/WildChat requests into 2 shards;
  - generate responses for two small local models;
  - build one judge batch with a placeholder template;
  - score a hand-written normalized judge result file.
  - build a tiny benchmark response fixture from existing materialized
    SuperGPQA, RepoBench/CrossCodeEval, and LongBench rows;
  - run each benchmark adapter on the fixture and inspect `score.json`,
    `per_item.jsonl`, and invalid parse accounting.

## Enforced Decisions

- Fixed decoding uses `temperature=0.6`, `top_p=0.95`, `top_k=20`,
  `min_p=0.0`, `n=1`, and model-context-aware `max_tokens=32768`.
- Prompt buckets are `short <100`, `medium 100-512`, and `long >512`.
- After materialization, no runtime request resampling is allowed.
- Full prompt text is included in response JSONL by default.
- Client concurrency is derived from 40% of existing MST results unless an
  explicit manifest override is required and recorded.
- Ground-truth benchmark evaluation is a two-step workflow: collect responses
  first, then score from saved response artifacts.
- Benchmark response collection must not bind a reference model, build judge
  prompts, or call a judge API.
- Benchmark scores must say whether they are full-benchmark scores or
  covered-task subset scores.

## Still Open

- Judge provider schema, upload mechanism, rate limit handling, retry policy,
  and exact judge prompt template.
- Whether overall `Q_chat` should be simple pooled valid comparisons or
  source/bucket reweighted when invalid judge results are nonuniform.
- Exact benchmark alias policy for `benchmark-scores-1.xlsx`, including which
  public substitute scores are acceptable enough to skip local evaluation.
- Whether SuperGPQA should be scored through an official evaluator script, a
  label-extraction compatibility adapter, or both.
- Exact RepoBench and CrossCodeEval metric implementations, including whether
  to vendor minimal official evaluator code or shell out to a checked-out
  benchmark repo.
- LongBench v1 evaluator packaging and task averaging for the covered subset.
- Benchmark-specific decoding defaults, max token caps, stop sequences, and
  response postprocessing rules.
