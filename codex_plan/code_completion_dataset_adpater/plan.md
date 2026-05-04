# Code Completion Workload Materialization Plan

## 0. Decision

Do not build code-completion adapters inside `llm_mst_finder`.

The existing runtime split stays:

```text
llm_mst_finder      -> trials, request execution, stability, MST search, reports
local_orchestrator -> server lifecycle, model/workload matrix, GPU/port leasing
energy_profiler    -> fixed-rate energy trials from MST/search outputs
```

Add one small independent materialization layer whose only job is:

```text
raw or preprocessed code dataset
  -> materialized single-turn prompts
  -> filtered/deduplicated prompt pool
  -> ordered JSONL shards
  -> ordinary llm_mst_finder workload YAMLs
```

No runtime retrieval, repo checkout, file I/O during profiling, tool calls,
editing, tests, or agentic SWE workflows belong in this layer.

## 1. Package Boundary

Create a small package outside `llm_mst_finder`:

```text
profiler/
  code_workload_materializer/
    __init__.py
    cli.py
    materialize.py
```

Split out `datasets.py` once CrossCodeEval and RepoBench both exist. Do not
create one file per dataset for the MVP.

Do not add dataset-specific behavior to:

```text
llm_mst_finder/request_client.py
llm_mst_finder/trial_runner.py
llm_mst_finder/search.py
local_orchestrator/scheduler.py
energy_profiler/executor.py
gpu_monitor.py
```

## 2. Contract With `llm_mst_finder`

The materializer emits JSONL rows already compatible with the existing
`dataset.type: jsonl` workload path:

```json
{
  "prompt": "Complete the code at the cursor. Return only the completion.\n\n...",
  "expected_output_len": 32,
  "metadata": {
    "dataset": "crosscodeeval",
    "task": "cross_file_materialized",
    "language": "python",
    "repo_id": "owner/repo",
    "file_path": "src/foo.py",
    "session_id": "crosscodeeval::python::owner/repo::src",
    "sample_id": "crosscodeeval-python-000001",
    "content_hash": "...",
    "target_hash": "...",
    "prompt_token_count": 1024,
    "target_token_count": 17,
    "shard_id": "shard_000"
  }
}
```

It also emits normal workload YAML files, one per shard:

```yaml
name: crosscodeeval-cross-file-shard-000
dataset:
  type: jsonl
  path: ./shards/shard_000.runner.jsonl
sampling:
  seed: 42
  num_requests: 8000
  entry_selection: sequential
  prompt_len:
    mode: from_dataset
  output_len:
    mode: from_dataset
request:
  stream: true
  temperature: 0.0
  ignore_eos: false
context_policy:
  max_model_len: 32768
  tokenizer_source: vllm_model_config
  over_limit: truncate_prompt
  truncation_side: left
```

Required `llm_mst_finder` enhancement:

```text
sampling.entry_selection: random_with_replacement | sequential
```

Default remains `random_with_replacement`. Materialized code shards use
`sequential` so cache-realistic ordering survives after offline sharding.

Do not add a new `code_completion_jsonl` workload type.

## 3. Output Directory Contract

Each preparation run writes:

```text
experiments/code_workloads/<workload_name>/
  materialization_config.yaml
  materialization_report.json
  shards_manifest.json
  shards/
    shard_000.runner.jsonl
    shard_001.runner.jsonl
  workload_yamls/
    shard_000.yaml
    shard_001.yaml
```

`local_orchestrator` consumes the generated YAML paths exactly like any other
workload. It should not prepare datasets or understand code-dataset internals.

## 4. Internal Data Shape

Avoid a large canonical schema. Use one small dict/dataclass internally:

```text
sample_id
prompt
target
expected_output_len
metadata
```

Recommended metadata:

```text
dataset
task
split
language
repo_id
file_path
session_id
sequence_index
content_hash
target_hash
prompt_token_count
target_token_count
```

The exported metadata contract matters more than the internal Python shape.

## 5. Dataset Scope

### Priority 1: CrossCodeEval-Like Materialization

Primary source:

```text
GitHub: https://github.com/amazon-science/cceval
Project: https://crosscodeeval.github.io/
```

Implement one offline materializer for CrossCodeEval or
CrossCodeEval-like preprocessed JSONL.

Why first:

```text
directly targets cross-file code completion
multilingual: Python, Java, TypeScript, C#
already organized around materialized retrieval/cross-file context settings
closest match to single-turn IDE-style serving workload
```

MVP input forms:

```text
local JSONL file
local directory of JSONL files
```

Supported field aliases:

```yaml
field_aliases:
  prompt: [prompt, input, code_context, current_file_prefix]
  target: [target, completion, reference, groundtruth]
  language: [language, lang]
  repo_id: [repo, repo_name, repository]
  file_path: [file_path, path]
  cross_file_context: [cross_file_context, retrieved_context, context]
  sequence_index: [sequence_index, cursor_index, order]
```

Default prompt:

```text
Complete the code at the cursor. Return only the completion.

<REPOSITORY_CONTEXT>
{cross_file_context}
</REPOSITORY_CONTEXT>

<CURRENT_FILE_PREFIX>
{current_file_prefix}
</CURRENT_FILE_PREFIX>
```

If no cross-file context exists, omit the repository-context block.

Do not add chain-of-thought, explanations, chat framing, quality judging, or
task-solving evaluation.

### Priority 2: RepoBench As A Complete Code Workload

Primary source:

```text
GitHub: https://github.com/Leolty/repobench
HF Python: tianyang/repobench_python_v1.1
HF Java: tianyang/repobench_java_v1.1
```

Implement RepoBench after CrossCodeEval, but value the whole benchmark rather
than only the LongBench subset.

Why second:

```text
dedicated repository-level code auto-completion benchmark
explicit in_file, cross_file_first, and cross_file_random settings
useful context-budget levels: 2k, 4k, 8k, 12k, 16k
better source for controlled code-context scaling than LongBench-code
```

Required RepoBench materialization modes:

```text
repobench/in_file
repobench/cross_file_first
repobench/cross_file_random
```

Default profiling mode:

```text
repobench/cross_file_first
```

Treat context budgets as materialization-time variants. Emit separate workload
directories or shard groups for each selected budget, rather than adding
runtime logic to `llm_mst_finder`.

### Priority 3: Existing LongBench-Code Baseline

Use the existing LongBench-code workload as a convenient baseline, not as the
main code-completion source:

```text
experiments/workloads/longbench_code_truncated.yaml
```

It covers:

```text
lcc
repobench-p
lcc_e
repobench-p_e
```

Important: LongBench-code includes `repobench-p`, but LongBench is a mixed
long-context benchmark collection rather than a complete code-completion
benchmark. Keep it for comparison and smoke coverage; do not let it replace the
dedicated RepoBench materializer.

### Dropped From Current Implementation Priority

Remove these from the active plan:

```text
M2RC-EVAL
R2C2-Bench
HumanEval
MBPP
EvalPlus
```

Reasons:

```text
M2RC-EVAL artifact availability is unclear.
R2C2 is only useful if earlier sources lack volume.
HumanEval/MBPP/EvalPlus are toy code-generation smoke tests, not serving workloads.
```

## 6. Offline Filtering And Reporting

Filtering happens before shard export and is summarized in
`materialization_report.json`.

Required drops:

```text
missing/empty prompt
missing/empty target
prompt too short
prompt too long
target too long
unsupported language
duplicate content_hash
```

Reasonable defaults:

```yaml
filtering:
  min_prompt_tokens: 128
  max_prompt_tokens: 8192
  min_target_tokens: 1
  max_target_tokens: 128
  languages:
    include: []
    exclude: []
  dedup:
    content_hash: true
```

Token counting should use the requested model/vLLM tokenizer when available.
If a fallback tokenizer is used, record it in the report. Do not silently mix
tokenizers in one materialization run.

## 7. Sharding And Ordering

Default ordering should preserve prefix-cache realism:

```text
1. build session_id as dataset::language::repo_id::directory
2. sort within each session by sequence_index, then file_path, then content_hash
3. shuffle sessions with fixed seed
4. emit short bursts from each session
5. avoid duplicate content_hash inside a shard
6. avoid duplicate content_hash across adjacent shards when practical
```

This materializes the order offline. Runtime should simply replay the shard
with `sampling.entry_selection: sequential`.

`shards_manifest.json` should report at least:

```text
workload_name
dataset/task
num_shards
samples_per_shard
per-shard path and workload YAML path
language counts
prompt token p50/p90/p95
target token p50/p90/p95
unique content-hash counts
```

## 8. CLI

Keep one command:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m code_workload_materializer.cli prepare \
  --config experiments/code_workloads/crosscodeeval_cache_realistic.yaml
```

No `llm_mst_finder.cli code-workload ...` nesting for MVP.

## 9. Example Materialization Config

```yaml
name: crosscodeeval_cross_file_cache_realistic

dataset:
  name: crosscodeeval
  raw_path: data/raw/crosscodeeval
  split: test
  mode: cross_file_materialized
  field_aliases:
    prompt: [prompt, input, current_file_prefix]
    target: [target, completion, reference, groundtruth]
    language: [language, lang]
    repo_id: [repo, repo_name, repository]
    file_path: [file_path, path]
    cross_file_context: [cross_file_context, retrieved_context, context]

tokenization:
  tokenizer: Qwen/Qwen3-8B

filtering:
  min_prompt_tokens: 128
  max_prompt_tokens: 8192
  min_target_tokens: 1
  max_target_tokens: 128

sampling:
  seed: 42
  policy: cache_realistic
  burst_size: 8

request:
  endpoint: /v1/completions
  max_tokens: from_target
  temperature: 0.0
  stream: true
  stop:
    - "\n\n"

sharding:
  output_dir: experiments/code_workloads/crosscodeeval_cross_file_cache_realistic
  samples_per_shard: 8000
  num_shards: 16

workload_yaml:
  context_policy:
    max_model_len: 32768
    tokenizer_source: vllm_model_config
    over_limit: truncate_prompt
    truncation_side: left
```

## 10. MVP Implementation Order

1. Implement `code_workload_materializer` with local JSONL/directory input,
   filtering, reports, sharding, and generated workload YAMLs.
2. Add `sampling.entry_selection` to `llm_mst_finder.workload`.
3. Materialize and run one CrossCodeEval-like shard through a short live MST
   smoke test.
4. Add RepoBench materialization as the second real adapter, covering
   `in_file`, `cross_file_first`, and `cross_file_random`, with selected
   context-budget variants emitted offline.
5. Compare CrossCodeEval and RepoBench behavior against existing LongBench-code:
   `experiments/workloads/longbench_code_truncated.yaml`.

## 11. Acceptance Criteria

Materialization:

```text
all exported rows have prompt, expected_output_len, and metadata.dataset/task
all exported rows have metadata.content_hash
materialization_report.json exists
shards_manifest.json exists
tokenizer identity and filtering counts are recorded
```

Runtime:

```text
llm_mst_finder consumes generated workload YAML without a new workload type
JSONL rows can be replayed sequentially
request metadata preserves dataset/task/language/repo/file/session fields
no dataset-specific code is added to request_client or trial_runner
local_orchestrator consumes generated YAML paths without schema changes
```

## 12. Non-Goals

Do not implement:

```text
agentic coding loops
SWE-bench style repo checkout/edit/test workflows
online retrieval during profiling
repo file I/O during profiling
GPU monitoring changes
energy profiler rewrites
orchestrator shard-selection logic
new MST search logic
```

Those are separate orchestrator or agentic-workload topics, not code-completion
materialization.
