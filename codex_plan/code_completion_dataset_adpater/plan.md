# Revised Coding Plan: IDE-Level Code Completion Workloads for Existing Profiler Repo

## 0. Scope Revision

Do **not** create a new profiling framework.

Existing repo already has:

```text
llm_mst_finder      -> fixed-rate / closed-loop trials and MST search
local_orchestrator -> model/workload matrix execution
energy_profiler    -> fixed-rate energy trials using MST outputs
gpu_monitor.py     -> GPU energy accounting
```

The missing layer is:

```text
IDE-level code completion dataset adapters
    -> prompt materialization
    -> workload filtering
    -> cache-realistic sampling
    -> shard generation
    -> export into existing llm_mst_finder workload format
```

The new implementation should focus on the following path:

```text
raw benchmark dataset
  -> normalized code-completion samples
  -> filtered workload pool
  -> cache-realistic ordered shards
  -> existing llm_mst_finder / local_orchestrator / energy_profiler execution
```

Runtime profiling should still be:

```text
JSONL workload shard
  -> existing request client
  -> OpenAI-compatible server
  -> existing metrics/energy pipeline
```

---

# 1. Recommended Integration Location

Add a new dataset/workload subpackage under `llm_mst_finder`, because MST search and fixed-rate trial execution already live there.

Recommended tree:

```text
profiler/
  llm_mst_finder/
    code_completion/
      __init__.py
      schema.py
      adapters.py
      crosscodeeval.py
      repobench.py
      m2rc_eval.py
      r2c2.py
      prompting.py
      tokenization.py
      filtering.py
      sampling.py
      sharding.py
      export.py
      inspect.py
```

Then wire this into:

```text
profiler/llm_mst_finder/workload.py
profiler/llm_mst_finder/cli.py
profiler/local_orchestrator/manifest.py
profiler/energy_profiler/planning.py
```

Do **not** modify:

```text
gpu_monitor.py
energy_profiler/executor.py
llm_mst_finder/trial_runner.py
llm_mst_finder/request_client.py
```

unless absolutely necessary. The dataset layer should emit workload files that look like existing workloads.

---

# 2. Dataset Priority

Implement adapters in this order.

## Priority 0: CrossCodeEval

Primary workload. GitHub Repo: https://github.com/amazon-science/cceval

Use for:

```text
realistic IDE-level cross-file completion
short output
moderate prompt length
multi-language workload
cache-realistic request ordering
```

Modes to support:

```text
crosscodeeval/in_file
crosscodeeval/cross_file_materialized
```

Default:

```text
crosscodeeval/cross_file_materialized
```

Rationale: this is the closest fit for single-turn OpenAI-compatible code-completion profiling without online tools or agentic interaction.

---

## Priority 1: RepoBench

Use for context-length scaling. GitHub Repo: https://github.com/Leolty/repobench

Modes:

```text
repobench/in_file
repobench/cross_file_first
repobench/cross_file_random
```

Default for energy scaling:

```text
repobench/cross_file_first
```

Support prompt budgets:

```text
2k
4k
8k
12k
16k
```

RepoBench is less important than CrossCodeEval for realistic workload diversity, but more useful for controlled prefill energy scaling.

---

## Priority 2: M2RC-EVAL ❌

Use for multilingual and semantic-bucket coverage. Artifact unavailable. Do not implement.

Modes:

```text
m2rc_eval/default
m2rc_eval/language_balanced
m2rc_eval/semantic_bucket_balanced
```

Default:

```text
m2rc_eval/language_balanced
```

This should be implemented after CrossCodeEval and RepoBench.

---

## Priority 3: R2C2-Bench or Similar Repo-Level Dataset

Optional high-volume stress workload.

Use only when the earlier datasets do not provide enough unique prompts for high request-rate experiments.

---

## Priority 4: HumanEval / MBPP / EvalPlus

Smoke-test only.

Do not use for main energy conclusions.

---

# 3. New Canonical Code Completion Sample Schema

Internally normalize every dataset into one schema.

Create:

```text
profiler/llm_mst_finder/code_completion/schema.py
```

Recommended dataclasses:

```python
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class CodeCompletionSample:
    sample_id: str
    dataset: str
    task: str
    split: str

    language: str | None
    repo_id: str | None
    file_path: str | None

    prompt: str
    target: str | None

    prompt_token_count: int | None = None
    target_token_count: int | None = None
    max_tokens: int = 32

    temperature: float = 0.0
    stop: list[str] = field(default_factory=list)

    session_id: str | None = None
    sequence_index: int | None = None

    content_hash: str | None = None
    target_hash: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
```

The schema should be separate from the existing request schema. The export step will convert this into the format expected by `llm_mst_finder`.

Required fields for every normalized sample:

```text
sample_id
dataset
task
split
prompt
max_tokens
temperature
content_hash
metadata
```

Recommended fields:

```text
language
repo_id
file_path
target
prompt_token_count
target_token_count
session_id
sequence_index
stop
```

---

# 4. Adapter Interface

Create:

```text
profiler/llm_mst_finder/code_completion/adapters.py
```

Base interface:

```python
from abc import ABC, abstractmethod
from collections.abc import Iterable


class CodeCompletionAdapter(ABC):
    name: str

    @abstractmethod
    def iter_raw(self, config: dict) -> Iterable[dict]:
        """Yield raw dataset examples."""

    @abstractmethod
    def normalize(self, raw: dict, config: dict) -> CodeCompletionSample:
        """Convert raw example into canonical CodeCompletionSample."""

    def iter_normalized(self, config: dict) -> Iterable[CodeCompletionSample]:
        for raw in self.iter_raw(config):
            yield self.normalize(raw, config)
```

Dataset adapters should **not** know anything about GPU monitoring, MST search, local orchestration, or energy trials.

---

# 5. Adapter Details

## 5.1 CrossCodeEval Adapter

File:

```text
profiler/llm_mst_finder/code_completion/crosscodeeval.py
```

Supported input forms:

```text
local JSONL
local directory
Hugging Face dataset cache, optional
pre-materialized prompt JSONL
```

Config example:

```yaml
dataset:
  name: crosscodeeval
  raw_path: data/raw/crosscodeeval
  split: test
  mode: cross_file_materialized
  languages: [python, java, typescript, csharp]
```

Normalization behavior:

```text
mode = in_file:
    prompt = current-file prefix only

mode = cross_file_materialized:
    prompt = retrieved/materialized cross-file context + current-file prefix
```

The adapter should support field aliases because CrossCodeEval-derived files may have slightly different column names depending on preprocessing.

Example alias config:

```yaml
field_aliases:
  prompt: [prompt, input, code_context]
  target: [groundtruth, reference, completion, target]
  language: [language, lang]
  repo_id: [repo, repo_name, repository]
  file_path: [file_path, path]
  cross_file_context: [cross_file_context, retrieved_context, context]
```

Prompt construction should be delegated to `prompting.py`, not hardcoded in the adapter.

---

## 5.2 RepoBench Adapter

File:

```text
profiler/llm_mst_finder/code_completion/repobench.py
```

Config example:

```yaml
dataset:
  name: repobench
  raw_path: data/raw/repobench
  split: test
  mode: cross_file_first
  languages: [python, java]
  context_budget_tokens: 4096
```

Supported modes:

```text
in_file
cross_file_first
cross_file_random
```

The adapter should preserve RepoBench context-length metadata when available.

Metadata to keep:

```text
original_context_length
repo_name
language
masking_policy
file_path
```

RepoBench should be used to generate context-length-specific processed files:

```text
repobench_2k.jsonl
repobench_4k.jsonl
repobench_8k.jsonl
repobench_16k.jsonl
```

---

## 5.3 M2RC-EVAL Adapter

File:

```text
profiler/llm_mst_finder/code_completion/m2rc_eval.py
```

Config example:

```yaml
dataset:
  name: m2rc_eval
  raw_path: data/raw/m2rc_eval
  split: test
  mode: language_balanced
  languages: all
  context_budget_tokens: 4096
```

Metadata to preserve:

```text
language
semantic_bucket
bucket_level
ast_depth
repo_id
file_path
```

M2RC-EVAL is especially useful for filtering and stratification, so the adapter should not discard bucket metadata.

---

## 5.4 R2C2 Adapter

File:

```text
profiler/llm_mst_finder/code_completion/r2c2.py
```

Implement only after the first three adapters are stable.

Use it as a high-volume source with the same normalized schema.

---

# 6. Prompting Layer

Create:

```text
profiler/llm_mst_finder/code_completion/prompting.py
```

The adapter should produce logical fields; the prompting layer should produce the final prompt string.

## 6.1 Default Prompt Template

Use a minimal IDE-style prompt:

```text
Complete the code at the cursor. Return only the completion.

{context}
```

Do not use:

```text
You are a helpful assistant.
Let's think step by step.
Explain the code.
```

Those are not IDE completion prompts and will distort output behavior.

## 6.2 Cross-File Context Template

For materialized cross-file context:

```text
# Relevant repository context:
{cross_file_context}

# Current file:
{current_file_prefix}
```

For languages where `#` is inappropriate, use a neutral separator:

```text
<REPOSITORY_CONTEXT>
{cross_file_context}
</REPOSITORY_CONTEXT>

<CURRENT_FILE_PREFIX>
{current_file_prefix}
</CURRENT_FILE_PREFIX>
```

Default should be language-neutral markers, because they are easier to implement consistently across datasets.

Recommended default:

```text
Complete the code at the cursor. Return only the completion.

<REPOSITORY_CONTEXT>
{cross_file_context}
</REPOSITORY_CONTEXT>

<CURRENT_FILE_PREFIX>
{current_file_prefix}
</CURRENT_FILE_PREFIX>
```

For `in_file` mode:

```text
Complete the code at the cursor. Return only the completion.

<CURRENT_FILE_PREFIX>
{current_file_prefix}
</CURRENT_FILE_PREFIX>
```

## 6.3 Optional FIM Mode

Support later, not MVP.

Config:

```yaml
prompt:
  style: plain_prefix   # plain_prefix | fim
```

MVP should implement only:

```text
plain_prefix
```

---

# 7. Tokenization and Context Budgeting

Create:

```text
profiler/llm_mst_finder/code_completion/tokenization.py
```

Main responsibilities:

```text
load tokenizer
count prompt tokens
count target tokens
truncate to context budget
store token counts in metadata
```

Config:

```yaml
tokenization:
  tokenizer_path: /path/to/tokenizer/or/model
  fallback: cl100k_base
  count_targets: true
```

If the model tokenizer cannot be loaded, use fallback tokenization but record:

```json
{
  "tokenizer_status": "fallback",
  "tokenizer_name": "cl100k_base"
}
```

Do not silently mix tokenizers across runs.

## 7.1 Truncation Policy

Create function:

```python
def truncate_prompt_to_budget(
    instruction: str,
    cross_file_context: str | None,
    current_file_prefix: str,
    max_context_tokens: int,
    tokenizer: TokenCounter,
) -> str:
    ...
```

Priority preservation:

```text
1. instruction
2. current file prefix near cursor
3. highest-priority cross-file context
4. older or lower-priority context
```

For code completion, the immediate prefix near the cursor is more important than old repository context.

---

# 8. Filtering Layer

Create:

```text
profiler/llm_mst_finder/code_completion/filtering.py
```

Filtering is essential. The goal is to remove pathological samples before MST or energy profiling.

## 8.1 Required Filters

Implement these filters:

```text
missing_prompt
empty_prompt
empty_target optional
prompt_too_short
prompt_too_long
target_too_long
unsupported_language
duplicate_prompt_hash
bad_unicode
license_exclusion optional
repo_exclusion optional
```

Config:

```yaml
filtering:
  min_prompt_tokens: 64
  max_prompt_tokens: 4096
  min_target_tokens: 1
  max_target_tokens: 128

  languages:
    include: [python, java, typescript, csharp]
    exclude: []

  dedup:
    exact_prompt_hash: true
    normalized_prompt_hash: false

  exclude_repos: []
  include_repos: []
```

## 8.2 Recommended Defaults by Dataset

### CrossCodeEval

```yaml
filtering:
  min_prompt_tokens: 128
  max_prompt_tokens: 4096
  max_target_tokens: 64
```

### RepoBench

```yaml
filtering:
  min_prompt_tokens: 512
  max_prompt_tokens: 16384
  max_target_tokens: 128
```

### M2RC-EVAL

```yaml
filtering:
  min_prompt_tokens: 512
  max_prompt_tokens: 8192
  max_target_tokens: 128
```

## 8.3 Filtering Report

Every preparation run should output:

```text
filtering_report.json
```

Example:

```json
{
  "dataset": "crosscodeeval",
  "input_samples": 10000,
  "kept_samples": 8432,
  "dropped": {
    "missing_prompt": 0,
    "prompt_too_short": 312,
    "prompt_too_long": 841,
    "target_too_long": 104,
    "unsupported_language": 0,
    "duplicate_prompt_hash": 311
  },
  "language_counts": {
    "python": 2200,
    "java": 2100,
    "typescript": 2070,
    "csharp": 2062
  },
  "prompt_tokens": {
    "mean": 1342.5,
    "p50": 1011,
    "p95": 3860
  },
  "target_tokens": {
    "mean": 18.7,
    "p50": 14,
    "p95": 48
  }
}
```

This report is important because the final energy results are only meaningful if the workload distribution is known.

---

# 9. Sampling and Cache-Realistic Ordering

Create:

```text
profiler/llm_mst_finder/code_completion/sampling.py
```

Default policy:

```text
cache_realistic
```

Do not use random nonce comments. Do not deliberately destroy prefix reuse.

## 9.1 Session ID Construction

Build `session_id` as:

```text
dataset::language::repo_id::directory
```

Fallbacks:

```text
dataset::language::repo_id
dataset::language
dataset
```

Implementation:

```python
def build_session_id(sample: CodeCompletionSample) -> str:
    ...
```

## 9.2 Sequence Index

If cursor position or original order exists, use it.

Otherwise stable order by:

```text
repo_id
file_path
content_hash
```

## 9.3 Cache-Realistic Sampling Policy

Goal: simulate a developer making many completions inside a repo or nearby files.

Algorithm:

```text
1. group samples by session_id
2. sort samples inside each session by sequence_index or stable path/hash order
3. shuffle session order with fixed seed
4. emit short bursts from each session
5. interleave sessions to avoid one huge repo dominating a trial
6. forbid exact duplicate prompt hashes inside one shard
```

Config:

```yaml
sampling:
  policy: cache_realistic
  seed: 42

  session:
    min_session_length: 4
    max_session_length: 32
    burst_size: 8
    interleave_sessions: true

  dedup:
    forbid_duplicate_prompt_within_shard: true
    forbid_duplicate_prompt_across_adjacent_shards: true
```

## 9.4 Cache-Cold Sampling Policy

Secondary ablation only.

```yaml
sampling:
  policy: cache_cold
  seed: 42
  shuffle_all_samples: true
  preserve_session_order: false
```

Use this only to separate prefix-cache effects from baseline inference cost.

---

# 10. Sharding Layer

Create:

```text
profiler/llm_mst_finder/code_completion/sharding.py
```

The sharding output is what the existing trial runner should consume.

## 10.1 Shard Types

Generate:

```text
warmup.jsonl
rate_000.jsonl
rate_001.jsonl
rate_002.jsonl
...
replicate_000.jsonl
replicate_001.jsonl
...
```

Or simpler:

```text
shard_000.jsonl
shard_001.jsonl
shard_002.jsonl
...
```

Keep a metadata file:

```text
shards_manifest.json
```

Example:

```json
{
  "dataset": "crosscodeeval",
  "task": "cross_file_materialized",
  "sampling_policy": "cache_realistic",
  "num_shards": 16,
  "samples_per_shard": 8000,
  "shards": [
    {
      "name": "shard_000.jsonl",
      "num_samples": 8000,
      "unique_prompt_hashes": 8000,
      "languages": {
        "python": 2022,
        "java": 1990,
        "typescript": 1988,
        "csharp": 2000
      },
      "prompt_tokens_mean": 1203.4,
      "prompt_tokens_p95": 3811
    }
  ]
}
```

## 10.2 Non-Overlap Policy

Default:

```text
no duplicate prompt hash inside a shard
no duplicate prompt hash across adjacent shards
```

Strict mode:

```text
no duplicate prompt hash across all shards
```

Config:

```yaml
sharding:
  samples_per_shard: 8000
  num_shards: 16
  no_overlap_within_shard: true
  no_overlap_across_adjacent_shards: true
  no_overlap_across_all_shards: false
```

## 10.3 Why Adjacent-Shard Non-Overlap Matters

Your MST sweeps likely run rates sequentially:

```text
rate 1 -> rate 2 -> rate 4 -> rate 8 -> ...
```

If the same prompts appear in adjacent trials and the server is not restarted, prefix cache behavior can become artificially favorable. Adjacent-shard non-overlap is a practical compromise: it preserves production-like cache behavior without exact repeat artifacts.

---

# 11. Export Into Existing Workload Format

Create:

```text
profiler/llm_mst_finder/code_completion/export.py
```

The exporter should support two formats.

## 11.1 Canonical Format

Full metadata for inspection and future analysis:

```json
{
  "sample_id": "crosscodeeval-python-000001",
  "dataset": "crosscodeeval",
  "task": "cross_file_materialized",
  "language": "python",
  "repo_id": "owner/repo",
  "file_path": "src/foo.py",
  "session_id": "crosscodeeval::python::owner/repo::src",
  "prompt": "...",
  "target": "...",
  "prompt_token_count": 1024,
  "target_token_count": 17,
  "max_tokens": 32,
  "temperature": 0.0,
  "stop": ["\n\n"],
  "content_hash": "...",
  "metadata": {}
}
```

## 11.2 Runner Format

Minimal format consumed by `llm_mst_finder.workload` and `request_client`.

Use whichever shape your existing `workload.py` expects. If it currently supports text prompts, export:

```json
{
  "id": "crosscodeeval-python-000001",
  "prompt": "...",
  "max_tokens": 32,
  "temperature": 0.0,
  "stop": ["\n\n"],
  "metadata": {
    "dataset": "crosscodeeval",
    "task": "cross_file_materialized",
    "language": "python",
    "repo_id": "owner/repo",
    "file_path": "src/foo.py",
    "session_id": "crosscodeeval::python::owner/repo::src",
    "prompt_token_count": 1024,
    "target_token_count": 17,
    "content_hash": "..."
  }
}
```

If the existing runner expects OpenAI request bodies directly, export:

```json
{
  "id": "crosscodeeval-python-000001",
  "request": {
    "prompt": "...",
    "max_tokens": 32,
    "temperature": 0.0,
    "stop": ["\n\n"],
    "stream": false
  },
  "metadata": {
    "dataset": "crosscodeeval",
    "task": "cross_file_materialized",
    "language": "python",
    "repo_id": "owner/repo",
    "file_path": "src/foo.py",
    "session_id": "crosscodeeval::python::owner/repo::src",
    "prompt_token_count": 1024,
    "target_token_count": 17,
    "content_hash": "..."
  }
}
```

The key rule: **preserve metadata all the way into trial outputs** so energy results can be stratified later by dataset, language, prompt length, repo, and mode.

---

# 12. Changes to Existing Modules

## 12.1 `llm_mst_finder/workload.py`

Add a workload source type:

```yaml
workload:
  type: code_completion_jsonl
  path: experiments/energy/workloads/crosscodeeval/shards/shard_000.runner.jsonl
```

The workload loader should yield request records with:

```text
id
prompt or request body
max_tokens
temperature
stop
metadata
```

Do not add dataset-specific logic here.

`workload.py` should only know:

```text
This is a JSONL workload with prompt-bearing records.
```

---

## 12.2 `llm_mst_finder/cli.py`

Add subcommands under existing CLI.

Recommended:

```bash
python -m profiler.llm_mst_finder.cli code-workload prepare ...
python -m profiler.llm_mst_finder.cli code-workload inspect ...
python -m profiler.llm_mst_finder.cli code-workload shard ...
python -m profiler.llm_mst_finder.cli code-workload export ...
```

Or a single prepare command:

```bash
python -m profiler.llm_mst_finder.cli prepare-code-completion-workload \
  --config experiments/energy/code_completion/crosscodeeval.yaml
```

Keep the user-facing path simple.

---

## 12.3 `local_orchestrator/manifest.py`

Extend manifest schema to support code-completion workload generation or reference to prebuilt shards.

Recommended: do **not** make the orchestrator prepare datasets by default. Make it consume prepared shards.

Example:

```yaml
workloads:
  - name: crosscodeeval_cross_file_cache_realistic
    type: code_completion_jsonl
    shard_dir: experiments/energy/workloads/crosscodeeval/shards
    shard_selection: sequential
    cache_policy: cache_realistic
```

The orchestrator should pick shards for trials but should not understand CrossCodeEval internals.

---

## 12.4 `energy_profiler/planning.py`

Energy profiler should consume MST outputs as before, but trial metadata should include workload metadata:

```json
{
  "workload_name": "crosscodeeval_cross_file_cache_realistic",
  "dataset": "crosscodeeval",
  "task": "cross_file_materialized",
  "cache_policy": "cache_realistic",
  "shard": "shard_003.runner.jsonl"
}
```

No major energy-profiler change should be necessary unless the current plan schema lacks workload metadata.

---

## 12.5 `mst_analyzer`

Later, add rules to compare:

```text
in_file vs cross_file_materialized
CrossCodeEval vs RepoBench
2k vs 4k vs 8k vs 16k
language-specific energy/latency
cache_realistic vs cache_cold
```

This is analysis-only. Not part of MVP.

---

# 13. Configuration Files

Add experiment configs under:

```text
experiments/
  energy/
    code_completion/
      crosscodeeval_cache_realistic.yaml
      crosscodeeval_in_file.yaml
      repobench_context_scaling.yaml
      m2rc_eval_language_balanced.yaml
```

## 13.1 CrossCodeEval Default Config

```yaml
name: crosscodeeval_cross_file_cache_realistic

dataset:
  name: crosscodeeval
  raw_path: data/raw/crosscodeeval
  split: test
  mode: cross_file_materialized
  languages: [python, java, typescript, csharp]

prompt:
  style: plain_prefix
  include_instruction: true
  template: ide_completion_v1

tokenization:
  tokenizer_path: null
  fallback: cl100k_base
  count_targets: true

filtering:
  min_prompt_tokens: 128
  max_prompt_tokens: 4096
  min_target_tokens: 1
  max_target_tokens: 64
  dedup:
    exact_prompt_hash: true
    normalized_prompt_hash: false

sampling:
  policy: cache_realistic
  seed: 42
  session:
    min_session_length: 4
    max_session_length: 32
    burst_size: 8
    interleave_sessions: true
  dedup:
    forbid_duplicate_prompt_within_shard: true
    forbid_duplicate_prompt_across_adjacent_shards: true

request:
  endpoint: completions
  max_tokens: 32
  temperature: 0.0
  stream: false
  stop:
    - "\n\n"

sharding:
  output_dir: experiments/energy/workloads/crosscodeeval_cross_file_cache_realistic
  samples_per_shard: 8000
  num_shards: 16
  runner_format: llm_mst_finder_jsonl
  no_overlap_within_shard: true
  no_overlap_across_adjacent_shards: true
  no_overlap_across_all_shards: false
```

---

## 13.2 RepoBench Context Scaling Config

```yaml
name: repobench_context_scaling_cache_realistic

dataset:
  name: repobench
  raw_path: data/raw/repobench
  split: test
  mode: cross_file_first
  languages: [python, java]
  context_budgets: [2048, 4096, 8192, 16384]

prompt:
  style: plain_prefix
  include_instruction: true
  template: ide_completion_v1

filtering:
  min_prompt_tokens: 512
  max_prompt_tokens: 16384
  min_target_tokens: 1
  max_target_tokens: 128

sampling:
  policy: cache_realistic
  seed: 42

request:
  endpoint: completions
  max_tokens: 64
  temperature: 0.0
  stream: false

sharding:
  output_dir: experiments/energy/workloads/repobench_context_scaling
  samples_per_shard: 4000
  num_shards: 16
```

For this config, the prepare command should generate separate workload directories:

```text
repobench_2k/
repobench_4k/
repobench_8k/
repobench_16k/
```

---

## 13.3 M2RC-EVAL Config

```yaml
name: m2rc_eval_language_balanced_cache_realistic

dataset:
  name: m2rc_eval
  raw_path: data/raw/m2rc_eval
  split: test
  mode: language_balanced
  languages: all
  context_budget_tokens: 4096

prompt:
  style: plain_prefix
  include_instruction: true
  template: ide_completion_v1

filtering:
  min_prompt_tokens: 512
  max_prompt_tokens: 8192
  min_target_tokens: 1
  max_target_tokens: 128

sampling:
  policy: cache_realistic
  seed: 42
  balance:
    by_language: true
    by_semantic_bucket: false

request:
  endpoint: completions
  max_tokens: 64
  temperature: 0.0
  stream: false

sharding:
  output_dir: experiments/energy/workloads/m2rc_eval_language_balanced
  samples_per_shard: 8000
  num_shards: 16
```

---

# 14. Preparation Command Flow

## 14.1 Prepare CrossCodeEval

```bash
python -m profiler.llm_mst_finder.cli prepare-code-completion-workload \
  --config experiments/energy/code_completion/crosscodeeval_cache_realistic.yaml
```

Expected outputs:

```text
experiments/energy/workloads/crosscodeeval_cross_file_cache_realistic/
  processed.canonical.jsonl
  processed.runner.jsonl
  filtering_report.json
  sampling_report.json
  shards_manifest.json
  shards/
    shard_000.runner.jsonl
    shard_001.runner.jsonl
    ...
```

## 14.2 Inspect Prepared Workload

```bash
python -m profiler.llm_mst_finder.cli inspect-code-completion-workload \
  --workload-dir experiments/energy/workloads/crosscodeeval_cross_file_cache_realistic
```

Should print:

```text
dataset
task
num samples
num shards
languages
prompt token distribution
target token distribution
duplicate prompt count
session count
mean session length
p95 session length
```

---

# 15. Existing MST / Energy Execution Integration

After workload generation, existing MST commands should work with minimal changes.

Example manifest entry:

```yaml
workloads:
  - name: crosscodeeval_cross_file_cache_realistic
    type: code_completion_jsonl
    shard_dir: experiments/energy/workloads/crosscodeeval_cross_file_cache_realistic/shards
    cache_policy: cache_realistic
```

Example trial-level workload reference:

```yaml
workload:
  type: code_completion_jsonl
  path: experiments/energy/workloads/crosscodeeval_cross_file_cache_realistic/shards/shard_000.runner.jsonl
```

The existing request client should receive a normal OpenAI-compatible completion request.

---

# 16. Default Request Body

For `/v1/completions`, each exported request should map to:

```json
{
  "model": "MODEL_NAME_FILLED_BY_RUNNER",
  "prompt": "<prompt>",
  "max_tokens": 32,
  "temperature": 0.0,
  "stream": false,
  "stop": ["\n\n"]
}
```

The workload file should not hardcode model name unless your current infrastructure already does that.

For chat-only servers, allow exporter option:

```yaml
request:
  endpoint: chat_completions
```

Mapped request:

```json
{
  "model": "MODEL_NAME_FILLED_BY_RUNNER",
  "messages": [
    {
      "role": "user",
      "content": "<prompt>"
    }
  ],
  "max_tokens": 32,
  "temperature": 0.0,
  "stream": false
}
```

Use `/v1/completions` by default.

---

# 17. Workload Metadata Preservation

Make sure `llm_mst_finder` request records preserve sample metadata into trial output.

At minimum, each completed request log should contain:

```json
{
  "request_id": "crosscodeeval-python-000001",
  "metadata": {
    "dataset": "crosscodeeval",
    "task": "cross_file_materialized",
    "language": "python",
    "repo_id": "owner/repo",
    "file_path": "src/foo.py",
    "session_id": "crosscodeeval::python::owner/repo::src",
    "prompt_token_count": 1024,
    "target_token_count": 17,
    "content_hash": "..."
  }
}
```

This is important for later analysis:

```text
energy/request by language
latency by prompt length bucket
output length distribution
repo-local cache behavior
in_file vs cross_file comparison
```

---

# 18. MVP Implementation Order

## MVP-1: Schema + CrossCodeEval Adapter

Implement:

```text
schema.py
adapters.py
crosscodeeval.py
prompting.py
filtering.py
export.py
```

Deliverable:

```text
processed.canonical.jsonl
processed.runner.jsonl
filtering_report.json
```

No sharding yet.

---

## MVP-2: Cache-Realistic Sharding

Implement:

```text
sampling.py
sharding.py
inspect.py
```

Deliverable:

```text
shards/
shards_manifest.json
sampling_report.json
```

---

## MVP-3: Wire Into `llm_mst_finder.workload`

Add workload type:

```text
code_completion_jsonl
```

Acceptance test:

```text
existing fixed-rate trial can run from shard_000.runner.jsonl
```

---

## MVP-4: Orchestrator Manifest Support

Add manifest support for:

```yaml
workload:
  type: code_completion_jsonl
  shard_dir: ...
  shard_selection: sequential
```

Acceptance test:

```text
local_orchestrator launches vLLM and runs MST over CrossCodeEval shards
```

---

## MVP-5: Energy Profiler Compatibility

Ensure energy profiler can consume MST output with workload metadata.

Acceptance test:

```text
energy_profiler runs fixed-rate energy trial using selected CrossCodeEval shard
```

---

## MVP-6: RepoBench Adapter

Add RepoBench after CrossCodeEval is stable.

Main deliverable:

```text
repobench_2k
repobench_4k
repobench_8k
repobench_16k
```

---

## MVP-7: M2RC-EVAL Adapter

Add M2RC-EVAL for multilingual balanced profiling.

---

# 19. Acceptance Criteria

## Dataset Preparation

A successful preparation run must satisfy:

```text
all samples have unique sample_id
all samples have prompt
all samples have content_hash
all samples have max_tokens
all samples have metadata.dataset
all samples have metadata.task
filtering report exists
sampling report exists
shards manifest exists
```

## Sharding

A successful sharding run must satisfy:

```text
no duplicate content_hash inside each shard
no duplicate content_hash across adjacent shards by default
session statistics are reported
language distribution is reported
prompt token distribution is reported
```

## MST Integration

A successful MST trial must satisfy:

```text
llm_mst_finder can load code_completion_jsonl
request_client sends normal OpenAI-compatible requests
trial outputs preserve request metadata
no dataset-specific code exists inside request_client
```

## Energy Integration

A successful energy trial must satisfy:

```text
energy_profiler can select a prepared shard
GPU energy/request is computed as before
summary includes dataset/task/cache_policy/shard path
```

---

# 20. Important Non-Goals

Do not implement these in the dataset layer:

```text
online retrieval during profiling
agentic coding loops
repo editing
test execution
web search
MCP tool calls
memory
server lifecycle management
GPU monitoring
energy integration
MST search logic
```

Those either already exist in your repo or are outside the intended workload scope.

The dataset layer’s job is only:

```text
normalize
prompt
filter
sample
shard
export
```

---

# 21. Final Default Workload

The default workload for your energy profiling should be:

```text
dataset: CrossCodeEval
mode: cross_file_materialized
sampling: cache_realistic
endpoint: /v1/completions
max_tokens: 32
temperature: 0.0
prefix cache: enabled on server
server restart between adjacent rates: false
duplicate prompts across adjacent shards: forbidden
```

This gives you a realistic IDE-style code completion workload while keeping the profiling runtime path clean and compatible with your existing infrastructure.
