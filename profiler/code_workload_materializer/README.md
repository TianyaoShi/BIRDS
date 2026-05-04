# Code Workload Materializer

This package prepares code-completion datasets for `llm_mst_finder` without adding
dataset-specific behavior to the request client, trial runner, search loop,
or profiler paths.

The materializer produces offline JSONL shards plus ordinary
`llm_mst_finder` workload YAMLs. The generated workload YAMLs use
`dataset.type: jsonl` and `sampling.entry_selection: sequential`, so profiling
consumes exactly the materialized shard order.

## Supported Inputs

- CrossCodeEval-like local JSONL files or directories of JSONL files.
- RepoBench local parquet artifacts, including aggregate mode across languages
  and tasks.

CrossCodeEval-like JSONL supports field aliases configured under
`dataset.field_aliases`. The default logical fields are:

- `prompt`
- `target`
- `language`
- `repo_id`
- `file_path`
- `cross_file_context`
- `sequence_index`

RepoBench support is intentionally compact and stays in `materialize.py`; there
is no one-file-per-dataset split.

## Prompt Format

The default prompt template is `plain_prefix`.

For code-completion profiling, this means the prompt ends directly at the code
cursor. If cross-file context exists, it is prepended as plain text:

```text
Relevant repository context:
...

<raw code prefix ending at cursor>
```

The legacy XML-style format remains available as `prompt_template: xml_tags`,
but it should not be the default for profiling. Live probes showed XML boundary
tags often make instruction-tuned models continue the task statement instead of
continuing code.

`openai/gpt-oss-20b` is not currently recommended for this workload path. It
appears to be conversation-oriented around the Harmony tokenizer and did not
produce useful code-completion continuations through `/v1/completions` in the
plain-prefix probe.

## Configs

Current real-dataset configs live under `experiments/code_workloads/`:

```text
experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic.yaml
experiments/code_workloads/repobench_python_java_aggregate_cache_realistic.yaml
```

Both explicitly set:

```yaml
dataset:
  prompt_template: plain_prefix
```

## Materialize

Run from the repository root with the project environment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m code_workload_materializer.cli prepare \
  --config experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic.yaml
```

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m code_workload_materializer.cli prepare \
  --config experiments/code_workloads/repobench_python_java_aggregate_cache_realistic.yaml
```

The generated output directory contains:

```text
materialization_config.yaml
materialization_report.json
shards_manifest.json
shards/*.runner.jsonl
workload_yamls/*.yaml
```

The generated directories under `experiments/code_workloads/*/` are ignored by
git; the source config YAMLs are tracked.

Recent real materialization counts:

- CrossCodeEval RG1 UnixCoder: 9,927 samples, 2 shards.
- RepoBench Python+Java aggregate: 46,781 samples, 6 shards.

## `llm_mst_finder` Integration

Generated YAMLs are ordinary workload YAMLs. Example:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -c "from llm_mst_finder.workload import prepare_workload_for_trial; p=prepare_workload_for_trial('experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic/workload_yamls/shard_000.yaml', model_name='fake-model'); print(len(p.samples), p.samples[0].metadata['sampling_entry_selection'])"
```

`llm_mst_finder` caches JSONL workload manifests. The cache key includes a
source-file fingerprint, so rematerialized shards invalidate stale cached rows.
If a probe unexpectedly shows old prompts, check for stale cache behavior first.

## Live Diagnostics

`live_model_prompt_loop.py` runs a small response-inspection loop over one or
more single-GPU models from `experiments/single_gpu_cached_models_l40.yaml`.
Llama-2 models are excluded by default because their context length is usually
too short for these workloads.

Example single-model probe on GPU 0:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m code_workload_materializer.live_model_prompt_loop \
  --model Qwen/Qwen3-8B \
  --samples-per-workload 8 \
  --max-output-tokens 64 \
  --output-dir results/live_code_model_prompt_loop_qwen3_8b_plain_cachefixed \
  --gpu-id 0 \
  --preserve-workload-decode
```

The loop logs full prompt, ground truth, decoded response, prompt mode, and
summary counts to:

```text
<output-dir>/responses.jsonl
<output-dir>/summary.json
<output-dir>/server_logs/
```

Use `--preserve-workload-decode` when checking behavior under the same stop/eos
settings used by profiling. Omit it when intentionally giving a model more room
to produce visible diagnostic text.

## Tests

Targeted tests:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m pytest \
  tests/llm_mst_finder/test_workload.py \
  tests/code_workload_materializer/test_materialize.py \
  tests/code_workload_materializer/test_live_code_workloads.py \
  -q
```

The live test is opt-in:

```bash
CODE_WORKLOAD_MATERIALIZER_RUN_LIVE=1 \
CODE_WORKLOAD_LIVE_MODEL=google/gemma-4-E4B-it \
CODE_WORKLOAD_LIVE_BASE_URL=http://127.0.0.1:8000 \
CODE_WORKLOAD_LIVE_LOG_PATH=results/live_code_workload_smoke/decoded_responses.jsonl \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m pytest \
  tests/code_workload_materializer/test_live_code_workloads.py -q -s
```

