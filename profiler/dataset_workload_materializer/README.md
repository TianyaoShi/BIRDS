# Dataset Workload Materializer

This package prepares dataset-specific profiling workloads for
`llm_mst_finder` without adding dataset-specific behavior to the request client,
trial runner, search loop, or profiler paths.

The import path is `dataset_workload_materializer`. Treat it as the dataset
materialization boundary: raw dataset artifacts go in, offline JSONL shards and
ordinary `llm_mst_finder` workload YAMLs come out.

The materializer produces offline JSONL shards plus ordinary
`llm_mst_finder` workload YAMLs. The generated workload YAMLs use
`dataset.type: jsonl` and `sampling.entry_selection: sequential`, so profiling
consumes exactly the materialized shard order.

## Supported Inputs

Implemented today:

- CrossCodeEval-like local JSONL files or directories of JSONL files.
- RepoBench local parquet artifacts, including aggregate mode across languages
  and tasks.
- LongBench realistic-NL profile materialization from local `data.zip` or
  unpacked task JSONL artifacts.
- Reasoning QA local JSONL files or directories for GPQA, MMLU, MMLU-Pro, and
  AIME-style rows.

CrossCodeEval-like JSONL supports field aliases configured under
`dataset.field_aliases`. The default logical fields are:

- `prompt`
- `target`
- `language`
- `repo_id`
- `file_path`
- `cross_file_context`
- `sequence_index`

Dataset-specific support is intentionally compact and stays in `materialize.py`;
there is no one-file-per-dataset split. Reports, manifests, and row metadata now
include `dataset_kind` so future non-code profiles can be distinguished without
changing the runner contract.

## Code Prompt Format

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

Current code-completion real-dataset configs live under
`experiments/code_workloads/`:

```text
experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic.yaml
experiments/code_workloads/repobench_python_java_aggregate_cache_realistic.yaml
experiments/code_workloads/repobench_python_java_aggregate_cache_realistic_8k_drop.yaml
```

These code-completion configs explicitly set:

```yaml
dataset:
  prompt_template: plain_prefix
```

LongBench profile materialization configs live under
`experiments/longbench_workloads/`.

Reasoning-question materialization configs live under
`experiments/reasoning_workloads/`:

```text
experiments/reasoning_workloads/gpqa_diamond_reasoning.yaml
experiments/reasoning_workloads/mmlu_reasoning.yaml
experiments/reasoning_workloads/mmlu_pro_reasoning.yaml
experiments/reasoning_workloads/aime_2024_2026_reasoning.yaml
```

## Materialize

Run from the repository root with the project environment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m dataset_workload_materializer.cli prepare \
  --config experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic.yaml
```

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m dataset_workload_materializer.cli prepare \
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
git; the same is true for generated directories under
`experiments/reasoning_workloads/*/`. The source config YAMLs are tracked.

## Reasoning QA Workloads

Reasoning QA materialization accepts local JSONL file or directory inputs. Rows
may be multiple-choice or free-response. Supported common fields include:

- Question text: `question`, `Question`, `prompt`, `input`, `problem`, `Problem`.
- Choices: `choices`, `options`, `answer_choices`, `Options`, or per-choice
  fields such as `A`, `B`, `C`, `D`.
- GPQA-style rows with `Correct Answer` and `Incorrect Answer 1` through
  `Incorrect Answer 3`.
- Answer labels or values: `answer`, `Answer`, `target`, `correct_answer`,
  `label`, `answer_idx`, `answer_index`, `gold`, `final_answer`.

The generated JSONL keeps final answers in `metadata.ground_truth` and
`metadata.ground_truth_text`, but generated workload YAMLs should use:

```yaml
sampling:
  output_len:
    mode: natural_until_eos
    max_tokens: 4096
request:
  ignore_eos: false
```

This is deliberate: GPQA, MMLU/MMLU-Pro, and AIME provide final answers, not
reference reasoning traces. The generation length is therefore a safety cap for
natural EOS stopping, not a target derived from the final-answer token count.

Recent real materialization counts:

- CrossCodeEval RG1 UnixCoder: 9,927 samples, 2 shards.
- RepoBench Python+Java aggregate: 46,781 samples, 6 shards.
- RepoBench Python+Java aggregate 8k-drop: 41,957 samples, 6 shards.

## Length Statistics

The materialized rows record whitespace-tokenized prompt and target lengths in
`metadata.prompt_token_count` and `metadata.target_token_count`. These are the
dataset/materializer token counts used by the generated workload YAMLs.

Tracked real workload summaries:

| Workload | Samples | Input mean | Input p50 | Input p90 | Input p95 | Input p99 | Input max | Output mean | Output p50 | Output p90 | Output p95 | Output p99 | Output max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CrossCodeEval RG1 UnixCoder | 9,927 | 583.5 | 490 | 963.4 | 1,221.4 | 1,842.4 | 4,517 | 4.0 | 3 | 8 | 10 | 14.7 | 37 |
| RepoBench Python+Java aggregate | 46,781 | 2,910.6 | 1,010 | 8,308 | 10,880 | 16,519 | 19,999 | 4.0 | 3 | 7 | 9 | 14 | 113 |
| RepoBench Python+Java aggregate 8k-drop | 41,957 | 1,885.3 | 691 | 5,366 | 6,732 | 7,858 | 8,192 | 4.0 | 3 | 7 | 9 | 14 | 72 |

RepoBench aggregate includes three task shapes:

| RepoBench task | Samples | Input mean | Input p50 | Input p90 | Input p95 | Input p99 | Output mean | Output p50 | Output p90 | Output p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `in_file` | 15,980 | 302.6 | 285 | 526 | 607 | 895.6 | 3.5 | 3 | 7 | 9 |
| `cross_file_first` | 15,768 | 4,357.4 | 3,196 | 10,269.7 | 12,702.9 | 18,589.9 | 4.2 | 4 | 7 | 9 |
| `cross_file_random` | 15,033 | 4,165.4 | 3,079 | 9,686.8 | 12,348.6 | 17,589.1 | 4.5 | 4 | 8 | 10 |

The important profiling consequence is that code-completion outputs are tiny:
p90 output is only 7-8 tokens and p95 is 9-10 tokens. Compared with WildChat,
TPOT is less informative than TTFT/prefill behavior. CrossCodeEval is a
moderate-input workload; RepoBench aggregate is dominated by cross-file prompt
prefill and has a long input tail.

### LongBench Comparison

LongBench-style serving papers commonly use a much looser summarization SLO
than interactive code completion. DistServe and GreenLLM both report LongBench
summarization with TTFT SLO 15s and TPOT SLO 150ms; DistServe also reports a
separate code-completion SLO of TTFT 125ms and TPOT 200ms. That 15s target is a
reasonable long-document summarization reference point, but it should not be
copied onto RepoBench: RepoBench prompts can be long, but the expected output is
still a single-line completion with p90/p95 around 7/9 tokens.

Reference points:

- DistServe: https://arxiv.org/abs/2401.09670
- GreenLLM: https://arxiv.org/abs/2412.20322

The local `llm_mst_finder` LongBench workload caches show the same split. The
NL workload is long-context and often long-output summarization:

The refined LongBench materialization keeps these regimes in separate buckets.
The buckets are intentionally not mixed at profiling time because their output
length distributions and task intentions differ. Since the smallest bucket has
only about 200 unique examples and the others are still below 1k examples, the
materializer supports deterministic corpus expansion:

```yaml
sampling:
  repeat_policy: epoch_shuffle
  target_samples: 1024
```

`epoch_shuffle` writes a larger ordinary JSONL shard by replaying the unique
corpus in shuffled epochs. Every full epoch contains each unique row exactly
once, and the row metadata records `original_sample_id`, `epoch_index`,
`epoch_position`, `unique_sample_count`, and `expanded_sample_count`. The
generated `llm_mst_finder` workload YAML still uses
`sampling.entry_selection: sequential`, so runtime profiling stays deterministic
and dataset-specific behavior remains in materialization.

For current LongBench buckets, 1k-2k expanded samples are enough because each
request carries long input and normally takes several seconds. The tracked
configs use 1,024 rows for `long_output_summarization` and
`medium_answer_rag_qa`, and 2,048 rows for `medium_output_summarization` and
`short_answer_document_qa`.

| LongBench-NL task | Samples | Input p50 | Input p90 | Input p95 | Output p50 | Output p90 | Output p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | 95 | 6,733 | 13,567 | 15,504 | 4 | 8 | 9 |
| `2wikimqa_e` | 128 | 8,427 | 16,041 | 16,667 | 4 | 7 | 8 |
| `dureader` | 92 | 10,575 | 14,275 | 16,289 | 60 | 202 | 259 |
| `gov_report` | 85 | 8,055 | 16,844 | 18,418 | 731 | 934 | 975 |
| `gov_report_e` | 128 | 7,533 | 15,773 | 19,130 | 757 | 927 | 975 |
| `hotpotqa` | 91 | 14,490 | 17,065 | 17,161 | 4 | 6 | 9 |
| `hotpotqa_e` | 139 | 9,720 | 16,406 | 16,718 | 4 | 9 | 10 |
| `lsht` | 84 | 13,442 | 21,963 | 23,028 | 3 | 5 | 5 |
| `multi_news` | 98 | 2,203 | 4,812 | 6,211 | 300 | 402 | 420 |
| `multi_news_e` | 122 | 7,644 | 16,711 | 18,603 | 334 | 480 | 537 |
| `multifieldqa_en` | 60 | 6,948 | 10,462 | 11,483 | 15 | 31 | 37 |
| `multifieldqa_en_e` | 73 | 7,468 | 13,233 | 13,723 | 15 | 29 | 36 |
| `multifieldqa_zh` | 84 | 3,958 | 7,568 | 8,520 | 10 | 25 | 35 |
| `musique` | 81 | 16,788 | 17,209 | 17,356 | 4 | 8 | 9 |
| `narrativeqa` | 71 | 31,315 | 46,729 | 64,142 | 5 | 12 | 14 |
| `passage_count` | 84 | 15,545 | 21,073 | 23,242 | 2 | 2 | 2 |
| `passage_count_e` | 132 | 8,142 | 13,153 | 13,736 | 1 | 2 | 2 |
| `passage_retrieval_en` | 77 | 12,875 | 14,545 | 15,006 | 4 | 4 | 4 |
| `passage_retrieval_en_e` | 136 | 7,677 | 12,891 | 13,301 | 3 | 4 | 4 |
| `passage_retrieval_zh` | 78 | 4,614 | 5,281 | 5,525 | 4 | 4 | 4 |
| `qasper` | 89 | 4,367 | 6,680 | 7,486 | 16 | 53 | 90 |
| `qasper_e` | 94 | 5,473 | 7,603 | 10,678 | 18 | 61 | 76 |
| `qmsum` | 104 | 13,009 | 23,456 | 29,670 | 78 | 130 | 151 |
| `samsum` | 89 | 9,670 | 15,897 | 16,490 | 25 | 49 | 52 |
| `samsum_e` | 131 | 9,256 | 15,372 | 16,283 | 23 | 41 | 51 |
| `trec` | 89 | 6,445 | 10,802 | 11,158 | 2 | 3 | 4 |
| `trec_e` | 135 | 8,612 | 15,309 | 16,759 | 2 | 3 | 4 |
| `triviaqa` | 90 | 11,130 | 20,299 | 22,045 | 6 | 9 | 11 |
| `triviaqa_e` | 146 | 9,982 | 18,628 | 20,267 | 6 | 10 | 11 |
| `vcsum` | 95 | 7,385 | 15,577 | 17,456 | 144 | 174 | 181 |

LongBench-code is closer to RepoBench in output length, but it is still a
benchmark workload rather than an IDE latency target:

| LongBench-code task | Samples | Input p50 | Input p90 | Input p95 | Output p50 | Output p90 | Output p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lcc` | 500 | 883 | 2,181 | 2,864 | 4 | 7 | 9 |
| `lcc_e` | 300 | 5,064 | 10,378 | 11,925 | 4 | 7 | 8 |
| `repobench-p` | 500 | 3,507 | 7,578 | 9,051 | 3 | 7 | 8 |
| `repobench-p_e` | 300 | 5,252 | 10,892 | 12,663 | 3 | 7 | 8 |

Because of this, the code-workload manifest uses the RepoBench 8k-drop
materialization and caps RepoBench TTFT at 5s. This is still slow for a one-line
IDE completion, but it avoids treating a 12-15s first-token delay as acceptable
just because the prompt has a long retrieved context tail.

### RepoBench Context Limits

The cleanest place to enforce a 4k/8k RepoBench prompt budget is the code
workload materializer, before shard writing. The materializer is where
dataset-specific semantics live, and it can apply an IDE-realistic retrieval
budget while preserving the current-file prefix and dropping or shortening
retrieved cross-file context. The `llm_mst_finder` workload path should remain a
generic runner over already-materialized JSONL rows.

The immediate drop-overlength option is config-only: lower
`filtering.max_prompt_tokens` in
`experiments/code_workloads/repobench_python_java_aggregate_cache_realistic.yaml`
from its current 20k ceiling to 8k or 4k and rematerialize. That gives a strict,
fail-fast dataset slice without adding dataset logic to sampling or request
execution. A smarter future option is a materializer-side truncation policy that
keeps the current file and cursor-near prefix intact, then truncates retrieved
context blocks to a configured token budget before final token counting.

The current 8k-drop config materializes to
`experiments/code_workloads/repobench_python_java_aggregate_cache_realistic_8k_drop/`.
Its report shows 6,416 rows dropped for `prompt_too_long`, reducing the prompt
distribution from p90/p95 8,308/10,880 tokens to p90/p95 5,366/6,732 tokens.

## Experiment Manifest

The L40 single-GPU manifest for these workloads is:

```text
experiments/single_gpu_cached_models_l40_code_workloads.yaml
```

It uses `/v1/completions`, not chat completions, and runs one representative
materialized shard per dataset:

```text
experiments/code_workloads/crosscodeeval_rg1_unixcoder_cache_realistic/workload_yamls/shard_000.yaml
experiments/code_workloads/repobench_python_java_aggregate_cache_realistic_8k_drop/workload_yamls/shard_000.yaml
```

The manifest intentionally does not expand over every RepoBench shard. RepoBench
is already aggregated across Python, Java, and the supported task modes before
sharding. `shard_000` is therefore a representative profiling slice used to
avoid multiplying each model by every shard. This is a throughput/goodput probe,
not full-dataset evaluation; if representativeness becomes critical, the next
step is to have the materializer emit a dedicated sampled profiling shard rather
than launching every physical shard.

SLOs are tuned around the measured input/output profile:

- CrossCodeEval: moderate prompts, p90 output 8 tokens. TTFT targets stay
  interactive-code oriented, with tighter SLOs for smaller models.
- RepoBench aggregate 8k-drop: p90 input 5.4k tokens and p95 input 6.7k tokens,
  but p90 output only 7 tokens. TTFT is prefill-oriented but capped at 5s
  because this is still an IDE completion workload; TPOT remains relatively
  tight because completions are short.

`openai/gpt-oss-20b` is omitted from this manifest because the live probes showed
it is not a useful `/v1/completions` code-completion model in this setup.
Llama-2 models are also omitted because their shorter context window is a poor
fit for the RepoBench aggregate input tail.

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
/local/scratch/a/shi676/.venv/bin/python -m dataset_workload_materializer.live_model_prompt_loop \
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
  tests/dataset_workload_materializer/test_materialize.py \
  tests/dataset_workload_materializer/test_live_code_workloads.py \
  -q
```

The live test is opt-in:

```bash
DATASET_WORKLOAD_MATERIALIZER_RUN_LIVE=1 \
CODE_WORKLOAD_LIVE_MODEL=google/gemma-4-E4B-it \
CODE_WORKLOAD_LIVE_BASE_URL=http://127.0.0.1:8000 \
CODE_WORKLOAD_LIVE_LOG_PATH=results/live_code_workload_smoke/decoded_responses.jsonl \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler \
/local/scratch/a/shi676/.venv/bin/python -m pytest \
  tests/dataset_workload_materializer/test_live_code_workloads.py -q -s
```
