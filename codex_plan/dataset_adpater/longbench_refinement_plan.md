# LongBench Refinement Plan

## Current Decision

LongBench should be refined through the dataset materialization path, not by
adding LongBench-specific behavior to `llm_mst_finder`.

The serving problem is that a single coarse LongBench workload mixes tasks with
very different intent, prompt length, and output length. Random sampling across
that pool can move the observed MST substantially between runs, even when SLOs
are lifted. The fix is to prepare deterministic, auditable LongBench shards with
clear profiles before profiling starts.

This plan intentionally does not implement LongBench refinement yet. It defines
the repo-aware boundary and the shape of the future work.

## Boundary

Use `profiler/dataset_workload_materializer` as the dataset preparation module.
The abstraction is:

```text
dataset-specific raw artifacts
  -> materialized offline JSONL shards
  -> ordinary llm_mst_finder workload YAMLs
  -> sequential profiling over those shards
```

Keep these components dataset-agnostic:

- `request_client`
- `trial_runner`
- `search`
- orchestrators
- energy profiler
- GPU monitor

Do not add a new `llm_mst_finder` workload type for refined LongBench. The
generated YAMLs should use the existing `dataset.type: jsonl` path with
`sampling.entry_selection: sequential`.

## Existing Infrastructure To Reuse

The materializer already writes:

- `materialization_config.yaml`
- `materialization_report.json`
- `shards_manifest.json`
- `shards/*.runner.jsonl`
- `workload_yamls/*.yaml`

It already supports fail-fast filtering, deterministic shard ordering, row-level
metadata, and ordinary workload YAML generation. LongBench should reuse this
contract instead of inventing a second workload generator.

## LongBench Problem Shape

The current LongBench-NL workload is too broad for one MST number. It includes:

- long-output summarization tasks
- medium-output synthesis tasks
- document QA tasks
- short-answer benchmark QA
- synthetic retrieval/counting tasks
- few-shot classification tasks
- few-shot dialogue summarization
- extreme reading-comprehension stress cases

Those categories stress different parts of serving. Mixing them by random draw
makes the profile unstable and makes the final MST hard to interpret.

## Candidate Realistic-NL Profiles

The future materializer support should expose task profiles in config, not hard
code one universal LongBench mix.

### Long-Output Summarization

```text
gov_report
gov_report_e
```

Shape: long report/document to long summary.

This is decode-heavy compared with most LongBench tasks and should have its own
SLO/goodput interpretation.

### Medium-Output Summarization / Synthesis

```text
multi_news
multi_news_e
qmsum
vcsum
```

Shape: multiple documents or meeting transcript to medium-length summary.

This profile is still generation-heavy, but less extreme than `gov_report`.

### Medium-Answer RAG-Style QA

```text
dureader
```

Shape: retrieved documents to answer paragraph.

This is a mixed prefill/decode workload and should not be averaged with
long-output summarization.

### Short-Answer Document QA

```text
multifieldqa_en
multifieldqa_en_e
multifieldqa_zh
qasper
qasper_e
```

Shape: long document or paper-like text to short factual answer.

This is mostly prefill-dominated. It can be useful, but it should be reported as
document QA rather than natural-language summarization.

## Tasks To Exclude From Realistic-NL Profiles

Exclude these from the default realistic-NL profiles:

```text
2wikimqa
2wikimqa_e
hotpotqa
hotpotqa_e
musique
triviaqa
triviaqa_e
trec
trec_e
lsht
samsum
samsum_e
passage_count
passage_count_e
passage_retrieval_en
passage_retrieval_en_e
passage_retrieval_zh
narrativeqa
```

Rationale:

- `passage_count*` and `passage_retrieval*` are synthetic retrieval/counting
  tests.
- `2wikimqa*`, `hotpotqa*`, `musique`, and `triviaqa*` are benchmark-style
  short-answer QA.
- `trec*` and `lsht` are classification-style workloads.
- `samsum*` is realistic in principle, but this LongBench version is few-shot
  formatted; much of the prompt is examples rather than the target request.
- `narrativeqa` is an extreme long-input, very-short-output stress case.
- LongBench code tasks belong with code-completion workloads, not realistic-NL.

## Future Config Shape

Keep this in a materializer config, for example:

```yaml
name: longbench_realistic_nl_summarization
dataset:
  name: longbench
  raw_path: ../../data/raw/longbench
  split: test
  profile: medium_output_summarization
  configs:
    - multi_news
    - multi_news_e
    - qmsum
    - vcsum

tokenization:
  tokenizer: Qwen/Qwen3-8B

filtering:
  min_prompt_tokens: 128
  max_prompt_tokens: 32768
  min_target_tokens: 1
  max_target_tokens: 2048

sampling:
  seed: 23
  policy: task_uniform
  samples_per_task: 256

sharding:
  output_dir: longbench_realistic_nl_summarization_qwen3_8b
  samples_per_shard: 1024
```

The exact raw path format can be decided during implementation. If we reuse the
Hugging Face cache, the materializer should fail clearly when the needed
LongBench artifacts are missing locally.

## Sampling Rules

Sampling randomness should end at materialization time.

The generated profiling YAMLs should remain sequential. Supported materializer
sampling policies can be added later, but they should produce explicit shard
contents and summary counts:

- `task_uniform`: fixed or equal sample count per task.
- `bucket_uniform`: equal representation per profile bucket, then per task.
- `bucket_weighted`: explicit user-provided bucket weights.

Do not hide a broad randomized LongBench draw behind runtime sampling.

## Metadata

Each materialized LongBench row should carry enough metadata to audit the
profile:

```json
{
  "dataset": "longbench",
  "dataset_kind": "long_context_nlp",
  "task": "qmsum",
  "profile": "medium_output_summarization",
  "workload_type": "summarization",
  "output_regime": "medium",
  "language": "en",
  "prompt_token_count": 13009,
  "target_token_count": 78
}
```

Reports and manifests should include per-profile and per-task sample counts plus
prompt/target length summaries.

## SLO Guidance

Do not use one canonical LongBench SLO across all refined profiles.

Summarization profiles can inherit the literature-style LongBench reference
point as a baseline, but document-QA profiles should be interpreted separately
because their outputs are short and their serving behavior is prefill-dominated.
The experiment manifests should set SLOs per materialized LongBench profile.

## Implementation Phases

1. Keep the current materializer boundary generic and documented.
2. Add an inspection-only script or materializer dry-run that reports LongBench
   task statistics from local artifacts.
3. Add LongBench materialization support that emits JSONL shards and ordinary
   workload YAMLs.
4. Add profile configs for the realistic-NL buckets above.
5. Add focused tests for task filtering, deterministic sampling, metadata, and
   YAML loading through `llm_mst_finder`.
6. Add experiment manifests that profile separate LongBench profiles rather
   than the whole LongBench-NL pool.

## Non-Goals

- No LongBench-specific request formatting inside `request_client`.
- No LongBench-specific runtime sampling inside `trial_runner` or `search`.
- No new profiler/orchestrator concept for LongBench.
- No implementation of the refined LongBench loader in this cleanup step.
