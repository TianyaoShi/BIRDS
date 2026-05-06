Below is a compact coding plan you can hand to an implementation agent.

---

# Coding Plan: Refined LongBench Natural-Language Workload Sampler

## Objective

The current workload generator samples uniformly or globally from the entire LongBench task pool. Refine it into a **realistic natural-language long-text workload sampler** by:

1. Excluding synthetic, few-shot classification, trivia/multi-hop exam-style QA, code tasks, and `narrativeqa`.
2. Keeping only tasks that resemble real long-text NLP service requests.
3. Grouping retained tasks into workload buckets according to task type and output-length regime.
4. Supporting both bucket-level sampling and task-level sampling within each bucket.

The goal is not to evaluate all LongBench capabilities, but to generate realistic serving workloads for long-context natural-language processing.

---

## Tasks to Include

Retain the following LongBench-NL tasks only:

```text
gov_report
gov_report_e
multi_news
multi_news_e
qmsum
vcsum
dureader
multifieldqa_en
multifieldqa_en_e
multifieldqa_zh
qasper
qasper_e
```

Exclude all code tasks and all other LongBench-NL tasks.

---

## Tasks to Exclude

Explicitly exclude:

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

* `passage_count*` and `passage_retrieval*`: synthetic retrieval/counting tests.
* `2wikimqa*`, `hotpotqa*`, `musique`, `triviaqa*`: benchmark-style QA with very short answers.
* `trec*`, `lsht`: few-shot classification, not natural long-document processing.
* `samsum*`: dialogue summarization is realistic in principle, but LongBench’s version is few-shot formatted; the long input mostly comes from concatenated examples.
* `narrativeqa`: very long input but extremely short output; closer to reading-comprehension stress testing than realistic production long-text processing.
* Code tasks: outside the natural-language workload scope.

---

## Workload Buckets

Implement four buckets.

### Bucket 1: Long-output summarization

```text
gov_report
gov_report_e
```

Workload shape:

```text
long report/document -> long summary
```

Expected profile:

* Long input.
* Very long output, typically hundreds to about one thousand tokens.
* Decode-heavy compared with most LongBench tasks.
* Useful for stressing sustained generation throughput, TPOT, KV-cache residency, and decode-side batching.

Suggested bucket name:

```python
"long_output_summarization"
```

---

### Bucket 2: Medium-output summarization / synthesis

```text
multi_news
multi_news_e
qmsum
vcsum
```

Workload shape:

```text
multiple documents / meeting transcript -> medium-length summary
```

Expected profile:

* Medium-to-long input.
* Medium-length output.
* Represents practical summarization workloads such as news synthesis and meeting summarization.
* Less decode-heavy than `gov_report`, but much more realistic than short-answer QA for long-text generation.

Suggested bucket name:

```python
"medium_output_summarization"
```

---

### Bucket 3: Medium-answer RAG-style QA

```text
dureader
```

Workload shape:

```text
retrieved documents -> answer paragraph
```

Expected profile:

* Long input.
* Medium-length output.
* Closer to search assistant, enterprise RAG, or document-grounded answer generation.
* Useful for evaluating mixed prefill/decode behavior.

Suggested bucket name:

```python
"medium_answer_rag_qa"
```

---

### Bucket 4: Short-answer document QA

```text
multifieldqa_en
multifieldqa_en_e
multifieldqa_zh
qasper
qasper_e
```

Workload shape:

```text
long document / paper / PDF-like text -> short factual answer
```

Expected profile:

* Medium-to-long input.
* Short output.
* Mostly prefill-dominated.
* Useful for long-context document-reading workloads, but should not be mixed directly with summarization workloads when reporting serving performance.

Suggested bucket name:

```python
"short_answer_document_qa"
```

---

## Recommended Configuration Structure

Add a workload configuration similar to:

```python
REALISTIC_LONGBENCH_BUCKETS = {
    "long_output_summarization": [
        "gov_report",
        "gov_report_e",
    ],
    "medium_output_summarization": [
        "multi_news",
        "multi_news_e",
        "qmsum",
        "vcsum",
    ],
    "medium_answer_rag_qa": [
        "dureader",
    ],
    "short_answer_document_qa": [
        "multifieldqa_en",
        "multifieldqa_en_e",
        "multifieldqa_zh",
        "qasper",
        "qasper_e",
    ],
}
```

Also define the explicit exclusion list:

```python
EXCLUDED_LONGBENCH_TASKS = {
    "2wikimqa",
    "2wikimqa_e",
    "hotpotqa",
    "hotpotqa_e",
    "musique",
    "triviaqa",
    "triviaqa_e",
    "trec",
    "trec_e",
    "lsht",
    "samsum",
    "samsum_e",
    "passage_count",
    "passage_count_e",
    "passage_retrieval_en",
    "passage_retrieval_en_e",
    "passage_retrieval_zh",
    "narrativeqa",
}
```

---

## Sampling Modes to Implement

### 1. Bucket-uniform sampling

Sample a bucket uniformly, then sample a task within that bucket.

```text
bucket ~ Uniform(realistic_buckets)
task   ~ Uniform(tasks_in_bucket)
sample ~ Uniform(samples_in_task)
```

Use this when the goal is to give each workload class equal representation.

---

### 2. Task-uniform sampling

Sample uniformly from all included tasks.

```text
task   ~ Uniform(all_included_tasks)
sample ~ Uniform(samples_in_task)
```

Use this when the goal is to avoid overrepresenting buckets with fewer tasks, such as `dureader`.

---

### 3. Weighted bucket sampling

Allow user-configurable bucket weights, for example:

```python
bucket_weights = {
    "long_output_summarization": 0.25,
    "medium_output_summarization": 0.35,
    "medium_answer_rag_qa": 0.20,
    "short_answer_document_qa": 0.20,
}
```

Then:

```text
bucket ~ Categorical(bucket_weights)
task   ~ Uniform(tasks_in_bucket)
sample ~ Uniform(samples_in_task)
```

This should be the default for production-like workload generation, because different workload classes have very different prefill/decode ratios.

---

## Metadata to Attach to Each Generated Request

Each sampled request should carry at least:

```python
{
    "task": task_name,
    "bucket": bucket_name,
    "input_tokens": input_length,
    "output_tokens": output_length,
    "language": "en" | "zh" | "mixed_or_unknown",
    "workload_type": "summarization" | "qa",
    "output_regime": "long" | "medium" | "short",
}
```

Recommended mapping:

```python
BUCKET_METADATA = {
    "long_output_summarization": {
        "workload_type": "summarization",
        "output_regime": "long",
    },
    "medium_output_summarization": {
        "workload_type": "summarization",
        "output_regime": "medium",
    },
    "medium_answer_rag_qa": {
        "workload_type": "qa",
        "output_regime": "medium",
    },
    "short_answer_document_qa": {
        "workload_type": "qa",
        "output_regime": "short",
    },
}
```

Language can be inferred from task name:

```python
zh_tasks = {"vcsum", "dureader", "multifieldqa_zh"}
```

Everything else can be treated as English unless the dataset metadata says otherwise.

---

## Implementation Steps

1. Add a new workload profile, for example:

```text
longbench_realistic_nl
```

2. Define the retained task set from `REALISTIC_LONGBENCH_BUCKETS`.

3. During dataset loading, filter out all tasks not in the retained set.

4. Implement three sampling policies:

```text
bucket_uniform
task_uniform
bucket_weighted
```

5. Make `bucket_weighted` the default policy.

6. Add CLI/config options:

```text
--longbench-profile realistic_nl
--sampling-policy bucket_weighted
--bucket-weights path/to/weights.json
--include-longbench-e true/false
```

7. If `--include-longbench-e false`, remove tasks ending with `_e`.

8. Attach bucket metadata to each generated request.

9. Log per-bucket and per-task sample counts after workload generation.

10. Add validation checks:

```text
- No excluded task appears in generated requests.
- Every included task belongs to exactly one bucket.
- Bucket weights sum to 1.0 if weighted sampling is used.
- Empty buckets raise a clear error.
```

---

## Expected Output Summary

At the end of generation, print or save a workload summary like:

```text
Generated LongBench realistic-NL workload:
- total requests: N
- sampling policy: bucket_weighted
- included tasks: 12
- excluded tasks: 18+

Bucket distribution:
- long_output_summarization: x requests
- medium_output_summarization: y requests
- medium_answer_rag_qa: z requests
- short_answer_document_qa: w requests

Task distribution:
- gov_report: ...
- gov_report_e: ...
- ...
```

This makes the generated workload auditable and prevents accidental regression back to whole-LongBench averaging.
