# Coding Plan: Minimal Dataset Expansion for Realistic Long-Text Workloads

## Goal

Extend the current LongBench-based sampler with six external datasets to increase sample diversity and reduce repeated-prefix / prefix-cache artifacts.

Do **not** change the existing four-bucket taxonomy.

---

## Datasets to Add

```python
EXTERNAL_DATASETS = {
    "long_output_summarization": [
        "gov_report_original",
    ],
    "medium_output_summarization": [
        "multi_news_original",
        "qmsum_original",
        "meetingbank",
    ],
    "medium_answer_rag_qa": [
        "dureader_full",
    ],
    "short_answer_document_qa": [
        "qasper_full",
    ],
}
```

---

## Dataset Adapter Interface

Each dataset adapter should output normalized records:

```python
{
    "dataset": str,
    "bucket": str,
    "sample_id": str,
    "group_id": str,
    "prompt": str,
    "reference_output": str | None,
    "metadata": dict,
}
```

Required fields:

* `dataset`: dataset name.
* `bucket`: one of the four workload buckets.
* `sample_id`: unique example ID.
* `group_id`: shared source-document ID used for cache-aware sampling.
* `prompt`: final model input.
* `reference_output`: gold summary/answer if available.
* `metadata`: optional raw IDs, language, split, length stats.

---

## Dataset-Specific Mapping

```python
DATASET_BUCKETS = {
    "gov_report_original": "long_output_summarization",
    "multi_news_original": "medium_output_summarization",
    "qmsum_original": "medium_output_summarization",
    "meetingbank": "medium_output_summarization",
    "dureader_full": "medium_answer_rag_qa",
    "qasper_full": "short_answer_document_qa",
}
```

---

## `group_id` Definition

Use `group_id` to prevent repeatedly sampling requests with the same long context.

```python
GROUP_ID_FIELDS = {
    "gov_report_original": "report_id",
    "multi_news_original": "article_cluster_id",
    "qmsum_original": "meeting_id",
    "meetingbank": "meeting_id",
    "dureader_full": "question_id_or_document_cluster_id",
    "qasper_full": "paper_id",
}
```

Fallback rule:

```python
group_id = provided_source_id if available else stable_hash(long_context_text)
```

---

## Prompt Construction

Use simple task-specific templates.

### Summarization

```text
Summarize the following document.

{document}
```

### Query-focused meeting summarization

```text
Given the meeting transcript below, answer the query with a concise summary.

Query:
{query}

Transcript:
{transcript}
```

### RAG-style QA

```text
Answer the question using the provided documents.

Question:
{question}

Documents:
{documents}
```

### Paper/document QA

```text
Answer the question based on the document.

Question:
{question}

Document:
{document}
```

---

## Sampling Rule with `group_id`

Within one generated workload:

```python
used_group_ids = set()

while len(requests) < target_num_requests:
    bucket = sample_bucket()
    dataset = sample_dataset(bucket)
    record = sample_record(dataset)

    if record.group_id in used_group_ids:
        continue

    requests.append(record)
    used_group_ids.add(record.group_id)
```

This avoids repeated long-context prefixes within the same run.

---

## Optional Relaxed Rule

If the dataset is too small:

```python
MAX_GROUP_REUSE = 1
```

Allow increasing it only when necessary:

```python
MAX_GROUP_REUSE = 2
```

But log reuse counts:

```python
group_reuse_count[group_id] += 1
```

---

## Validation

Before writing the workload file, check:

```python
assert record.bucket in VALID_BUCKETS
assert record.dataset in EXTERNAL_DATASETS[record.bucket]
assert record.sample_id is not None
assert record.group_id is not None
assert record.prompt is not None
```

Also report:

```text
- total requests
- requests per bucket
- requests per dataset
- unique group_id count
- max group_id reuse count
```

---

## Minimal Implementation Order

1. Add normalized adapter interface.
2. Implement adapters for:

   * `gov_report_original`
   * `multi_news_original`
   * `qmsum_original`
   * `meetingbank`
   * `dureader_full`
   * `qasper_full`
3. Add `group_id` extraction.
4. Add cache-aware sampling by `group_id`.
5. Add summary logging.
