# Dataset Materializer Modular Refactor

## Problem

`profiler/dataset_workload_materializer/materialize.py` grew into a single
module that mixes:

- top-level orchestration
- dataset-specific config parsing
- dataset-specific raw artifact loading
- prompt rendering
- reusable filtering and dedup logic
- shard/workload YAML writing
- report/manifest generation

That shape is manageable for one or two datasets, but it does not scale. Adding
new datasets would keep extending one file and one dispatch chain, making it
harder to reason about ownership and harder to test dataset logic in isolation.

## Goal

Keep the external contract unchanged:

```text
materialization config
  -> offline JSONL shards
  -> ordinary llm_mst_finder workload YAMLs
  -> generic runtime profiling over dataset.type: jsonl
```

But refactor the implementation so that:

- `materialize.py` is a thin coordinator
- shared utilities live in shared modules
- dataset-specific logic lives behind a small handler registry
- adding a new dataset mostly means adding one handler module and registering it

## Target Layout

```text
profiler/dataset_workload_materializer/
  materialize.py
  models.py
  common.py
  outputs.py
  datasets/
    __init__.py
    code.py
    longbench.py
```

## Responsibilities

### `materialize.py`

Owns only:

- loading YAML config
- parsing generic config sections
- resolving tokenizer
- creating a shared context object
- dispatching to a dataset handler
- running shared ordering/sharding/output writing

It should not contain dataset-specific row parsing or prompt rendering.

### `models.py`

Owns small shared dataclasses:

- `MaterializedSample`
- `Counters`
- `FilteringConfig`
- `SamplingConfig`
- `MaterializationContext`
- `DatasetLoadResult`
- `LongBenchProfileSpec`

These types define the boundary between generic orchestration and dataset
handlers.

### `common.py`

Owns reusable helpers that are not tied to one dataset:

- config parsing helpers
- path resolution
- language filter parsing and checks
- dedup settings
- hashing
- generic JSONL discovery

This module should not know LongBench task names or RepoBench prompt shape.

### `outputs.py`

Owns generic post-materialization steps:

- deterministic cache-realistic ordering
- sharding
- shard JSONL writing
- workload YAML generation
- report and manifest builders

This code should operate over `MaterializedSample` only and remain dataset
agnostic.

### `datasets/code.py`

Owns CrossCodeEval and RepoBench dataset-specific logic:

- field aliases
- repo/code prompt rendering
- parquet reading for RepoBench
- row-to-sample conversion
- aggregate mode handling

This keeps the code-completion family together without creating one file per
dataset.

### `datasets/longbench.py`

Owns LongBench dataset-specific logic:

- realistic-NL profile definitions
- excluded task validation
- profile/task selection
- LongBench prompt rendering
- zip/jsonl row loading
- deterministic task-uniform selection

## Handler Contract

Each dataset handler should return a shared `DatasetLoadResult`:

```python
DatasetLoadResult(
    samples=[...],
    task="...",
    prompt_template="...",
    profile="..." | None,
    selected_tasks=[...] | None,
)
```

This keeps generic output/report code independent from the raw dataset format.

## Why This Scales Better

With this split, adding a new dataset should usually require:

1. Add one handler module or extend an existing family module.
2. Register it in `datasets/__init__.py`.
3. Reuse shared output/report/order logic unchanged.

That is the right extension point. The generic pipeline changes only when a new
capability is truly shared, not every time a new dataset appears.

## Non-Goals

- No change to the `llm_mst_finder` runtime contract.
- No dataset-specific logic in request client, trial runner, search, or
  orchestrators.
- No one-file-per-dataset explosion.

## Refactor Sequence

1. Extract shared dataclasses into `models.py`.
2. Extract generic helpers into `common.py`.
3. Extract shared output/report logic into `outputs.py`.
4. Move CrossCodeEval/RepoBench logic into `datasets/code.py`.
5. Move LongBench logic into `datasets/longbench.py`.
6. Reduce `materialize.py` to orchestration and handler dispatch.
7. Re-run focused dataset materializer tests.
