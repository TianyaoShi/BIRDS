# Rerun Manifest Review

Use this checklist when reviewing `suggested_rerun_manifest.yaml` files emitted by
`mst_analyzer`.

## Severity Rules

- Treat `missing_confirmed_mst_rate` as blocking for downstream energy profiling.
  A run without a final confirmed MST cannot be used as an energy-profiling target,
  even if the analyzer report labels it low severity.
- Treat `search_rate_cap_reached` as blocking for MST finality. The search only
  proved that the configured cap was too low; it did not bound the MST.
- Treat comparison/outlier findings as review targets. They justify reruns when
  they are paired with conflicted traces, surprising model ordering, or unstable
  confirmation evidence.

## Manifest Editing Rules

- Keep reruns targeted to anomalous model/workload/serving-config pairs. Avoid
  expanding one anomalous workload into every workload in the original matrix
  unless the evidence is ambiguous.
- Include explicit controls only when they explain the anomaly, such as a same-size
  peer or nearest larger model used in the report comparator.
- For rate-cap findings, do not simply double the failed cap for high-throughput or
  higher-TP runs. Use a ceiling that is unlikely to become the next bottleneck,
  such as 40, 60, or 80 rps depending on model size, TP degree, and workload.
- Increase the search budget when raising rate caps. As a default, use at least
  `max_binary_steps: 10` and `max_bracket_trials: 18`; use 11/20 for very wide
  caps or TP4 runs.
- Preserve launch fixes already learned from previous runs. In particular, do not
  reintroduce expandable CUDA allocator segments for local gpt-oss-120B TP2
  manifests, and keep known-good gpt-oss-120B limits such as `max_model_len: 16384`,
  `max_num_seqs: 128`, and `max_num_batched_tokens: 1024` where applicable.
- Prefer portable workload paths. Cluster-local large datasets should go through
  stable symlink locations such as `data/local/sharegpt/...`, not through test
  fixture paths.

## Validation

- Validate the edited manifest with `local_orchestrator.manifest.load_manifest`.
- Expand it with `local_orchestrator.matrix.expand_manifest` and inspect the job
  count before launch.
- Confirm the expanded jobs cover all blocking findings and do not contain obvious
  accidental Cartesian products.
