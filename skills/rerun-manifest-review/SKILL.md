---
name: rerun-manifest-review
description: Review and edit mst_analyzer suggested_rerun_manifest.yaml files for BioLLM MST reruns. Use when checking rerun manifests for missing blocking anomalies, incorrect rate caps, accidental Cartesian expansion, unsupported models, stale workload paths, tensor-parallel mixing, or launch overrides before submitting local_orchestrator or slurm_orchestrator reruns.
---

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
- Keep tensor-parallel variants separate. MST rates from `tp1`, `tp2`, `tp4`, and
  `tp8` are distinct serving configurations and must not be bucketed together.

## Manifest Editing Rules

- Keep reruns targeted to anomalous model/workload/serving-config pairs. Avoid
  expanding one anomalous workload into every workload in the original matrix
  unless the evidence is ambiguous.
- Include explicit controls only when they explain the anomaly, such as a same-size
  peer or nearest larger model used in the report comparator.
- Remove intentionally excluded model families before launch, such as `<3B`
  H100 experiments or gpt-oss entries for code workloads that require the harmony
  chat template and cannot use `/v1/completions`.
- For rate-cap findings, do not simply double the failed cap for high-throughput or
  higher-TP runs. Use a ceiling that is unlikely to become the next bottleneck,
  such as 40, 60, or 80 rps depending on model size, TP degree, sparse MoE
  activation, and workload.
- For max-request-rate-limited reruns, jump-start open-loop search conservatively.
  Start below the prior cap when large models warmed poorly; small sparse MoE
  models can start closer to the previous cap.
- Increase the search budget when raising rate caps. As a default, use at least
  `max_binary_steps: 10` and `max_bracket_trials: 18`; use 11/20 for very wide
  caps or TP4/TP8 runs.
- Preserve launch fixes already learned from previous runs. In particular, do not
  reintroduce expandable CUDA allocator segments for gpt-oss-120B TP4 manifests;
  allocator segment env vars should only be used for the specific TP1/TP2 cases
  that need them.
- Prefer portable workload paths. Cluster-local large datasets should go through
  stable symlink locations such as `data/local/sharegpt/...`, not through test
  fixture paths or analysis-output-relative paths.
- Make output roots explicit and anchored where the user expects them. Avoid
  writing rerun orchestrator roots under `results/analysis/results/...`.

## Validation

- Validate the edited manifest with `local_orchestrator.manifest.load_manifest`.
- Expand it with `local_orchestrator.matrix.expand_manifest` and inspect the job
  count before launch.
- Confirm the expanded jobs cover all blocking findings and do not contain obvious
  accidental Cartesian products.
- For resume-based reruns, check whether `state.json` and group plans need matching
  updates; manifest edits alone may not affect an existing run root.
