Status: Approved for implementation on 2026-04-27.


## Plan: Local vLLM MST Orchestrator V1

Build a local upper orchestrator that runs MST finder experiments across a model x workload matrix, while managing local GPU capacity and vLLM server lifecycle outside MST finder. The recommended approach is a YAML-driven scheduler that validates manifest inputs, allocates up to 3 local GPUs, launches/reuses vLLM instances when signatures match, invokes llm_mst_finder search as an external command, and records resumable orchestration state.

**Steps**
1. Phase 1 - Contracts and plan docs: create codex_plan/local_orchestrator overview and implementation docs that codify scope boundaries, interfaces, and lifecycle states. This includes fixed-config MST orchestration only, single-GPU focus, max 3 GPUs active, and no multi-GPU scheduling in V1.
2. Phase 1 - Manifest schema and validation: add strict YAML schema parsing in profiler/local_orchestrator for run-level defaults and experiment-level overrides. Support both structured launch fields and raw launch template entries. Fail fast on invalid workload paths, duplicate IDs, invalid SLO fields, missing model, or incompatible launch fields.
3. Phase 1 - Matrix expansion and identity: implement deterministic expansion from manifest to concrete experiment jobs with stable experiment IDs, result directory slugs, and server signature keys. Reuse contract result layout pattern results/mst/{model_slug}/{dataset_slug}/{server_config_slug}. Depends on step 2.
4. Phase 2 - Resource managers: implement GPU lease manager and port allocator. GPU manager enforces allowed GPU IDs and max_active_gpus=3, tracks free/busy slots, and supports only gpu_count=1 jobs in V1. Port allocator reserves unique base/metrics ports and releases on teardown. Depends on step 3. Parallel with step 5.
5. Phase 2 - vLLM lifecycle manager: add start, readiness probe, reuse, retire, and hard-kill fallback for vLLM subprocesses. Reuse process only when server signature matches (model plus serving flags plus GPU assignment); otherwise restart. Health/readiness checks should be explicit and bounded. Depends on step 3. Parallel with step 4.
6. Phase 2 - MST invocation adapter: add an adapter that executes llm_mst_finder.cli search via shared environment command boundaries, passes required endpoint/model/workload inputs and server metadata fields, and captures stdout/stderr logs per experiment. Parse search_trace.json and final report outputs as orchestration evidence. Depends on step 3. Parallel with steps 4 and 5.
7. Phase 3 - Scheduler engine: orchestrate end-to-end experiment execution with retries and continuation policy. For each job: acquire GPU and port, ensure server ready, run MST search, classify terminal status, and continue to next job on failure after bounded retries. Persist per-job state transitions for resume support. Depends on steps 4, 5, and 6.
8. Phase 3 - Resume and state store: persist orchestrator state and event logs under run output root (planned/running/succeeded/failed/skipped), with restart-safe reconciliation against existing result directories and search traces. Depends on step 7.
9. Phase 3 - Run summary outputs: generate orchestrator-level summary JSON and Markdown listing experiment matrix coverage, rates, termination reasons, retry counts, failure reasons, and pointers to each MST result directory. Depends on step 8.
10. Phase 4 - CLI and UX: add orchestrator CLI entrypoint supporting dry-run (no process launch), run, resume, and status subcommands. Dry-run prints expanded matrix and planned resource usage. Depends on steps 2 and 3; fully functional run/resume depends on step 8.
11. Phase 4 - Testing: add offline unit tests for schema validation, matrix expansion, slugging, launch command rendering, retry policy, state transitions, and result parsing. Add subprocess and readiness probes as mocked components. Depends on steps 2 through 10.
12. Phase 4 - Manual validation protocol: run a small two-model single-GPU sequence with distinct ports and confirm result artifacts, resume behavior, and server reuse semantics for identical signatures.

**Relevant files**
- /local/scratch/a/shi676/arr26/codex_plan/llm_mst_finder/Upper_orchestrator_contract.md - authoritative responsibilities split and failure semantics to preserve.
- /local/scratch/a/shi676/arr26/profiler/llm_mst_finder/cli.py - canonical MST search command inputs and metadata arguments.
- /local/scratch/a/shi676/arr26/profiler/llm_mst_finder/search.py - search termination semantics and invalid-trial behavior to consume.
- /local/scratch/a/shi676/arr26/profiler/llm_mst_finder/records.py - output schema fields referenced by orchestration summary logic.
- /local/scratch/a/shi676/arr26/results/mst/gemma-4-E4B-it/run_until_waiting_queue.py - concrete command and metadata handling example for live runs.
- /local/scratch/a/shi676/arr26/profiler/local_orchestrator - new package root for implementation.
- /local/scratch/a/shi676/arr26/codex_plan/local_orchestrator - new planning docs folder for upper-layer design and runbook.
- /local/scratch/a/shi676/arr26/tests/local_orchestrator - new test suite for orchestrator modules.

**Verification**
1. Unit tests: run orchestrator-only tests with mocked subprocess and mocked HTTP probes; verify deterministic matrix expansion and retry state transitions.
2. Contract checks: validate every generated MST command includes required fields from contract and disallows unsupported SLO dimensions.
3. Dry-run checks: confirm no subprocess launch occurs while planned resources, ports, and output paths are correctly rendered.
4. Resume checks: interrupt a run mid-way, restart with resume, and verify completed experiments are not rerun unless force is enabled.
5. End-to-end smoke: execute at least one real single-GPU model/workload search and verify search_trace.json plus final_report artifacts are indexed in orchestrator summary.

**Decisions**
- V1 scope is fixed server config MST per model x workload matrix; no server-config sweep in this phase.
- Input format is YAML manifest.
- Launch interface supports both structured launch fields and raw template commands.
- Failure policy is bounded retries for startup/search, then continue and mark failed.
- GPU policy is to keep one GPU spare and use at most 3 GPUs concurrently.
- Scheduling focus is single-GPU jobs first; multi-GPU scheduling is explicitly deferred.
- Server topology is hybrid: reuse when signature matches, restart on model/config change.

**Further Considerations**
1. Recommendation: define server signature hash as model plus tokenizer mode plus tensor_parallel_size plus max_num_seqs plus max_num_batched_tokens plus dtype plus quantization plus GPU binding plus endpoint type; this prevents unsafe cross-run reuse.
2. Recommendation: include a per-experiment orchestrator timeout separate from MST timeout to prevent orphaned subprocesses.
3. Recommendation: reserve a dedicated run root such as results/orchestrator/{run_id} for state files and summaries, while preserving MST outputs under results/mst/...
