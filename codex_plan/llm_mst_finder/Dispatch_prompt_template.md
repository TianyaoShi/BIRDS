# Agent Dispatch Prompt Template

Use this template for every new agent dispatch and every continuation of an existing agent.

```text
You are Agent {N} for the `llm_mst_finder` implementation.

Before doing anything else, re-read these files from disk in this order:
1. `codex_plan/llm_mst_finder/Rules.md`
2. `codex_plan/llm_mst_finder/Implementation_order.md`
3. `codex_plan/llm_mst_finder/OVERVIEW.md`
4. `codex_plan/llm_mst_finder/Agent{N}.md`

Treat `Rules.md` as live and authoritative. If any rule changed since your last session, follow the newest rule and call out the adjustment in your final handoff. Do not rely on memory of earlier prompts when it conflicts with these files.

Scope:
{SCOPE}

Hard requirements:
- Keep edits scoped to your ownership and do not revert other agents' work.
- Use `/local/scratch/a/shi676/.venv/bin/python`.
- Run package commands with `PYTHONPATH=/local/scratch/a/shi676/arr26/profiler`.
- Preserve fail-fast behavior: explicit validation, explicit exceptions, no broad defensive fallbacks, no silent throttling/clamping, no silent skipping/truncation.
- For real JSONL/ShareGPT/live profiling workloads, context validation must use a serving/model-compatible tokenizer through `context_policy`. `whitespace` is only allowed in synthetic/offline tests or clearly labeled toy examples.
- If context validation cannot determine `max_model_len` or a compatible tokenizer, fail before trial dispatch.
- Do not add server restart, health-check, or auto-recovery logic.

Before finishing, run the narrow tests for your slice and, when feasible:
`PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m compileall profiler/llm_mst_finder tests/llm_mst_finder`
`PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m pytest tests/llm_mst_finder -q`

Final handoff must include:
- files changed
- tests/commands run
- Rules.md items that affected your work
- blockers or integration assumptions
```

## Current Scope Snippets

```text
Agent 1 scope: CLI/trial integration, request client, load generators, TrialRunner, and pre-trial context validation wiring.
```

```text
Agent 2 scope: workload YAML, deterministic sampling, model_context.py, and workload/model context compatibility.
```

```text
Agent 5 scope: stability classification only after trial validity is established; do not classify invalid workload samples as overload.
```

```text
Agent 10 scope: offline tests, CLI import checks, fail-fast assertions, and synthetic-only use of toy tokenizers such as whitespace.
```
