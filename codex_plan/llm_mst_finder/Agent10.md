# Agent 10 Retired

Do not dispatch Agent 10.

The original Agent 10 scope was centralized test hardening. That is no longer needed as a separate implementation role. Each implementation agent must run focused unit tests for its own slice, and the lead agent performs integrated review, live smoke checks when available, and final full-suite validation.

Default checks remain:

```bash
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m compileall profiler/llm_mst_finder tests/llm_mst_finder
PYTHONPATH=/local/scratch/a/shi676/arr26/profiler /local/scratch/a/shi676/.venv/bin/python -m pytest tests/llm_mst_finder -q
```

Live tests must stay behind explicit opt-in environment variables or markers.
