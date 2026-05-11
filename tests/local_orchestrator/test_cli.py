from __future__ import annotations

import json
from pathlib import Path

from local_orchestrator.cli import main


def _write_state(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"experiment_id": "job-planned", "status": "planned", "result_dir": str(run_root / "job-planned")},
        {"experiment_id": "job-running", "status": "running", "result_dir": str(run_root / "job-running")},
        {"experiment_id": "job-succeeded", "status": "succeeded", "result_dir": str(run_root / "job-succeeded")},
        {"experiment_id": "job-failed", "status": "failed", "result_dir": str(run_root / "job-failed")},
    ]
    (run_root / "state.json").write_text(
        json.dumps(
            {
                "run_id": "run-status",
                "manifest_path": str(run_root / "manifest.yaml"),
                "mst_output_root": None,
                "status": "running",
                "created_at": "2026-05-11T00:00:00+00:00",
                "updated_at": "2026-05-11T00:00:00+00:00",
                "jobs": jobs,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_status_reports_progress_for_runroot_alias(tmp_path: Path, capsys) -> None:
    run_root = tmp_path / "run"
    _write_state(run_root)

    assert main(["status", "--runroot", str(run_root)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_root"] == str(run_root.resolve())
    assert payload["summary"]["counts"] == {
        "planned": 1,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
        "skipped": 0,
    }
    assert payload["summary"]["progress"] == {
        "total_jobs": 4,
        "terminal_jobs": 2,
        "remaining_jobs": 2,
        "active_jobs": 1,
        "percent_complete": 50.0,
    }


def test_progress_alias_uses_status_handler(tmp_path: Path, capsys) -> None:
    run_root = tmp_path / "run"
    _write_state(run_root)

    assert main(["progress", "--run-root", str(run_root)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["progress"]["percent_complete"] == 50.0
