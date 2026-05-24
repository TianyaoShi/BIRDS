from __future__ import annotations

from pathlib import Path

from energy_profiler import cli


def test_managed_plan_run_cli_is_disabled(capsys) -> None:
    exit_code = cli.main(["run", "--plan", str(Path("experiments/energy/example.yaml"))])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "legacy serial executor" in captured.err
    assert "local_orchestrator.cli energy-run" in captured.err


def test_managed_plan_resume_cli_is_disabled(capsys) -> None:
    exit_code = cli.main(["resume", "--run-root", str(Path("results/energy/example"))])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "legacy serial executor" in captured.err
    assert "local_orchestrator.cli energy-resume" in captured.err
