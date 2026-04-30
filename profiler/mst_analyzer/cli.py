from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_settings
from .reporting import analyze_orchestrator_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mst_analyzer.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--orchestrator-run-root", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--max-rerun-models", type=int, default=7)
    analyze.add_argument("--settings-yaml", type=Path, default=None)
    analyze.set_defaults(handler=_analyze_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _analyze_command(args: argparse.Namespace) -> int:
    settings = None if args.settings_yaml is None else load_settings(args.settings_yaml)
    artifacts = analyze_orchestrator_run(
        orchestrator_run_root=args.orchestrator_run_root,
        output_dir=args.output_dir,
        max_rerun_models=args.max_rerun_models,
        settings=settings,
    )
    print(json.dumps(artifacts.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
