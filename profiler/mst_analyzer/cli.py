from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .reporting import analyze_orchestrator_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mst_analyzer.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--orchestrator-run-root", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--max-rerun-models", type=int, default=7)
    analyze.set_defaults(handler=_analyze_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _analyze_command(args: argparse.Namespace) -> int:
    artifacts = analyze_orchestrator_run(
        orchestrator_run_root=args.orchestrator_run_root,
        output_dir=args.output_dir,
        max_rerun_models=args.max_rerun_models,
    )
    print(json.dumps(artifacts.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
