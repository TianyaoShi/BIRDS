from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_settings
from .plotting import plot_model_size_vs_mst_from_json, plot_model_size_vs_mst_from_orchestrator_run
from .reporting import analyze_orchestrator_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mst_analyzer.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--orchestrator-run-root", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--max-rerun-models", type=int, default=7)
    analyze.add_argument("--emit-rerun-manifest", action="store_true")
    analyze.add_argument("--settings-yaml", type=Path, default=None)
    analyze.set_defaults(handler=_analyze_command)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--mst-rows-json", type=Path, default=None)
    plot.add_argument("--orchestrator-run-root", type=Path, default=None)
    plot.add_argument("--output-path", type=Path, required=True)
    plot.add_argument("--title", type=str, default="Model Size vs MST")
    plot.add_argument("--x-scale", type=str, default="log")
    plot.add_argument("--no-annotations", action="store_true")
    plot.set_defaults(handler=_plot_command)
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
        emit_rerun_manifest=args.emit_rerun_manifest,
        settings=settings,
    )
    print(json.dumps(artifacts.to_dict(), sort_keys=True))
    return 0


def _plot_command(args: argparse.Namespace) -> int:
    if (args.mst_rows_json is None) == (args.orchestrator_run_root is None):
        raise SystemExit("provide exactly one of --mst-rows-json or --orchestrator-run-root")
    if args.mst_rows_json is not None:
        output_path = plot_model_size_vs_mst_from_json(
            mst_rows_json_path=args.mst_rows_json,
            output_path=args.output_path,
            title=args.title,
            x_scale=args.x_scale,
            annotate=not args.no_annotations,
        )
    else:
        output_path = plot_model_size_vs_mst_from_orchestrator_run(
            orchestrator_run_root=args.orchestrator_run_root,
            output_path=args.output_path,
            title=args.title,
            x_scale=args.x_scale,
            annotate=not args.no_annotations,
        )
    print(json.dumps({"output_path": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
