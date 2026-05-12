#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OLD_ROOT = Path("/anvil/projects/x-cis250584/BioLLM")
DEFAULT_NEW_ROOT = Path("/anvil/scratch/x-tshi1/BioLLM")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a copied Slurm orchestrator run root from one BioLLM "
            "filesystem prefix to another so collect/resume no longer touch "
            "the old filesystem."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print files that would be rewritten without modifying them",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    old_text = str(args.old_root.resolve())
    new_text = str(args.new_root.resolve())

    if old_text == new_text:
        raise SystemExit("old-root and new-root resolve to the same path")
    if not run_root.is_dir():
        raise SystemExit(f"run root does not exist: {run_root}")
    if not str(run_root).startswith(new_text):
        raise SystemExit(
            f"run root should be under new-root to avoid rewriting the source run: {run_root}"
        )

    suffixes = {".json", ".md", ".sh"}
    changed: list[Path] = []
    for path in run_root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if old_text not in text:
            continue
        changed.append(path)
        if not args.dry_run:
            path.write_text(text.replace(old_text, new_text), encoding="utf-8")

    action = "would rewrite" if args.dry_run else "rewrote"
    print(f"{action} {len(changed)} files")
    for path in changed:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
