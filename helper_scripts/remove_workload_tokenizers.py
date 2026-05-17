#!/usr/bin/env python3
"""Remove top-level tokenizer keys from workload YAML configs.

The MST workload loader can now use the serving instance tokenizer by default
for validation/stats accounting. This helper strips the older hardcoded
top-level ``tokenizer`` field from generated workload configs while preserving
the rest of the YAML text as much as possible.

By default this is a dry run. Pass --apply to rewrite matching files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_ROOTS = (
    Path("experiments/workloads"),
    Path("experiments/code_workloads"),
    Path("experiments/longbench_workloads"),
    Path("experiments/reasoning_workloads"),
)


def main() -> int:
    args = _parse_args()
    files = _discover_yaml_files(args.path)
    changed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for path in files:
        result = _remove_top_level_tokenizer(path)
        if result.status == "changed":
            changed.append({"path": str(path), "removed": result.removed_preview})
            if args.apply:
                path.write_text(result.updated_text, encoding="utf-8")
        elif result.status == "skipped":
            skipped.append({"path": str(path), "reason": result.reason})

    summary = {
        "apply": args.apply,
        "scanned_count": len(files),
        "changed_count": len(changed),
        "skipped_count": len(skipped),
        "changed": changed,
        "skipped": skipped[: args.max_skipped],
        "skipped_truncated": max(0, len(skipped) - args.max_skipped),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help=(
            "YAML file or directory to scan. Defaults to experiments/workloads, "
            "experiments/code_workloads, experiments/longbench_workloads, and "
            "experiments/reasoning_workloads."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files. Without this flag the script only reports what would change.",
    )
    parser.add_argument(
        "--max-skipped",
        type=int,
        default=20,
        help="Maximum skipped-file details to include in stdout (default: 20).",
    )
    return parser.parse_args()


def _discover_yaml_files(paths: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix in {".yaml", ".yml"}:
            discovered.add(path)
        elif path.is_dir():
            for suffix in ("*.yaml", "*.yml"):
                discovered.update(candidate for candidate in path.rglob(suffix) if candidate.is_file())
    return sorted(discovered)


class RemovalResult:
    def __init__(
        self,
        status: str,
        *,
        updated_text: str = "",
        removed_preview: str = "",
        reason: str = "",
    ) -> None:
        self.status = status
        self.updated_text = updated_text
        self.removed_preview = removed_preview
        self.reason = reason


def _remove_top_level_tokenizer(path: Path) -> RemovalResult:
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        return RemovalResult("skipped", reason=f"read failed: {exc}")

    try:
        payload = yaml.safe_load(original)
    except yaml.YAMLError as exc:
        return RemovalResult("skipped", reason=f"YAML parse failed: {exc}")

    if not isinstance(payload, Mapping) or "tokenizer" not in payload:
        return RemovalResult("unchanged")

    updated, removed = _remove_top_level_key_block(original, "tokenizer")
    if updated == original:
        return RemovalResult("skipped", reason="top-level tokenizer exists in YAML payload but was not found in text")

    try:
        updated_payload = yaml.safe_load(updated)
    except yaml.YAMLError as exc:
        return RemovalResult("skipped", reason=f"updated YAML would not parse: {exc}")

    if not isinstance(updated_payload, Mapping) or "tokenizer" in updated_payload:
        return RemovalResult("skipped", reason="updated YAML still contains a top-level tokenizer key")

    return RemovalResult("changed", updated_text=updated, removed_preview="".join(removed).strip())


def _remove_top_level_key_block(text: str, key: str) -> tuple[str, list[str]]:
    lines = text.splitlines(keepends=True)
    start: int | None = None
    for index, line in enumerate(lines):
        if _is_top_level_key_line(line, key):
            start = index
            break
    if start is None:
        return text, []

    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.startswith((" ", "\t")):
            break
        end += 1

    removed = lines[start:end]
    updated = "".join(lines[:start] + lines[end:])
    return updated, removed


def _is_top_level_key_line(line: str, key: str) -> bool:
    if line.startswith((" ", "\t")):
        return False
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return stripped == f"{key}:" or stripped.startswith(f"{key}: ")


if __name__ == "__main__":
    raise SystemExit(main())
