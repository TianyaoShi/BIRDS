#!/usr/bin/env bash
set -euo pipefail

# Export unique model repo IDs from local HF cache to a plain text file.
# One repo id per line, e.g.:
#   meta-llama/Llama-3.1-8B-Instruct
#
# Usage:
#   ./helper_scripts/export_hf_cached_models.sh [output_file]
#
# Notes:
# - Activates .venv first as requested.
# - Uses uv-managed hf CLI.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="$HOME/.venv/bin/activate"
OUT_FILE="${1:-$ROOT_DIR/hf_cached_models.txt}"
TMP_FILE="$(mktemp)"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "error: missing virtualenv activate script at $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not available in PATH after activating .venv" >&2
  exit 1
fi

# Preferred path: structured output.
if uv run hf cache ls --format json >"$TMP_FILE" 2>/dev/null; then
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$TMP_FILE" "$OUT_FILE" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

repo_ids = set()
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict):
            rid = item.get("repo_id")
            if isinstance(rid, str) and rid.count("/") >= 1:
                repo_ids.add(rid)

with open(dst, "w", encoding="utf-8") as f:
    for rid in sorted(repo_ids):
        f.write(rid + "\n")
print(f"wrote {len(repo_ids)} model repo ids to {dst}")
PY
    rm -f "$TMP_FILE"
    exit 0
  fi
fi

# Fallback path: parse plain text output.
if uv run hf cache ls >"$TMP_FILE" 2>/dev/null; then
  awk 'NF>0 {print $1}' "$TMP_FILE" | grep '/' | sort -u > "$OUT_FILE"
  count=$(wc -l < "$OUT_FILE" | tr -d ' ')
  echo "wrote $count model repo ids to $OUT_FILE"
  rm -f "$TMP_FILE"
  exit 0
fi

rm -f "$TMP_FILE"
echo "error: failed to run 'uv run hf cache ls'. Is huggingface_hub CLI installed in .venv?" >&2
exit 1
