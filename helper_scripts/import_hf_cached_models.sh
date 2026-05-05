#!/usr/bin/env bash
set -euo pipefail

# Import model caches on another machine by replaying hf downloads.
#
# Usage:
#   ./helper_scripts/import_hf_cached_models.sh /path/to/hf_cached_models.txt
#
# Optional env vars:
#   HF_IMPORT_FORCE=1         # pass --force-download to hf download
#   HF_IMPORT_REVISION=main   # add --revision <value>

# ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="/anvil/projects/x-cis250584/BioLLM"
VENV_ACTIVATE="$ROOT_DIR/.venv-a100/bin/activate"
LIST_FILE="${1:-}"

if [[ -z "$LIST_FILE" ]]; then
  echo "usage: $0 /path/to/hf_cached_models.txt" >&2
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "error: list file does not exist: $LIST_FILE" >&2
  exit 1
fi

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

extra_args=()
if [[ "${HF_IMPORT_FORCE:-0}" == "1" ]]; then
  extra_args+=("--force-download")
fi
if [[ -n "${HF_IMPORT_REVISION:-}" ]]; then
  extra_args+=("--revision" "$HF_IMPORT_REVISION")
fi

total=0
ok=0
failed=0

while IFS= read -r repo_id; do
  # Skip empty/comment lines.
  [[ -z "$repo_id" ]] && continue
  [[ "$repo_id" =~ ^# ]] && continue

  total=$((total + 1))
  echo "[$total] downloading $repo_id"

  if uv run hf download "$repo_id" "${extra_args[@]}" >/dev/null; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1))
    echo "warning: failed to download $repo_id" >&2
  fi
done < "$LIST_FILE"

echo "done: total=$total succeeded=$ok failed=$failed"
if [[ "$failed" -gt 0 ]]; then
  exit 2
fi
