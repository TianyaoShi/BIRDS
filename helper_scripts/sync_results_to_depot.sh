#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/scratch/gautschi/shi676/BioLLM/results}"
DST="${2:-/depot/yiding/data/BioLLM-results/results}"
LOCK="${DST}.sync.lock"

if [[ ! -d "$SRC" ]]; then
  echo "Source directory does not exist: $SRC" >&2
  exit 1
fi

mkdir -p "$(dirname "$DST")"

if [[ -L "$DST" ]]; then
  echo "Destination is a symlink; remove it first:"
  echo "  rm '$DST'"
  echo "  mkdir -p '$DST'"
  exit 2
fi

mkdir -p "$DST"

(
  flock -n 9 || {
    echo "Another sync is already running for $DST" >&2
    exit 3
  }

  rsync -a \
    --delete-delay \
    --partial \
    --human-readable \
    --info=stats2,progress2 \
    --chmod=D755,F644 \
    "$SRC"/ "$DST"/
) 9>"$LOCK"
