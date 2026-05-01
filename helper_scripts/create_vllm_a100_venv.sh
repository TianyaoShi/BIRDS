#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VLLM_A100_VENV_DIR:-$ROOT_DIR/.venv-a100}"
PYTHON_VERSION="${VLLM_A100_PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${VLLM_A100_TORCH_BACKEND:-cu128}"
UV_BIN="${UV_BIN:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "error: uv is not available on PATH or at $HOME/.local/bin/uv" >&2
    exit 1
  fi
fi

mkdir -p "$UV_CACHE_DIR"

"$UV_BIN" venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed --managed-python --clear
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --torch-backend="$TORCH_BACKEND" \
  -r "$ROOT_DIR/requirements/profiler.txt" \
  -r "$ROOT_DIR/requirements/vllm-h100.txt"

"$VENV_DIR/bin/python" - <<'PY'
import torch
import vllm

print(f"vllm={vllm.__version__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

cat <<EOF

Created A100 vLLM environment:
  $VENV_DIR

Use it with:
  source "$ROOT_DIR/helper_scripts/activate_vllm_a100.sh"
EOF
