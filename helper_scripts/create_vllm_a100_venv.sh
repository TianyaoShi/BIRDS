#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VLLM_A100_VENV_DIR:-$ROOT_DIR/.venv-a100}"
PYTHON_VERSION="${VLLM_A100_PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${VLLM_A100_TORCH_BACKEND:-cu128}"
RECREATE_VENV="${VLLM_A100_RECREATE_VENV:-0}"
UV_BIN="${UV_BIN:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export MAX_JOBS="${MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-80}"
export CC="${VLLM_A100_CC:-gcc}"
export CXX="${VLLM_A100_CXX:-g++}"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "error: uv is not available on PATH or at $HOME/.local/bin/uv" >&2
    exit 1
  fi
fi

mkdir -p "$UV_CACHE_DIR"

if [[ "$RECREATE_VENV" == "1" || ! -x "$VENV_DIR/bin/python" ]]; then
  VENV_ARGS=("$VENV_DIR" --python "$PYTHON_VERSION" --seed --managed-python)
  if [[ "$RECREATE_VENV" == "1" ]]; then
    VENV_ARGS+=(--clear)
  fi
  "$UV_BIN" venv "${VENV_ARGS[@]}"
fi
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  -r "$ROOT_DIR/requirements/profiler.txt"

# Anvil currently reports glibc 2.28, while vLLM 0.19.1's PyPI wheel is
# manylinux_2_31. Build in the target venv so build tools are not hidden in
# short-lived uv isolation directories.
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --torch-backend="$TORCH_BACKEND" \
  "cmake>=3.26.1" \
  ninja \
  "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" \
  "torch==2.10.0" \
  wheel \
  jinja2 \
  numpy

rm -rf \
  "$UV_CACHE_DIR/builds-v0" \
  "$UV_CACHE_DIR/sdists-v9/pypi/vllm" \
  "$UV_CACHE_DIR/wheels-v6/pypi/vllm"

PATH="$VENV_DIR/bin:$PATH" "$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --torch-backend="$TORCH_BACKEND" \
  --no-build-isolation \
  --no-cache \
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
