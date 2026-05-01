#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VLLM_A100_VENV_DIR:-$ROOT_DIR/.venv-a100}"
CACHE_ROOT="${VLLM_A100_CACHE_ROOT:-$ROOT_DIR/.cache/a100}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "error: missing A100 virtualenv at $VENV_DIR" >&2
  echo "run: $ROOT_DIR/helper_scripts/create_vllm_a100_venv.sh" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$CACHE_ROOT"/{vllm,triton,torchinductor,flashinfer,jit,flashinfer-cubins}

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export FLASHINFER_CACHE_DIR="$CACHE_ROOT/flashinfer"
export FLASHINFER_JIT_DIR="$CACHE_ROOT/jit"
export FLASHINFER_CUBIN_DIR="$CACHE_ROOT/flashinfer-cubins"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-80}"

if [[ -x "$HOME/.local/bin/uv" ]]; then
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
  esac
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
