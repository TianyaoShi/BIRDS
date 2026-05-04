#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACCOUNT=""
PARTITION="${SLURM_PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
JOB_NAME="${JOB_NAME:-biollm-vllm-a100-env}"

SBATCH_ARGS=(
  --job-name "$JOB_NAME"
  --output "$ROOT_DIR/slurm-%x-%j.out"
  --error "$ROOT_DIR/slurm-%x-%j.err"
  --nodes 1
  --ntasks 1
  --gres gpu:1
  --time "$TIME_LIMIT"
  --chdir "$ROOT_DIR"
)

if [[ -n "$ACCOUNT" ]]; then
  SBATCH_ARGS+=(--account "$ACCOUNT")
fi

if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition "$PARTITION")
fi

sbatch "${SBATCH_ARGS[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/profile ]]; then
  # Load site module init if the cluster provides it here.
  set +u
  source /etc/profile
  set -u
fi

echo "job_id=\${SLURM_JOB_ID:-unknown}"
echo "host=\$(hostname)"
echo "pwd=\$(pwd)"

# Edit these if your site requires extra toolchain modules for uv, Python, or
# source builds.
if command -v module >/dev/null 2>&1; then
  module load modtree/gpu
  module load gcc/11.2.0
  module load cuda/12.8.0
fi

"$ROOT_DIR/helper_scripts/create_vllm_a100_venv.sh"
EOF
