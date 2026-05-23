#!/bin/bash
#SBATCH -J materialize-repobench-chat
#SBATCH -p smallgpu
#SBATCH -A yiding
#SBATCH --gres=gpu:1
#SBATCH -c 64
#SBATCH -t 04:00:00
#SBATCH -o /scratch/gautschi/shi676/BioLLM/results/quality/materialization_logs/repobench-code-chat-%j.out
#SBATCH -e /scratch/gautschi/shi676/BioLLM/results/quality/materialization_logs/repobench-code-chat-%j.err

set -euo pipefail

REPO_ROOT=/scratch/gautschi/shi676/BioLLM
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/results/quality/materialization_logs"

source "$REPO_ROOT/helper_scripts/activate_vllm_h100.sh"

export PYTHONPATH="$REPO_ROOT/profiler:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

python scripts/quality/build_repobench_chat_workload.py \
  --source experiments/code_workloads/repobench_python_java_aggregate_cache_realistic \
  --output experiments/code_workloads/repobench_python_java_aggregate_code_chat_completion \
  --tokenizer Qwen/Qwen3-8B
