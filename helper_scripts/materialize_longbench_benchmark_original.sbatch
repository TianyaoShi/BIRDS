#!/bin/bash
#SBATCH -J lb-bench-mat
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=/scratch/gautschi/shi676/BioLLM/experiments/longbench_workloads/benchmark_original/slurm_logs/%x-%A_%a.out
#SBATCH --error=/scratch/gautschi/shi676/BioLLM/experiments/longbench_workloads/benchmark_original/slurm_logs/%x-%A_%a.err
#SBATCH -A yiding
#SBATCH -p smallgpu
#SBATCH --qos=normal
#SBATCH -t 00:30:00

set -euo pipefail

REPO_ROOT=/scratch/gautschi/shi676/BioLLM
PYTHON_BIN=/scratch/gautschi/shi676/BioLLM/.venv-h100/bin/python

CONFIGS=(
  "experiments/longbench_workloads/benchmark_original/longbench_long_output_summarization_original_qwen3_8b.yaml"
  "experiments/longbench_workloads/benchmark_original/longbench_medium_output_summarization_original_qwen3_8b.yaml"
  "experiments/longbench_workloads/benchmark_original/longbench_medium_answer_rag_qa_original_qwen3_8b.yaml"
  "experiments/longbench_workloads/benchmark_original/longbench_short_answer_document_qa_original_qwen3_8b.yaml"
)

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$TASK_INDEX]}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/profiler:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "materializing $CONFIG"
"$PYTHON_BIN" -m dataset_workload_materializer.cli prepare --config "$CONFIG"
