#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  helper_scripts/submit_quality_judge_batches.sh --split-manifest PATH [options]

Options:
  --split-manifest PATH    Required split_manifest.json from split-judge-batch-by-candidate.
  --api-key-file PATH      OpenAI API key file. Default: /home/shi676/openai_llm_as_a_judge_key
  --ledger PATH            Submission ledger. Default: <split-dir>/submission_ledger.json
  --log PATH               Nohup log. Default: <split-dir>/submission_loop.nohup.log
  --pid-file PATH          PID file. Default: <split-dir>/submission_loop.pid
  --poll-interval-s SEC    Poll interval. Default: 60
  --completion-window VAL  OpenAI batch completion window. Default: 24h
  --foreground             Run in the foreground instead of nohup/background.
  -h, --help               Show this help.

The submitter processes split parts sequentially. With --wait-for-completion it
submits one OpenAI batch, waits for terminal status, retrieves output/error files,
updates the ledger, then advances to the next unfinished part.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
split_manifest=""
api_key_file="/home/shi676/openai_llm_as_a_judge_key"
ledger=""
log_path=""
pid_file=""
poll_interval_s="60"
completion_window="24h"
foreground=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-manifest)
      split_manifest="${2:?missing value for --split-manifest}"
      shift 2
      ;;
    --api-key-file)
      api_key_file="${2:?missing value for --api-key-file}"
      shift 2
      ;;
    --ledger)
      ledger="${2:?missing value for --ledger}"
      shift 2
      ;;
    --log)
      log_path="${2:?missing value for --log}"
      shift 2
      ;;
    --pid-file)
      pid_file="${2:?missing value for --pid-file}"
      shift 2
      ;;
    --poll-interval-s)
      poll_interval_s="${2:?missing value for --poll-interval-s}"
      shift 2
      ;;
    --completion-window)
      completion_window="${2:?missing value for --completion-window}"
      shift 2
      ;;
    --foreground)
      foreground=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$split_manifest" ]]; then
  echo "--split-manifest is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -f "$split_manifest" ]]; then
  echo "split manifest does not exist: $split_manifest" >&2
  exit 1
fi
if [[ ! -f "$api_key_file" ]]; then
  echo "API key file does not exist: $api_key_file" >&2
  exit 1
fi

split_dir="$(cd "$(dirname "$split_manifest")" && pwd)"
ledger="${ledger:-$split_dir/submission_ledger.json}"
log_path="${log_path:-$split_dir/submission_loop.nohup.log}"
pid_file="${pid_file:-$split_dir/submission_loop.pid}"

mkdir -p "$(dirname "$ledger")" "$(dirname "$log_path")" "$(dirname "$pid_file")"

if [[ -f "$pid_file" ]]; then
  existing_pid="$(tr -d '[:space:]' < "$pid_file" || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "submission loop already appears to be running: pid=$existing_pid pid_file=$pid_file" >&2
    exit 1
  fi
fi

cmd=(
  "$repo_root/.venv-h100/bin/python"
  -m output_quality_profiler.cli
  submit-openai-batches
  --split-manifest "$split_manifest"
  --api-key-file "$api_key_file"
  --ledger "$ledger"
  --wait-for-completion
  --poll-interval-s "$poll_interval_s"
  --completion-window "$completion_window"
)

cd "$repo_root"
export PYTHONPATH="profiler:."

if [[ "$foreground" -eq 1 ]]; then
  exec "${cmd[@]}"
fi

nohup "${cmd[@]}" > "$log_path" 2>&1 &
pid="$!"
printf '%s\n' "$pid" > "$pid_file"

echo "started quality judge batch submission loop"
echo "pid: $pid"
echo "pid_file: $pid_file"
echo "log: $log_path"
echo "ledger: $ledger"
