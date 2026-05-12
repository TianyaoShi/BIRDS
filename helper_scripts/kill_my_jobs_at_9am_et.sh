#!/usr/bin/env bash
set -euo pipefail

TARGET_TIME="09:00"
TARGET_TZ="America/New_York"
GRACE_SECONDS=30
DRY_RUN=0
RELATED_PROCS=0
LOG_DIR="results/process_cleanup"

usage() {
  cat <<'EOF'
Usage:
  helper_scripts/kill_my_jobs_at_9am_et.sh [options]

Wait until 9:00 AM Eastern time, then terminate this user's GPU processes.
By default it only kills processes owned by the current user that appear in
nvidia-smi. Use --related-procs to also clear remaining MST finder /
local-orchestrator processes owned by this user on the node.

Options:
  --time HH:MM          Target time in Eastern time. Default: 09:00
  --timezone TZ        IANA timezone. Default: America/New_York
  --grace SECONDS      Seconds between SIGTERM and SIGKILL. Default: 30
  --related-procs      After GPU cleanup, terminate remaining MST/orchestrator
                       processes owned by this user
  --dry-run            Print what would be killed without killing anything
  --run-now            Do not wait; execute cleanup immediately
  -h, --help           Show this help

Examples:
  nohup helper_scripts/kill_my_jobs_at_9am_et.sh > cleanup_9am.log 2>&1 &
  nohup helper_scripts/kill_my_jobs_at_9am_et.sh --related-procs > cleanup_9am.log 2>&1 &
EOF
}

RUN_NOW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time)
      TARGET_TIME="${2:?--time requires HH:MM}"
      shift 2
      ;;
    --timezone)
      TARGET_TZ="${2:?--timezone requires a timezone}"
      shift 2
      ;;
    --grace)
      GRACE_SECONDS="${2:?--grace requires seconds}"
      shift 2
      ;;
    --related-procs)
      RELATED_PROCS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run-now)
      RUN_NOW=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$GRACE_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "--grace must be a non-negative integer" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/cleanup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "cleanup helper started at $(date)"
echo "user=$USER host=$(hostname) dry_run=$DRY_RUN related_procs=$RELATED_PROCS"
echo "log=$LOG_PATH"

if [[ "$RUN_NOW" -eq 0 ]]; then
  now_epoch="$(date +%s)"
  target_epoch="$(TZ="$TARGET_TZ" date -d "today $TARGET_TIME" +%s)"
  if [[ "$target_epoch" -le "$now_epoch" ]]; then
    target_epoch="$(TZ="$TARGET_TZ" date -d "tomorrow $TARGET_TIME" +%s)"
  fi
  sleep_seconds=$((target_epoch - now_epoch))
  echo "target=$(TZ="$TARGET_TZ" date -d "@$target_epoch") ($TARGET_TZ), sleeping ${sleep_seconds}s"
  sleep "$sleep_seconds"
fi

echo "cleanup starting at $(date)"

current_shell_pid="$$"
current_process_group="$(ps -o pgid= -p "$current_shell_pid" | tr -d ' ')"

kill_pids() {
  local label="$1"
  shift
  local pids=("$@")
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "$label: no processes found"
    return 0
  fi

  echo "$label: candidate pids: ${pids[*]}"
  ps -fp "${pids[@]}" || true

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$label: dry-run, not killing"
    return 0
  fi

  echo "$label: sending SIGTERM"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep "$GRACE_SECONDS"

  local still_alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_alive+=("$pid")
    fi
  done
  if [[ "${#still_alive[@]}" -gt 0 ]]; then
    echo "$label: sending SIGKILL to remaining pids: ${still_alive[*]}"
    kill -KILL "${still_alive[@]}" 2>/dev/null || true
  fi
}

gpu_pids=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r raw_pid; do
    pid="$(echo "$raw_pid" | tr -d ' ')"
    [[ -z "$pid" ]] && continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')"
    if [[ "$owner" == "$USER" && "$pid" != "$current_shell_pid" ]]; then
      gpu_pids+=("$pid")
    fi
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
else
  echo "nvidia-smi not found; skipping GPU-specific process discovery"
fi

kill_pids "gpu-processes-owned-by-$USER" "${gpu_pids[@]}"

if [[ "$RELATED_PROCS" -eq 1 ]]; then
  related_pids=()
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$current_shell_pid" ]] && continue
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$pgid" == "$current_process_group" ]] && continue
    cmdline="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    if [[ "$cmdline" =~ (llm_mst_finder|local_orchestrator|slurm_orchestrator|mst_adapter|run-trial|run_trial|search|vllm|vllm.entrypoints|live_reasoning_smoke|live_code_workloads|single_gpu_model_loop|single-gpu-model-loop) ]]; then
      related_pids+=("$pid")
    fi
  done < <(pgrep -u "$USER" || true)
  kill_pids "remaining-mst-orchestrator-processes-owned-by-$USER" "${related_pids[@]}"
fi

echo "cleanup finished at $(date)"
