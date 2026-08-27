#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 G0_EVAL_GPU" >&2
  exit 2
fi

gpu="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
pilot="$root/videos/capsule_pilot_20260715"
exit_file="$pilot/logs/g0_bear_probe_off_seed0.exit_code"
gpu_file="$pilot/logs/g0_bear_probe_off_seed0.gpu"
status_file="$pilot/bear_continuation.status"
cd "$root"

exec 7>"$pilot/logs/orchestrator.lock"
if ! flock -n 7; then
  echo "another capsule orchestrator is already running" >&2
  exit 0
fi
flock -u 7

on_error() {
  code=$?
  previous="$(cat "$status_file" 2>/dev/null || true)"
  printf 'CONTINUATION_FAILED code=%s previous=%q at %s\n' \
    "$code" "$previous" "$(date -Iseconds)" > "$status_file"
  exit "$code"
}
trap on_error ERR

echo "WAITING_BEAR_DIAGNOSTIC $(date -Iseconds)" > "$status_file"
while [[ ! -f "$exit_file" ]]; do
  sleep 30
done

exit_code="$(tr -d '[:space:]' < "$exit_file")"
if [[ "$exit_code" != "0" ]]; then
  /home/chendong/miniconda3/bin/python scripts/build_capsule_pilot_report.py
  echo "BEAR_DIAGNOSTIC_FAILED code=$exit_code $(date -Iseconds)" > "$status_file"
  exit 0
fi

run_gpu="$gpu"
if [[ -f "$gpu_file" ]]; then
  selected_gpu="$(tr -d '[:space:]' < "$gpu_file")"
  if [[ "$selected_gpu" =~ ^[0-9]+$ ]]; then
    run_gpu="$selected_gpu"
  fi
fi

echo "RUNNING_G0_THROUGH_G1 gpu=$run_gpu $(date -Iseconds)" > "$status_file"
CAPSULE_G0_EVAL_GPU="$run_gpu" scripts/orchestrate_capsule_through_g1.sh
orchestrator_status="$(cat "$pilot/orchestrator.status")"
printf '%s\n' "$orchestrator_status" > "$status_file"

if [[ "$orchestrator_status" == G1_PASS_READY_ALPHA_RENDER_SCAN* ]]; then
  echo "WAITING_ALPHA_RENDER_SCAN gpu=$run_gpu $(date -Iseconds)" > "$status_file"
  scripts/wait_and_run_capsule.sh "$run_gpu" alpha
  echo "EVALUATING_ALPHA_RENDER_SCAN gpu=$run_gpu $(date -Iseconds)" > "$status_file"
  CUDA_VISIBLE_DEVICES="$run_gpu" \
    /home/chendong/miniconda3/bin/python scripts/evaluate_capsule_alpha_scan.py \
    --device cuda:0
  echo "ALPHA_RENDER_SCAN_READY_FOR_VISUAL_REVIEW $(date -Iseconds)" > "$status_file"
fi
