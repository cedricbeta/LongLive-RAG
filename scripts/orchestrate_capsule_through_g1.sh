#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
pilot="$root/videos/capsule_pilot_20260715"
status_file="$pilot/orchestrator.status"
g0_eval_gpu="${CAPSULE_G0_EVAL_GPU:-0}"
cd "$root"

exec 8>"$pilot/logs/orchestrator.lock"
if ! flock -n 8; then
  echo "another capsule orchestrator is already running" >&2
  exit 0
fi

on_error() {
  code=$?
  echo "FAILED code=$code at $(date -Iseconds)" > "$status_file"
  exit "$code"
}
trap on_error ERR

wait_for_exit_files() {
  label="$1"
  shift
  while true; do
    missing=0
    for path in "$@"; do
      if [[ ! -f "$path" ]]; then
        missing=1
      fi
    done
    if (( missing == 0 )); then
      break
    fi
    echo "$(date -Iseconds) waiting for $label"
    sleep 30
  done
  for path in "$@"; do
    if [[ "$(tr -d '[:space:]' < "$path")" != "0" ]]; then
      echo "$label failed: $path=$(cat "$path")" >&2
      return 1
    fi
  done
}

seed0_exits=()
for scene in \
  bearded_watchmaker_workshop \
  tattooed_noodle_chef \
  brown_bear_river \
  sandy_beach_driftwood; do
  seed0_exits+=("$pilot/logs/probe_seed0__${scene}.exit_code")
done

echo "WAITING_P0 $(date -Iseconds)" > "$status_file"
wait_for_exit_files P0 "${seed0_exits[@]}"
/home/chendong/miniconda3/bin/python scripts/audit_capsule_probe_replays.py --seeds 0
CUDA_VISIBLE_DEVICES="$g0_eval_gpu" \
  /home/chendong/miniconda3/bin/python scripts/evaluate_capsule_g0.py --device cuda:0
if ! jq -e '.gate_pass == true' "$pilot/g0/g0_results.json" >/dev/null; then
  /home/chendong/miniconda3/bin/python scripts/build_capsule_pilot_report.py
  echo "G0_INFRA_BLOCKED_COMPLETE $(date -Iseconds)" > "$status_file"
  exit 0
fi

echo "RUNNING_P1_REPLAYS $(date -Iseconds)" > "$status_file"
p1_queue="$pilot/logs/p1_probe_queue.txt"
{
  for scene in \
    bearded_watchmaker_workshop \
    tattooed_noodle_chef \
    brown_bear_river \
    sandy_beach_driftwood; do
    for seed in 1 2; do
      printf 'videos/capsule_pilot_20260715/configs/probe_seed%s__%s.yaml\n' \
        "$seed" "$scene"
    done
  done
} > "$p1_queue"
workers=()
for gpu in 0 1 2 3; do
  scripts/run_capsule_probe_queue_worker.sh "$gpu" "$p1_queue" &
  workers+=("$!")
done
worker_status=0
for worker in "${workers[@]}"; do
  wait "$worker" || worker_status=1
done
if (( worker_status != 0 )); then
  echo "one or more P1 replay workers failed" >&2
  exit 1
fi

/home/chendong/miniconda3/bin/python scripts/audit_capsule_probe_replays.py
echo "RUNNING_G1 $(date -Iseconds)" > "$status_file"
/home/chendong/miniconda3/bin/python scripts/estimate_capsule_subspace.py \
  --random_trials 100 --bootstrap_repeats 1000
/home/chendong/miniconda3/bin/python scripts/build_capsule_g1_visuals.py

if jq -e '.gate_pass == true' "$pilot/g1/g1_results.json" >/dev/null; then
  if /home/chendong/miniconda3/bin/python scripts/calibrate_capsule_alpha.py; then
    echo "G1_PASS_READY_ALPHA_RENDER_SCAN $(date -Iseconds)" > "$status_file"
  else
    /home/chendong/miniconda3/bin/python scripts/build_capsule_pilot_report.py
    echo "ALPHA_NUMERIC_INFRA_BLOCKED_COMPLETE $(date -Iseconds)" > "$status_file"
  fi
else
  /home/chendong/miniconda3/bin/python scripts/build_capsule_pilot_report.py
  if jq -e '.verdict_if_stop == "blocked"' "$pilot/g1/g1_results.json" >/dev/null; then
    echo "G1_ARTIFACT_INFRA_BLOCKED_COMPLETE $(date -Iseconds)" > "$status_file"
  else
    echo "G1_NO_SUBSPACE_KILL_COMPLETE $(date -Iseconds)" > "$status_file"
  fi
fi
