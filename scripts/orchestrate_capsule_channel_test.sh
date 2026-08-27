#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
pilot="$root/videos/capsule_pilot_20260715"
status_file="$pilot/channel_test/status"
queue="$pilot/logs/channel_test_queue.txt"
cd "$root"
mkdir -p "$pilot/channel_test"

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

/home/chendong/miniconda3/bin/python scripts/build_capsule_channel_test.py
jq -r '.runs[].config' "$pilot/channel_test/channel_run_manifest.json" > "$queue"
echo "RUNNING_4_ARMS $(date -Iseconds)" > "$status_file"
workers=()
for gpu in 0 1 2 3; do
  scripts/run_capsule_channel_queue_worker.sh "$gpu" "$queue" &
  workers+=("$!")
done
worker_status=0
for worker in "${workers[@]}"; do
  wait "$worker" || worker_status=1
done
if (( worker_status != 0 )); then
  echo "one or more channel-test workers failed after the allowed retry" >&2
  exit 1
fi

echo "RENDERS_COMPLETE_READY_EVAL $(date -Iseconds)" > "$status_file"
