#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/videos/agentic_runtime_v1/demos"
gpu="${AGENT_VLM_GPU:-3}"
mkdir -p "$out/failures"

exec 9>"$out/driver.lock"
if ! flock -n 9; then
  echo "another agent VLM demo driver holds $out/driver.lock" >&2
  exit 75
fi

for attempt in 0 1; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  if (( free < 61440 )); then
    echo "GPU $gpu has ${free} MiB free; require >= 61440 MiB" >&2
    exit 70
  fi
  log="$out/driver.attempt${attempt}.log"
  set +e
  /home/chendong/miniconda3/bin/python "$root/scripts/run_agent_vlm_demos.py" \
    --gpu "$gpu" >"$log" 2>&1
  status=$?
  set -e
  if (( status == 0 )); then
    cp "$log" "$out/driver.log"
    echo 0 >"$out/driver.exit_code"
    exit 0
  fi
  archive="$out/failures/driver_attempt${attempt}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$archive"
  cp "$log" "$archive/driver.log"
  for ledger in "$out"/*/ledger.jsonl; do
    [[ -e "$ledger" ]] || continue
    scene="$(basename "$(dirname "$ledger")")"
    cp "$ledger" "$archive/${scene}.ledger.jsonl"
  done
done

echo "$status" >"$out/driver.exit_code"
exit "$status"
