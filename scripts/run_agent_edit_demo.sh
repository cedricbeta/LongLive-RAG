#!/usr/bin/env bash
# Closed-loop agent edit_shot demo launcher (watchmaker + driftwood):
# drift judge -> masked-renoise repair -> re-review -> commit/quarantine.
# Long renders: run inside tmux. Engineering demo, no quality claim.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/videos/agent_edit_demo"
vlm_gpu="${AGENT_EDIT_VLM_GPU:-3}"
render_gpu="${AGENT_EDIT_RENDER_GPU:-2}"
mkdir -p "$out/failures"

exec 9>"$out/driver.lock"
if ! flock -n 9; then
  echo "another agent edit demo driver holds $out/driver.lock" >&2
  exit 75
fi

# GPU headroom: >=60 GiB on the VLM GPU (server) and on the render GPU.
for gpu in "$vlm_gpu" "$render_gpu"; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  if (( free < 61440 )); then
    echo "GPU $gpu has ${free} MiB free; require >= 61440 MiB" >&2
    exit 70
  fi
done

for attempt in 0 1; do
  log="$out/driver.attempt${attempt}.log"
  set +e
  /home/chendong/miniconda3/bin/python "$root/scripts/run_agent_edit_demo.py" \
    --gpu "$vlm_gpu" \
    --render_gpu "$render_gpu" \
    --clipseg_device "cuda:0" \
    --output_root "videos/agent_edit_demo" \
    >"$log" 2>&1
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
