#!/usr/bin/env bash
# Queue worker for the NarraStream three-arm renders (arms A/B).
# Each worker claims the next unclaimed config via flock and renders it.
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

gpu="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
run="$root/videos/narrastream_threearm_20260827"
lock="$run/queue.lock"
cd "$root"
mkdir -p "$run/logs" "$run/claims"

while true; do
  next=""
  exec 9>"$lock"
  flock 9
  for cfg in "$run"/configs/ns3arm__*.yaml; do
    stem="$(basename "$cfg" .yaml)"
    if [[ ! -e "$run/claims/$stem" ]]; then
      touch "$run/claims/$stem"
      next="$cfg"
      break
    fi
  done
  flock -u 9
  exec 9>&-

  [[ -z "$next" ]] && { echo "worker gpu=$gpu: queue empty, exiting"; break; }

  stem="$(basename "$next" .yaml)"
  log="$run/logs/${stem}.log"
  echo "worker gpu=$gpu: rendering $stem"
  {
    echo "START $(date -Iseconds) gpu=$gpu config=$next"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      PYTHONUNBUFFERED=1 \
      /home/chendong/miniconda3/bin/python inference.py --config_path "$next"
    status=$?
    echo "END $(date -Iseconds) status=$status"
    echo "$status" > "$run/logs/${stem}.exit_code"
  } > "$log" 2>&1
done
