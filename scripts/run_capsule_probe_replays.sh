#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU CONFIG [CONFIG ...]" >&2
  exit 2
fi

gpu="$1"
shift
root="$(cd "$(dirname "$0")/.." && pwd)"
log_dir="$root/videos/capsule_pilot_20260715/logs"
mkdir -p "$log_dir"
cd "$root"

overall=0
for config in "$@"; do
  stem="$(basename "$config" .yaml)"
  log="$log_dir/$stem.log"
  exit_file="$log_dir/$stem.exit_code"
  rm -f "$exit_file"
  {
    echo "START $(date -Iseconds) gpu=$gpu config=$config"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      PYTHONUNBUFFERED=1 \
      /usr/bin/time -v /home/chendong/miniconda3/bin/python inference.py \
      --config_path "$config"
    status=$?
    echo "END $(date -Iseconds) status=$status"
    echo "$status" > "$exit_file"
    if [[ $status -ne 0 ]]; then
      overall=1
    fi
  } > "$log" 2>&1
done

exit "$overall"
