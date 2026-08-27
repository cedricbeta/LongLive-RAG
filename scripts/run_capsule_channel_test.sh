#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU CONFIG" >&2
  exit 2
fi

gpu="$1"
config="$2"
root="$(cd "$(dirname "$0")/.." && pwd)"
pilot="$root/videos/capsule_pilot_20260715"
stem="$(basename "$config" .yaml)"
log="$pilot/logs/${stem}.log"
exit_file="$pilot/logs/${stem}.exit_code"
cd "$root"
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
} > "$log" 2>&1

exit "$(tr -d '[:space:]' < "$exit_file")"
