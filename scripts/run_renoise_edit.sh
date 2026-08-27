#!/usr/bin/env bash
# SDEdit masked-renoise repair arms (docs attention study follow-up).
set -uo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU CONFIG [CONFIG ...]" >&2
  exit 2
fi
gpu="$1"; shift
root="$(cd "$(dirname "$0")/.." && pwd)"
log_dir="$root/videos/renoise_edit_20260821/logs"
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
      /home/chendong/miniconda3/bin/python inference.py --config_path "$config"
    status=$?
    echo "END $(date -Iseconds) status=$status"
    echo "$status" > "$exit_file"
    [[ $status -ne 0 ]] && overall=1
  } > "$log" 2>&1
done
exit "$overall"
