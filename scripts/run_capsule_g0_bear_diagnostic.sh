#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

gpu="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
pilot="$root/videos/capsule_pilot_20260715"
config="videos/capsule_pilot_20260715/configs/g0_bear_probe_off_seed0.yaml"
log="$pilot/logs/g0_bear_probe_off_seed0.log"
exit_file="$pilot/logs/g0_bear_probe_off_seed0.exit_code"
gpu_file="$pilot/logs/g0_bear_probe_off_seed0.gpu"
lock_file="$pilot/logs/g0_bear_probe_off_seed0.lock"
mkdir -p "$pilot/logs"
cd "$root"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "another brown-bear diagnostic runner owns $lock_file" >&2
  exit 0
fi
if [[ -f "$exit_file" ]]; then
  previous_exit="$(tr -d '[:space:]' < "$exit_file")"
  if [[ "$previous_exit" == "0" ]] \
    && compgen -G "$pilot/g0/bear_probe_off/renders/*-brown_bear_river-*_regular.mp4" >/dev/null; then
    echo "brown-bear diagnostic is already complete; reusing existing render" >&2
    exit 0
  fi
  echo "brown-bear diagnostic already ended with code=$previous_exit; explicit cleanup is required before retry" >&2
  exit "${previous_exit:-1}"
fi
rm -f "$exit_file"
printf '%s\n' "$gpu" > "$gpu_file"

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
  /home/chendong/miniconda3/bin/python - "$status" <<'PY'
import json
import sys
from pathlib import Path

path = Path("videos/capsule_pilot_20260715/g0/bear_diagnostic_manifest.json")
payload = json.loads(path.read_text())
payload["status"] = "complete" if int(sys.argv[1]) == 0 else "failed"
payload["exit_code"] = int(sys.argv[1])
payload["log"] = "videos/capsule_pilot_20260715/logs/g0_bear_probe_off_seed0.log"
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
  exit "$status"
} > "$log" 2>&1
