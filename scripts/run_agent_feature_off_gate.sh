#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/videos/agentic_runtime_v1/feature_off"
config_dir="$out/configs"
render_dir="$out/renders"
log_dir="$out/logs"
mkdir -p "$config_dir" "$render_dir" "$log_dir"

exec 9>"$out/driver.lock"
if ! flock -n 9; then
  echo "another agent feature-off driver holds $out/driver.lock" >&2
  exit 75
fi

scenes=(
  bearded_watchmaker_workshop
  tattooed_noodle_chef
  brown_bear_river
  sandy_beach_driftwood
)
gpus=(0 1 2 3)

for i in "${!scenes[@]}"; do
  scene="${scenes[$i]}"
  source="$root/videos/capsule_pilot_20260715/configs/probe_seed0__${scene}.yaml"
  target="$config_dir/agent_off__${scene}.yaml"
  SOURCE="$source" TARGET="$target" RENDER_DIR="$render_dir" SCENE="$scene" \
    /home/chendong/miniconda3/bin/python - <<'PY'
import os
from pathlib import Path
import yaml

source = Path(os.environ["SOURCE"])
target = Path(os.environ["TARGET"])
data = yaml.safe_load(source.read_text(encoding="utf-8"))
data["output_folder"] = os.environ["RENDER_DIR"]
data.setdefault("inference", {})["output_folder"] = os.environ["RENDER_DIR"]
data["inference"]["filename_prefix"] = "agent_off"
data["inference"]["agent_runtime"] = {"enabled": False}
data["inference"].setdefault("capsule_probe", {})["enabled"] = False
data["inference"].setdefault("capsule_edit", {})["enabled"] = False
data["sample_names"] = [os.environ["SCENE"]]
target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
done

run_one() {
  local scene="$1"
  local gpu="$2"
  local config="$config_dir/agent_off__${scene}.yaml"
  local log="$log_dir/${scene}.log"
  local archive="$log_dir/failures/${scene}"
  local attempt=0
  while (( attempt < 2 )); do
    local free
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if (( free < 61440 )); then
      echo "GPU $gpu has ${free} MiB free; require >= 61440 MiB" >>"$log"
      return 70
    fi
    echo "START $(date -Iseconds) scene=$scene gpu=$gpu attempt=$attempt free_mib=$free" >>"$log"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
      /home/chendong/miniconda3/bin/python "$root/inference.py" --config_path "$config" \
      >>"$log" 2>&1
    status=$?
    set -e
    echo "END $(date -Iseconds) scene=$scene attempt=$attempt status=$status" >>"$log"
    if (( status == 0 )); then
      echo 0 >"$log_dir/${scene}.exit_code"
      return 0
    fi
    mkdir -p "$archive"
    cp "$log" "$archive/attempt${attempt}.log"
    attempt=$((attempt + 1))
  done
  echo "$status" >"$log_dir/${scene}.exit_code"
  return "$status"
}

pids=()
for i in "${!scenes[@]}"; do
  run_one "${scenes[$i]}" "${gpus[$i]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
