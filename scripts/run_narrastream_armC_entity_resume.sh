#!/usr/bin/env bash
# Resume the arm-C entity campaign: run only scenes without g*/<scene>/summary.json, one scene per
# driver invocation, writing into the ORIGINAL group dir (g = manifest index mod 3) so the layout
# narrastream_prepare_eval.py expects is preserved.
#
# Each worker is pinned with CUDA_VISIBLE_DEVICES to its own card, so every "cuda:0" default
# allocation (SAM2/CLIPSeg init) lands on that card; the render child re-exports CUDA_VISIBLE_DEVICES
# to the PHYSICAL id (run_agent_edit_demo.render overrides the variable), hence --render_gpu "$gpu".
# GPU. (2026-09-02 19:45: SAM2 init OOMed on GPU 0 because the SGLang server had grown to 77 GB.)
#
#   ARMC_ENTITY_GPUS="1 2" scripts/run_narrastream_armC_entity_resume.sh          # rebuild queue
#   REUSE_QUEUE=1 ARMC_ENTITY_GPUS="1 2" scripts/run_narrastream_armC_entity_resume.sh  # keep queue
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${ARMC_ENTITY_OUT:-$root/videos/narrastream_threearm_20260827/armC_entity}"
base_url="${BASE_URL:-http://127.0.0.1:53456/v1}"
gpus=(${ARMC_ENTITY_GPUS:-1 2})
specs="${ARMC_ENTITY_SPECS:-example/narrastream_bench/entity_specs_subset33.json}"
rounds="${ARMC_ENTITY_ROUNDS:-3}"
mkdir -p "$out/logs" "$out/queue"
curl -s -m 5 "$base_url/models" >/dev/null || { echo "judge server not reachable at $base_url" >&2; exit 70; }

queue="$out/queue/remaining.txt"
if [ "${REUSE_QUEUE:-0}" != "1" ]; then
  python3 - "$root" "$out" >"$queue" <<'EOF'
import json, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
scenes = json.load(open(root / "videos/narrastream_threearm_20260827/manifest.json"))["scenes"]
for i, scene in enumerate(scenes):
    if not (out / f"g{i % 3}" / scene / "summary.json").exists():
        print(scene, i % 3)
EOF
fi
echo "$(wc -l <"$queue") scenes queued; workers on GPUs ${gpus[*]}"

worker() {
  local gpu=$1
  while true; do
    local line
    line=$(flock "$queue.lock" bash -c "head -n1 '$queue'; sed -i '1d' '$queue'")
    [ -z "$line" ] && break
    local scene=${line% *} g=${line#* }
    local d="$out/g$g/$scene"
    if [ -f "$d/summary.json" ]; then echo "[gpu$gpu] skip $scene (done)"; continue; fi
    if [ -d "$d" ]; then mv "$d" "${d}__partial_$(date +%m%d_%H%M%S)"; fi
    echo "[gpu$gpu] $(date +%H:%M) start $scene -> g$g"
    CUDA_VISIBLE_DEVICES="$gpu" "$root/venv/bin/python" "$root/scripts/run_agent_edit_demo.py" \
      --base_url "$base_url" --render_gpu "$gpu" --clipseg_device cuda:0 \
      --base_config videos/narrastream_threearm_20260827/armC_base.yaml \
      --data_root example/narrastream_bench/multitext_subset33_b8 \
      --output_root "$out/g$g" --scenes "$scene" --allow_missing_global \
      --judge entity --entity_specs "$specs" --max_edits_per_shot "$rounds" \
      >"$out/logs/${scene}.resume.log" 2>&1 \
      && echo "[gpu$gpu] $(date +%H:%M) done $scene" || echo "[gpu$gpu] $(date +%H:%M) FAILED $scene (see logs/${scene}.resume.log)"
  done
}
for gpu in "${gpus[@]}"; do worker "$gpu" & done
wait
echo "resume finished: $(ls "$out"/g*/ns*/summary.json | wc -l) scenes with summary.json"
