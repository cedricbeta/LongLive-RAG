#!/usr/bin/env bash
# NarraStream arm C (repair loop) with the entity judge over the 33-scene subset:
# three drivers, one per render GPU, sharing one Qwen3-VL judge server.
#   scripts/run_narrastream_armC_entity.sh            # launches; logs under $out/logs
# Requires the judge server on $BASE_URL (default http://127.0.0.1:53456/v1) and
# example/narrastream_bench/entity_specs_subset33.json (scripts/extract_entity_specs.py).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${ARMC_ENTITY_OUT:-$root/videos/narrastream_threearm_20260827/armC_entity}"
base_url="${BASE_URL:-http://127.0.0.1:53456/v1}"
gpus=(${ARMC_ENTITY_GPUS:-0 1 2})
specs="${ARMC_ENTITY_SPECS:-example/narrastream_bench/entity_specs_subset33.json}"
rounds="${ARMC_ENTITY_ROUNDS:-3}"
mkdir -p "$out/logs"
curl -s -m 5 "$base_url/models" >/dev/null || { echo "judge server not reachable at $base_url" >&2; exit 70; }
mapfile -t scenes < <(python3 -c "import json;print('\n'.join(json.load(open('$root/videos/narrastream_threearm_20260827/manifest.json'))['scenes']))")
n=${#gpus[@]}
for ((g = 0; g < n; g++)); do
  group=()
  for ((i = g; i < ${#scenes[@]}; i += n)); do group+=("${scenes[$i]}"); done
  gpu=${gpus[$g]}
  nohup /home/chendong/miniconda3/bin/python "$root/scripts/run_agent_edit_demo.py" \
    --base_url "$base_url" --render_gpu "$gpu" --clipseg_device "cuda:$gpu" \
    --base_config videos/narrastream_threearm_20260827/armC_base.yaml \
    --data_root example/narrastream_bench/multitext_subset33_b8 \
    --output_root "$out/g$g" --scenes "${group[@]}" --allow_missing_global \
    --judge entity --entity_specs "$specs" --max_edits_per_shot "$rounds" \
    >"$out/logs/g$g.log" 2>&1 &
  echo "group $g on gpu $gpu: ${#group[@]} scenes, pid $!"
done
