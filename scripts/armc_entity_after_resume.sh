#!/usr/bin/env bash
# After the arm-C entity resume finishes: build its NarraStream eval_data and run the 11-metric
# evaluation on GPU 3 (waits for any running evaluator to finish first, and for the judge server).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
bench=/home/chendong/video-gen/NarraStream-Bench
resume_log="${1:?path to armC_resume.log}"
base_url="${BASE_URL:-http://127.0.0.1:53456/v1}"

until grep -q "resume finished" "$resume_log"; do sleep 120; done
n=$(ls "$root"/videos/narrastream_threearm_20260827/armC_entity/g*/ns*/summary.json | wc -l)
echo "$(date +%H:%M) arm C resume finished: $n scenes with summary.json"

cd "$root"
venv/bin/python scripts/narrastream_prepare_eval.py --arm armC_entity \
  --summary_glob 'videos/narrastream_threearm_20260827/armC_entity/g*/{scene}/summary.json' \
  --out videos/narrastream_eval/armC_entity
echo "$(date +%H:%M) eval_data written: $(python3 -c "import json;print(len(json.load(open('videos/narrastream_eval/armC_entity/processed/eval_data.json'))))") samples"

# GPU 3 is the evaluation card; wait until no other evaluator is on it
until ! pgrep -f "core\.evaluat[e]" >/dev/null; do sleep 60; done
curl -s -m 5 "$base_url/models" >/dev/null || { echo "judge server not reachable at $base_url" >&2; exit 70; }
cd "$bench"
bash scripts/run_narrastream_bench.sh --run-name armC_entity_q8b \
  --eval-data "$root/videos/narrastream_eval/armC_entity/processed/eval_data.json" \
  --config configs/local_qwen3vl8b.yaml --gpu-id 3 --api-workers 2 --resume
echo "$(date +%H:%M) arm C evaluation finished"
