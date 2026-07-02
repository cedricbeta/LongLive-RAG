#!/bin/bash
# Round 18 resume runner: continues the crashed run (kv_only OOM'd when
# neighbor processes swarmed GPU 1), then builds grids on success.
cd /home/chendong/video-gen/LongLive-RAG
LOG=docs/round18_bench_20260701/run.log

python scripts/run_round17_vlm_loop.py \
  --docs_root docs/round18_bench_20260701 \
  --video_root videos/round18_bench_20260701 \
  --admission_json_override docs/benchmark_hard_multishot/stage_A_drift_audit.json \
  --prompts_dir example/benchmark_hard_multishot \
  --baseline_seeds 0 \
  --rounds 1 \
  --min_admitted_main_scenes 9 \
  --extra_logit_bias_lambdas "" \
  --optimizer_gpu 2 --generator_gpu 3 --clip_device cuda:3 \
  --resume 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
echo "ROUND18B_EXIT=$code" >> "$LOG"

if [ "$code" -eq 0 ] && [ -f docs/round18_bench_20260701/round17_manifest.json ]; then
  python scripts/build_round17_artifacts.py \
    --docs_root docs/round18_bench_20260701 \
    --video_root videos/round18_bench_20260701 >> "$LOG" 2>&1 \
    && echo "GRIDS_DONE" >> "$LOG" \
    || echo "GRIDS_FAILED" >> "$LOG"
else
  echo "GRIDS_SKIPPED code=$code" >> "$LOG"
fi
