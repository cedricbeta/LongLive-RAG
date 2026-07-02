#!/bin/bash
# Restartable Round 18 runner: picks the emptiest GPU for the generator at
# launch time (the shared machine's neighbor jobs move around), resumes the
# loop, then builds grids on success.
cd /home/chendong/video-gen/LongLive-RAG
LOG=docs/round18_bench_20260701/run.log

FREEST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "[runner2] $(date +%H:%M) generator on GPU $FREEST" >> "$LOG"

python scripts/run_round17_vlm_loop.py \
  --docs_root docs/round18_bench_20260701 \
  --video_root videos/round18_bench_20260701 \
  --admission_json_override docs/benchmark_hard_multishot/stage_A_drift_audit.json \
  --prompts_dir example/benchmark_hard_multishot \
  --baseline_seeds 0 \
  --rounds 1 \
  --min_admitted_main_scenes 9 \
  --extra_logit_bias_lambdas "" \
  --optimizer_gpu 2 --generator_gpu "$FREEST" --clip_device "cuda:$FREEST" \
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
