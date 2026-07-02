#!/bin/bash
# Round 18 watchdog: while the run is unfinished, relaunch the resume runner
# whenever the loop/grid builder is not running. Per-stem resume makes each
# restart cheap. Caps restarts to avoid an infinite crash loop.
cd /home/chendong/video-gen/LongLive-RAG
LOG=docs/round18_bench_20260701/watchdog.log
RUNLOG=docs/round18_bench_20260701/run.log
restarts=0
echo "[watchdog] started $(date +%H:%M)" >> "$LOG"
while true; do
  if grep -q "GRIDS_DONE\|GRIDS_FAILED" "$RUNLOG" 2>/dev/null; then
    echo "[watchdog] $(date +%H:%M) run finished ($(grep -o 'GRIDS_[A-Z]*' "$RUNLOG" | tail -1)); exiting" >> "$LOG"
    break
  fi
  if ! pgrep -f "run_round17_vlm_loop.py" >/dev/null \
     && ! pgrep -f "build_round17_artifacts.py" >/dev/null \
     && ! pgrep -f "resume_runner" >/dev/null; then
    if [ "$restarts" -ge 8 ]; then
      echo "[watchdog] $(date +%H:%M) restart cap (8) reached; giving up" >> "$LOG"
      break
    fi
    restarts=$((restarts + 1))
    echo "[watchdog] $(date +%H:%M) loop dead and run unfinished -> relaunch (attempt $restarts)" >> "$LOG"
    bash docs/round18_bench_20260701/resume_runner2.sh >> "$LOG" 2>&1
  fi
  sleep 120
done
