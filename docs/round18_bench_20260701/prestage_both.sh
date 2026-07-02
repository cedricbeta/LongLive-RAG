#!/bin/bash
# Pre-render the `both` arm on a spare GPU while the main loop is still on
# kv_only. Reverse scene order + stop-at-both.log guarantee zero overlap with
# the loop: everything we finish shrinks the loop's missing set; the moment the
# loop claims the both phase (generation_logs/both.log appears) we stop
# rendering AND stop moving, so no same-path writes ever happen.
set -u
cd /home/chendong/video-gen/LongLive-RAG
ROOT=videos/round18_bench_20260701/iter01
LOG=docs/round18_bench_20260701/prestage.log
BOTH=$ROOT/both
STAGE=$ROOT/both_prestage
BOTHLOG=docs/round18_bench_20260701/iter01/generation_logs/both.log
GPU=0
mkdir -p "$BOTH" "$STAGE/out" "$STAGE/prompts" "$STAGE/configs"

SCENES="shimmering_puzzle_surface warm_indoor_dining vintage_archival_scene tattooed_noodle_chef sandy_beach_driftwood indoor_tender_moment brown_bear_river bearded_watchmaker_workshop aquamarine_underwater african_savanna"

for s in $SCENES; do
  if [ -f "$BOTHLOG" ]; then echo "$(date +%H:%M) loop claimed both phase; stopping" >> "$LOG"; break; fi
  if ls "$BOTH"/both-rank*-"$s"-*_regular.mp4 >/dev/null 2>&1; then
    echo "$(date +%H:%M) $s already in both/; skip" >> "$LOG"; continue
  fi
  if ! ls "$STAGE"/out/both-rank*-"$s"-*_regular.mp4 >/dev/null 2>&1; then
    python - "$s" <<'PY' >> "$LOG" 2>&1
import shutil, sys
from pathlib import Path
from omegaconf import OmegaConf
s = sys.argv[1]
root = Path("videos/round18_bench_20260701/iter01")
stage = root / "both_prestage"
pdir = stage / "prompts" / f"{s}_only"
if pdir.exists():
    shutil.rmtree(pdir)
pdir.mkdir(parents=True)
shutil.copytree(root / "prompt_refined" / s, pdir / s)
cfg = OmegaConf.load(root / "configs" / "kv_only.yaml")
cfg.data.data_path = str(pdir)
cfg.output_folder = str(stage / "out")
cfg.inference.output_folder = str(stage / "out")
cfg.inference.filename_prefix = "both"
OmegaConf.save(cfg, stage / "configs" / f"{s}.yaml")
print(f"[prestage] wrote config for {s}")
PY
    ok=0
    for attempt in 1 2; do
      echo "$(date +%H:%M) rendering $s (attempt $attempt) on GPU $GPU" >> "$LOG"
      CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python inference.py --config_path "$STAGE/configs/$s.yaml" >> "$LOG" 2>&1 && { ok=1; break; }
      echo "$(date +%H:%M) $s attempt $attempt failed" >> "$LOG"; sleep 30
    done
    [ "$ok" -eq 1 ] || { echo "$(date +%H:%M) $s failed twice; moving on" >> "$LOG"; continue; }
  fi
  if [ -f "$BOTHLOG" ]; then echo "$(date +%H:%M) loop claimed both phase; not moving $s" >> "$LOG"; break; fi
  for f in "$STAGE"/out/both-rank*-"$s"-*; do
    base=$(basename "$f")
    [ -e "$BOTH/$base" ] || cp "$f" "$BOTH/$base"
  done
  echo "$(date +%H:%M) staged+moved $s" >> "$LOG"
done
echo "PRESTAGE_DONE $(date +%H:%M)" >> "$LOG"
