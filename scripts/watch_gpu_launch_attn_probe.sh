#!/usr/bin/env bash
# Wait for a GPU with >=60GiB free (ops rule: VAE decode spikes OOM'd renders
# at 46-49GB margins), then launch the attn_probe replays inside tmux and wait
# for their exit codes. Exit 0 = both scenes done OK, 1 = a render failed,
# 3 = timed out waiting for a GPU or for completion.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
need_mib=61440
poll_s=90
timeout_s=$(( 12 * 3600 ))
session=attn_probe_20260820
log_dir="$root/videos/attn_probe_20260820/logs"
configs=(
  "videos/attn_probe_20260820/configs/sandy_beach_driftwood.yaml"
  "videos/attn_probe_20260820/configs/bearded_watchmaker_workshop.yaml"
)
mkdir -p "$log_dir"

start_ts=$(date +%s)
gpu=""
while true; do
  now=$(date +%s)
  if (( now - start_ts > timeout_s )); then
    echo "TIMEOUT waiting for a free GPU after ${timeout_s}s"
    exit 3
  fi
  gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used \
          --format=csv,noheader,nounits |
        awk -F', *' -v need="$need_mib" '($2-$3)>=need {print $1; exit}')
  if [[ -n "$gpu" ]]; then
    echo "GPU $gpu has enough free memory at $(date -Iseconds); launching"
    break
  fi
  sleep "$poll_s"
done

tmux new-session -d -s "$session" \
  "bash '$root/scripts/run_attn_probe_replays.sh' '$gpu' ${configs[*]}"
echo "launched tmux session $session on GPU $gpu"

# wait for both exit_code files
while true; do
  now=$(date +%s)
  if (( now - start_ts > timeout_s )); then
    echo "TIMEOUT waiting for renders after ${timeout_s}s"
    exit 3
  fi
  done_count=0
  overall=0
  for config in "${configs[@]}"; do
    stem="$(basename "$config" .yaml)"
    f="$log_dir/$stem.exit_code"
    if [[ -f "$f" ]]; then
      done_count=$((done_count + 1))
      code="$(cat "$f")"
      [[ "$code" != "0" ]] && overall=1
    fi
  done
  if (( done_count == ${#configs[@]} )); then
    echo "all renders finished at $(date -Iseconds), overall=$overall"
    exit "$overall"
  fi
  sleep "$poll_s"
done
