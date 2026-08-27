#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU MODE [CONFIG ...]" >&2
  exit 2
fi

gpu="$1"
mode="$2"
shift 2
if [[ "$mode" != "g0_bear" && "$mode" != "alpha" && $# -lt 1 ]]; then
  echo "mode $mode requires at least one config" >&2
  exit 2
fi
root="$(cd "$(dirname "$0")/.." && pwd)"
wait_log="$root/videos/capsule_pilot_20260715/logs/wait_gpu${gpu}_${mode}.log"
min_free_mib=61440
mkdir -p "$(dirname "$wait_log")"

while true; do
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  timestamp="$(date -Iseconds)"
  echo "$timestamp gpu=$gpu free_mib=$free min_free_mib=$min_free_mib" >> "$wait_log"
  if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= min_free_mib )); then
    break
  fi
  sleep 30
done

echo "$(date -Iseconds) gpu=$gpu starting mode=$mode configs=$*" >> "$wait_log"
case "$mode" in
  probe)
    exec "$root/scripts/run_capsule_probe_replays.sh" "$gpu" "$@"
    ;;
  p2)
    exec "$root/scripts/run_capsule_p2.sh" "$gpu" "$@"
    ;;
  g0_bear)
    if [[ $# -ne 0 ]]; then
      echo "g0_bear mode takes no config arguments" >&2
      exit 2
    fi
    exec "$root/scripts/run_capsule_g0_bear_diagnostic.sh" "$gpu"
    ;;
  alpha)
    if [[ $# -ne 0 ]]; then
      echo "alpha mode takes no config arguments" >&2
      exit 2
    fi
    exec "$root/scripts/run_capsule_alpha_scan.sh" "$gpu"
    ;;
  *)
    echo "unsupported mode: $mode" >&2
    exit 2
    ;;
esac
