#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU QUEUE_FILE" >&2
  exit 2
fi

gpu="$1"
queue_file="$2"
root="$(cd "$(dirname "$0")/.." && pwd)"
pilot="$root/videos/capsule_pilot_20260715"
lock_file="${queue_file}.lock"
min_free_mib=61440
overall=0
cd "$root"

exec 9>"$lock_file"

wait_for_gpu() {
  while true; do
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= min_free_mib )); then
      return 0
    fi
    sleep 30
  done
}

queue_has_work() {
  flock 9
  local first
  first="$(head -n 1 "$queue_file" 2>/dev/null || true)"
  flock -u 9
  [[ -n "$first" ]]
}

wait_for_gpu_or_empty() {
  while true; do
    if ! queue_has_work; then
      return 1
    fi
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= min_free_mib )); then
      return 0
    fi
    sleep 30
  done
}

config_is_complete() {
  config="$1"
  stem="$(basename "$config" .yaml)"
  seed="${stem#probe_seed}"
  seed="${seed%%__*}"
  scene="${stem#*__}"
  exit_file="$pilot/logs/${stem}.exit_code"
  manifest="$pilot/probe_replays/seed${seed}/dumps/${scene}/seed${seed}/manifest.json"
  render_dir="$pilot/probe_replays/seed${seed}/renders"
  if [[ ! -f "$exit_file" ]]; then
    return 1
  fi
  exit_code="$(tr -d '[:space:]' < "$exit_file")"
  if [[ "$exit_code" != "0" ]]; then
    echo "existing failed replay: $stem code=$exit_code" >&2
    return 2
  fi
  [[ -f "$manifest" ]] \
    && compgen -G "$render_dir/probe_seed${seed}-rank0-${scene}-*_regular.mp4" >/dev/null
}

archive_failed_replay() {
  config="$1"
  stem="$(basename "$config" .yaml)"
  seed="${stem#probe_seed}"
  seed="${seed%%__*}"
  scene="${stem#*__}"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  for path in \
    "$pilot/logs/${stem}.exit_code" \
    "$pilot/logs/${stem}.log"; do
    if [[ -e "$path" ]]; then
      mv "$path" "${path}.failed.${timestamp}"
    fi
  done
  dump_dir="$pilot/probe_replays/seed${seed}/dumps/${scene}"
  if [[ -d "$dump_dir" ]]; then
    mv "$dump_dir" "${dump_dir}.failed.${timestamp}"
  fi
  render_dir="$pilot/probe_replays/seed${seed}/renders"
  while IFS= read -r path; do
    mv "$path" "${path}.failed.${timestamp}"
  done < <(
    find "$render_dir" -maxdepth 1 -type f \
      -name "probe_seed${seed}-rank0-${scene}-*" -print 2>/dev/null
  )
  echo "gpu=$gpu archived failed replay $stem timestamp=$timestamp" >&2
}

while true; do
  # Wait without reserving a config, and exit promptly once another admissible
  # worker has drained the queue.
  if ! wait_for_gpu_or_empty; then
    break
  fi
  flock 9
  config="$(head -n 1 "$queue_file" 2>/dev/null || true)"
  if [[ -z "$config" ]]; then
    flock -u 9
    continue
  fi
  tail -n +2 "$queue_file" > "${queue_file}.tmp.$$"
  mv "${queue_file}.tmp.$$" "$queue_file"
  flock -u 9

  attempts=2
  stem="$(basename "$config" .yaml)"
  if config_is_complete "$config"; then
    echo "gpu=$gpu reusing complete replay $config"
    continue
  else
    status=$?
    if [[ $status -eq 2 ]]; then
      archive_failed_replay "$config"
      attempts=1
    fi
  fi
  # Preserve the retry budget across orchestrator restarts.
  if compgen -G "$pilot/logs/${stem}.exit_code.failed.*" >/dev/null; then
    attempts=1
  fi

  succeeded=0
  for attempt in $(seq 1 "$attempts"); do
    if scripts/run_capsule_probe_replays.sh "$gpu" "$config"; then
      succeeded=1
      break
    fi
    if (( attempt < attempts )); then
      archive_failed_replay "$config"
      wait_for_gpu
    fi
  done
  if (( succeeded == 0 )); then
    overall=1
    break
  fi
done

exit "$overall"
