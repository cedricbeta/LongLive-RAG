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
manifest="$pilot/channel_test/channel_run_manifest.json"
lock_file="${queue_file}.lock"
min_free_mib=61440
overall=0
cd "$root"
exec 9>"$lock_file"

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

run_field() {
  config="$1"
  field="$2"
  jq -r --arg config "$config" --arg field "$field" \
    '.runs[] | select(.config == $config) | .[$field]' "$manifest"
}

config_is_complete() {
  config="$1"
  stem="$(basename "$config" .yaml)"
  exit_file="$pilot/logs/${stem}.exit_code"
  [[ -f "$exit_file" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$exit_file")" == 0 ]] || return 2
  scene="$(run_field "$config" scene)"
  runtime_dir="$root/$(run_field "$config" runtime_dir)"
  render_dir="$root/$(run_field "$config" render_dir)"
  [[ -f "$runtime_dir/$scene/seed0/manifest.json" ]] \
    && compgen -G "$render_dir/*-${scene}-*_regular.mp4" >/dev/null
}

archive_failed_run() {
  config="$1"
  stem="$(basename "$config" .yaml)"
  timestamp="$(date +%Y%m%dT%H%M%S)"
  for path in "$pilot/logs/${stem}.exit_code" "$pilot/logs/${stem}.log"; do
    if [[ -e "$path" ]]; then
      mv "$path" "${path}.failed.${timestamp}"
    fi
  done
  runtime_dir="$root/$(run_field "$config" runtime_dir)"
  render_dir="$root/$(run_field "$config" render_dir)"
  if [[ -d "$runtime_dir" ]]; then
    mv "$runtime_dir" "${runtime_dir}.failed.${timestamp}"
  fi
  if [[ -d "$render_dir" ]]; then
    mv "$render_dir" "${render_dir}.failed.${timestamp}"
  fi
  echo "gpu=$gpu archived failed channel arm $stem timestamp=$timestamp" >&2
}

while true; do
  # Do not reserve a config until this GPU is admissible. This lets an
  # already-free worker consume the queue instead of leaving work stranded on
  # a busy GPU.
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
    echo "gpu=$gpu reusing complete channel arm $config"
    continue
  else
    status=$?
    if [[ $status -eq 2 ]]; then
      archive_failed_run "$config"
      attempts=1
    fi
  fi
  # A preserved failed exit proves that this config already consumed its first
  # attempt in an earlier orchestrator process. Never reset its retry budget.
  if compgen -G "$pilot/logs/${stem}.exit_code.failed.*" >/dev/null; then
    attempts=1
  fi
  succeeded=0
  for attempt in $(seq 1 "$attempts"); do
    if scripts/run_capsule_channel_test.sh "$gpu" "$config"; then
      succeeded=1
      break
    fi
    if (( attempt < attempts )); then
      archive_failed_run "$config"
      while true; do
        free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= min_free_mib )); then
          break
        fi
        sleep 30
      done
    fi
  done
  if (( succeeded == 0 )); then
    overall=1
    break
  fi
done

exit "$overall"
