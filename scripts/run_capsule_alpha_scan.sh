#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

gpu="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
/home/chendong/miniconda3/bin/python scripts/build_capsule_alpha_configs.py || exit $?
mapfile -t configs < <(/home/chendong/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("videos/capsule_pilot_20260715")
payload = json.loads((root / "p2/alpha_scan/render_manifest.json").read_text())
for run in payload["runs"]:
    print(run["config"])
PY
)
scripts/run_capsule_p2.sh "$gpu" "${configs[@]}"
