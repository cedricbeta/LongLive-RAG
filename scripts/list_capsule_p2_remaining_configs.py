#!/usr/bin/env python3
"""Print final P2 configs that still require rendering, one path per line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    manifest = json.loads((pilot_root / "p2/p2_run_manifest.json").read_text(encoding="utf-8"))
    remaining = []
    for record in manifest["runs"]:
        if str(record.get("status", "")).startswith("complete_reused"):
            continue
        config = ROOT / record["config"]
        exit_path = pilot_root / f"p2/logs/{config.stem}.exit_code"
        videos = sorted(
            (ROOT / record["render_dir"]).glob(
                f"*-{record['scene']}-*_regular.mp4"
            )
        )
        runtime_path = (
            ROOT
            / record["runtime_dir"]
            / record["scene"]
            / "seed0/manifest.json"
        )
        runtime_ok = False
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime_ok = not runtime.get("stats", {}).get("errors")
        exit_ok = exit_path.exists() and exit_path.read_text().strip() == "0"
        if not (exit_ok and len(videos) == 1 and runtime_ok):
            remaining.append(record)
    if len(remaining) > 11:
        raise SystemExit(
            "alpha promotion is missing: more than 11 final configs require rendering"
        )
    for record in remaining:
        print(record["config"])


if __name__ == "__main__":
    main()
