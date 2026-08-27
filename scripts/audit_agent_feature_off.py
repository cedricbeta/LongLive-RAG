#!/usr/bin/env python3
"""Audit decoded-bit equality for explicit agent_runtime.enabled=false renders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)


def decoded_sha(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        digest.update(rgb.tobytes())
        count += 1
    capture.release()
    return digest.hexdigest(), count


def one(folder: Path, scene: str, prefix: str = "*") -> Path:
    matches = sorted(folder.glob(f"{prefix}-rank*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {scene} video in {folder}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="videos/agentic_runtime_v1/feature_off/audit.json")
    args = parser.parse_args()
    explicit_dir = ROOT / "videos/agentic_runtime_v1/feature_off/renders"
    reference_dir = ROOT / "videos/capsule_pilot_20260715/probe_replays/seed0/renders"
    records = {}
    for scene in SCENES:
        explicit = one(explicit_dir, scene, "agent_off")
        reference = one(reference_dir, scene, "probe_seed0")
        explicit_sha, explicit_frames = decoded_sha(explicit)
        reference_sha, reference_frames = decoded_sha(reference)
        records[scene] = {
            "explicit_agent_off": str(explicit.relative_to(ROOT)),
            "incumbent_reference": str(reference.relative_to(ROOT)),
            "explicit_decoded_sha256": explicit_sha,
            "reference_decoded_sha256": reference_sha,
            "explicit_frames": explicit_frames,
            "reference_frames": reference_frames,
            "decoded_bit_equal": explicit_sha == reference_sha and explicit_frames == reference_frames,
        }
    result = {
        "schema_version": 1,
        "gate": "feature_off_decoded_bit_equivalence",
        "scenes": records,
        "passed": all(record["decoded_bit_equal"] for record in records.values()),
        "quality_claim": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "scenes": len(records)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
