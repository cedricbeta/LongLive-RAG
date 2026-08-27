#!/usr/bin/env python3
"""Score the additive-route last-stand arms against the pre-registered gate.

Pre-registration: docs/laststand-and-narrastream-prereg.zh.md. For each scene,
delta = anchor_drift_aggregate_consistency(laststand_full) - (baseline_rerun).
Win requires delta >= the frozen 2-sigma paired-seed threshold in BOTH scenes
(stage_A_drift_audit.json). Anything else -> the additive route is archived.

Also records the knob-engagement counters from the render logs as the cheap
manipulation check (the attention-mass replay via analyze_attn_probe.py is the
full check and runs separately on the dumped probes).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.video_consistency import shot_anchor_centroid_consistency
from evaluation.vbench_consistency import build_clip_image_encoder, build_dino_encoder

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_capsule_g0 import (  # noqa: E402
    decode_selected,
    normalized_shot_embeddings,
    sampled_indices,
    shot_ranges,
)

SCENES = ("bearded_watchmaker_workshop", "sandy_beach_driftwood")
ARMS = ("baseline_rerun", "laststand_full")


def frozen_threshold(scene: str) -> float:
    audit = json.loads(
        (ROOT / "docs/benchmark_hard_multishot/stage_A_drift_audit.json").read_text()
    )
    entry = audit["noise_floor"][scene]
    return float(entry["sigma"]) * float(entry["sigma_multiplier"])


def locate_video(folder: Path) -> Path:
    hits = sorted(folder.rglob("*.mp4"))
    if not hits:
        raise RuntimeError(f"no mp4 under {folder}")
    return hits[0]


def knob_counters(log_path: Path) -> dict:
    counters = {}
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        for key in (
            "active_step_inject_skips",
            "head_filter_bias_calls",
            "phase_matched_k_captures",
            "phase_matched_k_replacements",
            "phase_matched_k_misses",
        ):
            m = re.findall(rf"{key}\D+(\d+)", text)
            if m:
                counters[key] = int(m[-1])
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="videos/laststand_20260827")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_root = ROOT / args.run_root
    dino = build_dino_encoder(device=args.device)
    clip = build_clip_image_encoder(device=args.device)

    result = {"scenes": {}, "gate": None}
    wins = []
    for scene in SCENES:
        ranges = shot_ranges(scene)
        indices = sampled_indices(ranges)
        per_arm = {}
        for arm in ARMS:
            video = locate_video(run_root / "renders" / f"{scene}__{arm}")
            frames, decode_info = decode_selected(video, indices)
            subject = normalized_shot_embeddings(frames, dino, shots=len(ranges))
            background = normalized_shot_embeddings(frames, clip, shots=len(ranges))
            metrics = shot_anchor_centroid_consistency(subject, background)
            per_arm[arm] = {
                "video": str(video),
                "frames": decode_info["frames"],
                "metrics": metrics,
            }
        threshold = frozen_threshold(scene)
        delta = (
            per_arm["laststand_full"]["metrics"]["anchor_drift_aggregate_consistency"]
            - per_arm["baseline_rerun"]["metrics"]["anchor_drift_aggregate_consistency"]
        )
        win = bool(delta >= threshold)
        wins.append(win)
        result["scenes"][scene] = {
            "arms": per_arm,
            "delta_anchor_drift_aggregate": delta,
            "frozen_2sigma_threshold": threshold,
            "win": win,
            "knob_counters": knob_counters(
                run_root / "logs" / f"laststand__{scene}__laststand_full.log"
            ),
        }

    result["gate"] = "WIN_both_scenes" if all(wins) else "ARCHIVE_additive_route"
    out = run_root / "laststand_verdict.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({s: {
        "delta": round(result["scenes"][s]["delta_anchor_drift_aggregate"], 4),
        "threshold": round(result["scenes"][s]["frozen_2sigma_threshold"], 4),
        "win": result["scenes"][s]["win"],
    } for s in SCENES}, indent=2))
    print("GATE:", result["gate"])
    print("wrote", out)


if __name__ == "__main__":
    main()
