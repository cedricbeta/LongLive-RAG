#!/usr/bin/env python3
"""Build the 12 budgeted P2 configs after, and only after, a passing G1."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "bearded_watchmaker_workshop": ["watchmaker"],
    "tattooed_noodle_chef": ["chef"],
    "brown_bear_river": ["bear"],
    "sandy_beach_driftwood": ["driftwood"],
}
ARMS = ("capsule", "full_vector", "random_subspace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    g1_path = pilot_root / "g1/g1_results.json"
    if not g1_path.exists():
        raise SystemExit("G1 result is missing; P2 must not start")
    g1 = json.loads(g1_path.read_text(encoding="utf-8"))
    if not g1.get("gate_pass") or not g1.get("selection"):
        raise SystemExit(
            "G1 is not artifact-ready (no-subspace kill or final-refit block); "
            "P2 is forbidden"
        )
    layer = int(g1["selection"]["layer"])
    rank = int(g1["selection"]["rank"])
    g0_path = pilot_root / "g0/g0_results.json"
    if not g0_path.exists():
        raise SystemExit("G0 result is missing; P2 must not start")
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    if not g0.get("gate_pass"):
        raise SystemExit("G0 did not pass; P2 is forbidden")
    alpha_scan_path = pilot_root / "p2/alpha_scan.json"
    if not alpha_scan_path.exists():
        raise SystemExit("alpha stability scan is missing; P2 must not start")
    alpha_scan = json.loads(alpha_scan_path.read_text(encoding="utf-8"))
    if (
        not alpha_scan.get("gate_pass")
        or not alpha_scan.get("visual_review_complete")
        or alpha_scan.get("selected_alpha") is None
    ):
        raise SystemExit(
            "rendered alpha scan and visual review are incomplete; P2 must not start"
        )
    selected_alpha = float(alpha_scan["selected_alpha"])
    if args.alpha is not None and abs(float(args.alpha) - selected_alpha) > 1e-8:
        raise SystemExit(
            f"--alpha={args.alpha} cannot override sealed alpha={selected_alpha}"
        )
    if selected_alpha not in (0.25, 0.5, 1.0):
        raise SystemExit("P2 alpha must be one of {0.25,0.5,1.0}")
    base_path = (
        ROOT
        / "videos/benchmark_hard_multishot/stage_A/configs/baseline_seed0.yaml"
    )
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    records = []
    for arm in ARMS:
        for scene, subject_phrases in SCENES.items():
            config = deepcopy(base)
            config["data"]["data_path"] = "example/benchmark_hard_multishot"
            config["data_path"] = "example/benchmark_hard_multishot"
            config["sample_names"] = [scene]
            render_dir = pilot_root / "p2" / arm / "renders"
            runtime_dir = pilot_root / "p2" / arm / "runtime"
            inference = config.setdefault("inference", {})
            config["output_folder"] = str(render_dir.relative_to(ROOT))
            inference["output_folder"] = str(render_dir.relative_to(ROOT))
            inference["filename_prefix"] = f"p2_{arm}"
            inference["filename_from_sample_name"] = True
            inference["kv_rag"] = {"enabled": False}
            inference["capsule_probe"] = {"enabled": False}
            mask_path = pilot_root / "clipseg_masks/seed0" / f"{scene}.pt"
            mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
            mask_method = (
                "clipseg_precomputed_with_manual_visibility_overrides"
                if mask_payload.get("metadata", {}).get("manual_visibility_exclusions")
                else "clipseg_precomputed"
            )
            artifact_path = pilot_root / "g1/artifacts" / f"{scene}.pt"
            if not artifact_path.exists():
                raise SystemExit(f"missing selected G1 artifact: {artifact_path}")
            scene_g0 = g0["scenes"][scene]
            subject_stream = scene_g0.get(
                "reference_per_shot", scene_g0.get("baseline_per_shot")
            )[
                "subject_anchor_to_shot0"
            ]
            worst_shot = int(np.argmin(np.asarray(subject_stream[1:])) + 1)
            boundaries = list(range(worst_shot, 8))
            inference["capsule_edit"] = {
                "enabled": True,
                "output_dir": str(runtime_dir.relative_to(ROOT)),
                "arm": arm,
                "artifact_path": str(artifact_path.relative_to(ROOT)),
                "layer": layer,
                "rank": rank,
                "alpha": selected_alpha,
                "boundaries": boundaries,
                "random_seed": 20260715,
                "mask_method": mask_method,
                "mask_path": str(mask_path.relative_to(ROOT)),
                "subject_phrases": subject_phrases,
                "allow_missing_subject": scene == "sandy_beach_driftwood",
                "fail_open": False,
            }
            config_path = pilot_root / "p2/configs" / f"{arm}__{scene}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, width=1000),
                encoding="utf-8",
            )
            records.append(
                {
                    "arm": arm,
                    "scene": scene,
                    "seed": 0,
                    "config": str(config_path.relative_to(ROOT)),
                    "render_dir": str(render_dir.relative_to(ROOT)),
                    "runtime_dir": str(runtime_dir.relative_to(ROOT)),
                    "layer": layer,
                    "rank": rank,
                    "alpha": selected_alpha,
                    "worst_round18_drift_shot": worst_shot,
                    "boundaries": boundaries,
                    "baseline_reference": scene_g0.get(
                        "reference", scene_g0.get("baseline")
                    ),
                    "status": "not_started",
                }
            )
    manifest = {
        "schema_version": 2,
        "render_budget": 12,
        "actual_planned_renders": len(records),
        "new_renders_after_alpha_promotion": 11,
        "alpha_scan_selected_render_reused_as_final": True,
        "delivery": "same in-place active-window V boundary edit for all arms",
        "invariants": [
            "K cache untouched",
            "RoPE phase untouched",
            "same seed0 masks, boundaries, alpha, layer, and rank",
            "only projector arm differs",
        ],
        "g1_result": str(g1_path.relative_to(ROOT)),
        "alpha_scan": str(alpha_scan_path.relative_to(ROOT)),
        "selected_alpha": selected_alpha,
        "runs": records,
    }
    output_path = pilot_root / "p2/p2_run_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {len(records)} P2 configs to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
