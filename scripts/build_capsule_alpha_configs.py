#!/usr/bin/env python3
"""Build the dedicated one-scene rendered alpha-scan configs after G1."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENE = "bearded_watchmaker_workshop"


def alpha_token(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    g0 = json.loads((pilot_root / "g0/g0_results.json").read_text(encoding="utf-8"))
    g1 = json.loads((pilot_root / "g1/g1_results.json").read_text(encoding="utf-8"))
    numeric_path = pilot_root / "p2/alpha_numeric_prefilter.json"
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if not g0.get("gate_pass") or not g1.get("gate_pass") or not numeric.get("gate_pass"):
        raise SystemExit("G0, G1, and the alpha numeric prefilter must pass")
    candidates = [float(value) for value in numeric["render_alphas"]]
    if set(candidates) != {0.25, 0.5, 1.0}:
        raise SystemExit(
            "the rendered protocol requires all finite alpha candidates {0.25,0.5,1.0}; "
            f"got {candidates}"
        )
    if any(value not in (0.25, 0.5, 1.0) for value in candidates):
        raise SystemExit(f"unexpected alpha candidates: {candidates}")

    layer = int(g1["selection"]["layer"])
    rank = int(g1["selection"]["rank"])
    scene_g0 = g0["scenes"][SCENE]
    subject_stream = scene_g0.get(
        "reference_per_shot", scene_g0.get("baseline_per_shot")
    )["subject_anchor_to_shot0"]
    worst_shot = int(np.argmin(np.asarray(subject_stream[1:])) + 1)
    boundaries = list(range(worst_shot, 8))
    base_path = ROOT / "videos/benchmark_hard_multishot/stage_A/configs/baseline_seed0.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    mask_path = pilot_root / f"clipseg_masks/seed0/{SCENE}.pt"
    mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    mask_method = (
        "clipseg_precomputed_with_manual_visibility_overrides"
        if mask_payload.get("metadata", {}).get("manual_visibility_exclusions")
        else "clipseg_precomputed"
    )
    artifact_path = pilot_root / f"g1/artifacts/{SCENE}.pt"
    if not artifact_path.exists():
        raise SystemExit(f"missing G1 artifact: {artifact_path}")

    records = []
    for alpha in candidates:
        token = alpha_token(alpha)
        run_root = pilot_root / "p2/alpha_scan" / f"alpha_{token}"
        config = deepcopy(base)
        config["data"]["data_path"] = "example/benchmark_hard_multishot"
        config["data_path"] = "example/benchmark_hard_multishot"
        config["sample_names"] = [SCENE]
        render_dir = run_root / "renders"
        runtime_dir = run_root / "runtime"
        inference = config.setdefault("inference", {})
        config["output_folder"] = str(render_dir.relative_to(ROOT))
        inference["output_folder"] = str(render_dir.relative_to(ROOT))
        inference["filename_prefix"] = f"alpha_{token}"
        inference["filename_from_sample_name"] = True
        inference["kv_rag"] = {"enabled": False}
        inference["capsule_probe"] = {"enabled": False}
        inference["capsule_edit"] = {
            "enabled": True,
            "output_dir": str(runtime_dir.relative_to(ROOT)),
            "arm": "capsule",
            "artifact_path": str(artifact_path.relative_to(ROOT)),
            "layer": layer,
            "rank": rank,
            "alpha": alpha,
            "boundaries": boundaries,
            "random_seed": 20260715,
            "mask_method": mask_method,
            "mask_path": str(mask_path.relative_to(ROOT)),
            "subject_phrases": ["watchmaker"],
            "allow_missing_subject": False,
            "fail_open": False,
        }
        config_path = pilot_root / "p2/alpha_scan/configs" / f"alpha_{token}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8")
        records.append(
            {
                "alpha": alpha,
                "config": str(config_path.relative_to(ROOT)),
                "render_dir": str(render_dir.relative_to(ROOT)),
                "runtime_dir": str(runtime_dir.relative_to(ROOT)),
                "status": "not_started",
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "one-scene rendered alpha stability scan before fixed-alpha P2",
        "scene": SCENE,
        "seed": 0,
        "layer": layer,
        "rank": rank,
        "worst_round18_drift_shot": worst_shot,
        "boundaries": boundaries,
        "numeric_prefilter": str(numeric_path.relative_to(ROOT)),
        "runs": records,
    }
    output = pilot_root / "p2/alpha_scan/render_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
