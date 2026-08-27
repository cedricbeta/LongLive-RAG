#!/usr/bin/env python3
"""Build the four preregistered full-vector causal-channel mini-test arms."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import estimate_capsule_subspace as g1


SCENES = {
    "sandy_beach_driftwood": ["driftwood"],
    "bearded_watchmaker_workshop": ["watchmaker"],
}
VARIANTS = {
    "layer14": [14],
    "all5": [0, 7, 14, 21, 29],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    g0 = json.loads((pilot_root / "g0/g0_results.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (pilot_root / "probe_replay_audit.json").read_text(encoding="utf-8")
    )
    if not g0.get("gate_pass"):
        raise SystemExit("channel test requires passing G0 transparency")
    if not (
        audit.get("passed")
        and audit.get("all_complete")
        and audit.get("frozen_inputs_unchanged")
    ):
        raise SystemExit("channel test requires the complete frozen-input probe audit")

    dump_root = pilot_root / "probe_replays"
    artifacts_dir = pilot_root / "channel_test/artifacts"
    configs_dir = pilot_root / "channel_test/configs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for scene, subject_phrases in SCENES.items():
        per_layer = {}
        for layer in g1.LAYERS:
            per_layer[str(layer)] = {
                "layer": int(layer),
                "rank": 1,
                "donor": g1.donor_for_scene(dump_root, scene, layer).float(),
            }
        artifact_path = artifacts_dir / f"{scene}.pt"
        torch.save(
            {
                "schema_version": 1,
                "scene": scene,
                "projection": "identity_full_vector",
                "donor_source": "seed0 shot0 clean-recache entity-token mean",
                "per_layer": per_layer,
            },
            artifact_path,
        )
        base_path = pilot_root / f"configs/probe_seed0__{scene}.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        subject_stream = g0["scenes"][scene]["reference_per_shot"][
            "subject_anchor_to_shot0"
        ]
        worst_shot = int(np.argmin(np.asarray(subject_stream[1:])) + 1)
        boundaries = list(range(worst_shot, 8))
        mask_path = pilot_root / f"clipseg_masks/seed0/{scene}.pt"
        mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        mask_method = (
            "clipseg_precomputed_with_manual_visibility_overrides"
            if mask_payload.get("metadata", {}).get("manual_visibility_exclusions")
            else "clipseg_precomputed"
        )
        for variant, layers in VARIANTS.items():
            arm_id = f"{scene}__{variant}"
            config = deepcopy(base)
            render_dir = pilot_root / f"channel_test/renders/{arm_id}"
            runtime_dir = pilot_root / f"channel_test/runtime/{arm_id}"
            inference = config.setdefault("inference", {})
            config["output_folder"] = str(render_dir.relative_to(ROOT))
            inference["output_folder"] = str(render_dir.relative_to(ROOT))
            inference["filename_prefix"] = f"channel_{variant}"
            inference["filename_from_sample_name"] = True
            inference["kv_rag"] = {"enabled": False}
            inference["capsule_probe"] = {"enabled": False}
            inference["capsule_edit"] = {
                "enabled": True,
                "output_dir": str(runtime_dir.relative_to(ROOT)),
                "arm": "full_vector",
                "artifact_path": str(artifact_path.relative_to(ROOT)),
                "layer": int(layers[0]),
                "layers": [int(layer) for layer in layers],
                "rank": 1,
                "alpha": 1.0,
                "boundaries": boundaries,
                "random_seed": 20260716,
                "mask_method": mask_method,
                "mask_path": str(mask_path.relative_to(ROOT)),
                "subject_phrases": subject_phrases,
                "allow_missing_subject": scene == "sandy_beach_driftwood",
                "fail_open": False,
            }
            config_path = configs_dir / f"channel__{arm_id}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, width=1000),
                encoding="utf-8",
            )
            records.append(
                {
                    "arm_id": arm_id,
                    "scene": scene,
                    "variant": variant,
                    "layers": layers,
                    "alpha": 1.0,
                    "fallback_alpha": 0.5,
                    "seed": 0,
                    "config": str(config_path.relative_to(ROOT)),
                    "artifact": str(artifact_path.relative_to(ROOT)),
                    "render_dir": str(render_dir.relative_to(ROOT)),
                    "runtime_dir": str(runtime_dir.relative_to(ROOT)),
                    "strongest_drift_shot": worst_shot,
                    "boundaries": boundaries,
                    "baseline_reference": g0["scenes"][scene]["reference"],
                    "status": "not_started",
                }
            )
    manifest = {
        "schema_version": 1,
        "track": "causal_channel_mini_test",
        "preregistered_arms": 4,
        "planned_renders": len(records),
        "scenes": list(SCENES),
        "variants": VARIANTS,
        "projection": "P=I full-vector V edit",
        "edit_rule": "v' = v + alpha * (mean_v_donor - v)",
        "donor_rule": "same-scene seed0 shot0 entity-token pooled clean-recache V",
        "identity_metric_effect_threshold_abs": 0.01,
        "reachability_rule": "decoded frames after first edited boundary differ from paired baseline",
        "alpha_fallback_rule": "if alpha=1.0 itself causes collapse, rerun that arm once at alpha=0.5 and ledger it",
        "guard_interpretation": "composition or motion leakage is causal evidence, not a failure",
        "runs": records,
    }
    output = pilot_root / "channel_test/channel_run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    verdict = pilot_root / "channel_test/human_verdict.json"
    if not verdict.exists():
        verdict.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": False,
                    "reviewer": None,
                    "arms": {
                        record["arm_id"]: {
                            "visible_subject_appearance_change": None,
                            "only_trivial_perturbation": None,
                            "evidence": None,
                        }
                        for record in records
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"planned_renders": len(records), "manifest": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
