#!/usr/bin/env python3
"""Build reproducible probe configs for the typed-memory capsule pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml
import torch


ROOT = Path(__file__).resolve().parents[1]
DATE = "20260715"
SCENES = {
    "bearded_watchmaker_workshop": {
        "subject_phrases": ["watchmaker"],
        "allow_missing_subject": False,
    },
    "tattooed_noodle_chef": {
        "subject_phrases": ["chef"],
        "allow_missing_subject": False,
    },
    "brown_bear_river": {
        "subject_phrases": ["bear"],
        "allow_missing_subject": False,
    },
    "sandy_beach_driftwood": {
        "subject_phrases": ["driftwood"],
        # Shot 2 does not contain an identity word. Preserve baseline prompts and
        # record an explicit empty mask rather than silently masking another noun.
        "allow_missing_subject": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(seed: int, scene: str, pilot_root: Path) -> tuple[Path, dict]:
    base_path = ROOT / "videos/benchmark_hard_multishot/stage_A/configs" / f"baseline_seed{seed}.yaml"
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["data"]["data_path"] = "example/benchmark_hard_multishot"
    config["data_path"] = "example/benchmark_hard_multishot"
    config["sample_names"] = [scene]
    render_dir = pilot_root / "probe_replays" / f"seed{seed}" / "renders"
    dump_dir = pilot_root / "probe_replays" / f"seed{seed}" / "dumps"
    config["output_folder"] = str(render_dir.relative_to(ROOT))
    inference = config.setdefault("inference", {})
    inference["output_folder"] = str(render_dir.relative_to(ROOT))
    inference["filename_prefix"] = f"probe_seed{seed}"
    inference["filename_from_sample_name"] = True
    inference["kv_rag"] = {"enabled": False}
    mask_path = pilot_root / "clipseg_masks" / f"seed{seed}" / f"{scene}.pt"
    mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    mask_metadata = mask_payload.get("metadata", {})
    mask_method = (
        "clipseg_precomputed_with_manual_visibility_overrides"
        if mask_metadata.get("manual_visibility_exclusions")
        else "clipseg_precomputed"
    )
    inference["capsule_probe"] = {
        "enabled": True,
        "output_dir": str(dump_dir.relative_to(ROOT)),
        "layers": [0, 7, 14, 21, 29],
        "frame_stride": 4,
        "store_dtype": "float16",
        # Cross-attention was rejected by the mandatory seed0 overlay QC. The
        # second-priority CLIPSeg fallback is precomputed from each frozen
        # stage-A replay and has its own accepted QC sheet.
        "mask_method": mask_method,
        "mask_path": str(mask_path.relative_to(ROOT)),
        "mask_layers": [14],
        "mask_max_heads": 4,
        "mask_top_fraction": 0.12,
        "mask_min_relative": 0.25,
        "mask_largest_component": False,
        "mask_dilation": 0,
        "subject_phrases": SCENES[scene]["subject_phrases"],
        "allow_missing_subject": SCENES[scene]["allow_missing_subject"],
        "save_soft_masks": True,
        "fail_open": False,
    }
    inference["capsule_edit"] = {"enabled": False}
    config_path = pilot_root / "configs" / f"probe_seed{seed}__{scene}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8"
    )
    record = {
        "stage": "P0" if seed == 0 else "P1_dump_replay",
        "seed": seed,
        "scene": scene,
        "config": str(config_path.relative_to(ROOT)),
        "render_dir": str(render_dir.relative_to(ROOT)),
        "dump_dir": str(dump_dir.relative_to(ROOT)),
        "command": (
            "CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_ALLOC_CONF=expandable_segments:True "
            f"{Path('/home/chendong/miniconda3/bin/python')} inference.py "
            f"--config_path {config_path.relative_to(ROOT)}"
        ),
        "status": "not_started",
    }
    return config_path, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot_root",
        default=f"videos/capsule_pilot_{DATE}",
        help="Pilot intermediate/output root relative to repository",
    )
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    pilot_root.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in (0, 1, 2):
        for scene in SCENES:
            _, record = build_config(seed, scene, pilot_root)
            records.append(record)

    frozen_files: list[Path] = []
    for scene in SCENES:
        frozen_files.extend(sorted((ROOT / "example/benchmark_hard_multishot" / scene).glob("*")))
    for seed in (0, 1, 2):
        folder = ROOT / "videos/benchmark_hard_multishot/stage_A" / f"baseline_seed{seed}"
        for scene in SCENES:
            frozen_files.extend(sorted(folder.glob(f"*-{scene}-*_regular.mp4")))
    frozen_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in frozen_files
        if path.is_file()
    }
    (pilot_root / "frozen_inputs_sha256.json").write_text(
        json.dumps(frozen_hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "date": DATE,
        "python": "/home/chendong/miniconda3/bin/python",
        "probe_layers": [0, 7, 14, 21, 29],
        "mask_priority": ["cross_attention", "CLIPSeg", "manual_box"],
        "selected_mask_method": "clipseg_precomputed_after_cross_attention_qc_rejection",
        "mask_qc": "videos/capsule_pilot_20260715/masks/mask_qc_stats.json",
        "render_budget": {
            "probe_replays": 12,
            "rendered_alpha_candidates_if_g1_passes": 3,
            "final_arm_scene_cells": 12,
            "selected_alpha_candidate_reused_in_final_cells": 1,
            "new_final_renders_after_promotion": 11,
            "unique_p2_renders_if_g1_passes": 14,
        },
        "runs": records,
    }
    (pilot_root / "probe_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {len(records)} configs and manifest under {pilot_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
