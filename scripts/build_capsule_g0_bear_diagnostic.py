#!/usr/bin/env python3
"""Build the v2 G0 brown-bear probe-off diagnostic config and ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "videos/capsule_pilot_20260715"
SOURCE = PILOT / "configs/probe_seed0__brown_bear_river.yaml"
STAGE_A_SOURCE = (
    ROOT / "videos/benchmark_hard_multishot/stage_A/configs/baseline_seed0.yaml"
)
OUTPUT = PILOT / "configs/g0_bear_probe_off_seed0.yaml"
RENDER_DIR = PILOT / "g0/bear_probe_off/renders"
MANIFEST = PILOT / "g0/bear_diagnostic_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_payload = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    config = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    render_dir = str(RENDER_DIR.relative_to(ROOT))
    config["output_folder"] = render_dir
    inference = config["inference"]
    inference["output_folder"] = render_dir
    inference["filename_prefix"] = "g0_bear_probe_off_seed0"
    inference["capsule_probe"]["enabled"] = False
    inference["capsule_edit"]["enabled"] = False
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8"
    )

    allowed_differences = {
        "output_folder",
        "inference.output_folder",
        "inference.filename_prefix",
        "inference.capsule_probe.enabled",
    }

    def flatten(value, prefix=""):
        out = {}
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                out.update(flatten(child, name))
        else:
            out[prefix] = value
        return out

    source_flat = flatten(source_payload)
    config_flat = flatten(config)
    differences = {
        key: {"probe_on": source_flat.get(key), "probe_off": config_flat.get(key)}
        for key in sorted(set(source_flat) | set(config_flat))
        if source_flat.get(key) != config_flat.get(key)
    }
    unexpected = sorted(set(differences) - allowed_differences)
    if unexpected:
        raise RuntimeError(f"unexpected probe-off config differences: {unexpected}")

    manifest = {
        "schema_version": 2,
        "stage": "G0_bear_probe_off_diagnostic",
        "scene": "brown_bear_river",
        "seed": 0,
        "source_probe_on_config": str(SOURCE.relative_to(ROOT)),
        "source_probe_on_config_sha256": sha256(SOURCE),
        "stage_a_source_config": str(STAGE_A_SOURCE.relative_to(ROOT)),
        "stage_a_source_config_sha256": sha256(STAGE_A_SOURCE),
        "probe_off_config": str(OUTPUT.relative_to(ROOT)),
        "probe_off_config_sha256": sha256(OUTPUT),
        "render_dir": render_dir,
        "differences_from_probe_on": differences,
        "command": (
            "CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_ALLOC_CONF=expandable_segments:True "
            "/home/chendong/miniconda3/bin/python inference.py "
            f"--config_path {OUTPUT.relative_to(ROOT)}"
        ),
        "status": "not_started",
    }
    prompt_pairs = []
    prompt_source = ROOT / "example/benchmark_hard_multishot/brown_bear_river"
    stage_a_snapshot = (
        ROOT
        / "videos/benchmark_hard_multishot/stage_A/prompt_subset/brown_bear_river"
    )
    for source_path in sorted(prompt_source.iterdir()):
        if not source_path.is_file():
            continue
        snapshot_path = stage_a_snapshot / source_path.name
        if not snapshot_path.exists():
            raise RuntimeError(f"missing stage_A prompt snapshot: {snapshot_path}")
        source_hash = sha256(source_path)
        snapshot_hash = sha256(snapshot_path)
        prompt_pairs.append(
            {
                "source": str(source_path.relative_to(ROOT)),
                "source_sha256": source_hash,
                "stage_a_snapshot": str(snapshot_path.relative_to(ROOT)),
                "stage_a_snapshot_sha256": snapshot_hash,
                "identical": source_hash == snapshot_hash,
            }
        )
    if not prompt_pairs or not all(record["identical"] for record in prompt_pairs):
        raise RuntimeError("brown-bear prompt source and stage_A snapshot differ")
    manifest["prompt_snapshot_equivalence"] = prompt_pairs
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
