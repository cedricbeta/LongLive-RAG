#!/usr/bin/env python3
"""Reuse the selected rendered-alpha candidate as the final watchmaker capsule run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENE = "bearded_watchmaker_workshop"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_token(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def locate(folder: Path, suffix: str) -> Path:
    matches = sorted(folder.glob(suffix))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix} in {folder}, found {matches}")
    return matches[0]


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def comparable_edit(config: dict) -> dict:
    edit = dict(config["inference"]["capsule_edit"])
    edit.pop("output_dir", None)
    return edit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    alpha_scan = json.loads((pilot_root / "p2/alpha_scan.json").read_text(encoding="utf-8"))
    if not alpha_scan.get("gate_pass") or not alpha_scan.get("visual_review_complete"):
        raise SystemExit("rendered alpha gate is not complete")
    alpha = float(alpha_scan["selected_alpha"])
    token = alpha_token(alpha)
    source_root = pilot_root / f"p2/alpha_scan/alpha_{token}"
    source_video = locate(source_root / "renders", f"*-{SCENE}-*_regular.mp4")
    source_prompts = locate(source_root / "renders", f"*-{SCENE}-*_regular_prompts.txt")
    source_runtime = source_root / f"runtime/{SCENE}/seed0/manifest.json"
    if not source_runtime.exists():
        raise RuntimeError(f"missing alpha runtime manifest: {source_runtime}")

    manifest_path = pilot_root / "p2/p2_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = next(
        record
        for record in manifest["runs"]
        if record["arm"] == "capsule" and record["scene"] == SCENE
    )
    source_config = yaml.safe_load(
        (pilot_root / f"p2/alpha_scan/configs/alpha_{token}.yaml").read_text(encoding="utf-8")
    )
    final_config = yaml.safe_load((ROOT / run["config"]).read_text(encoding="utf-8"))
    if comparable_edit(source_config) != comparable_edit(final_config):
        raise RuntimeError("selected alpha and final capsule intervention configs differ")
    for key in ("data_path", "sample_names", "seed", "num_output_frames"):
        if source_config.get(key) != final_config.get(key):
            raise RuntimeError(f"alpha/final config mismatch at {key}")

    final_render_dir = ROOT / run["render_dir"]
    target_stem = f"p2_capsule-rank0-{SCENE}-seed0_regular"
    target_video = final_render_dir / f"{target_stem}.mp4"
    target_prompts = final_render_dir / f"{target_stem}_prompts.txt"
    link_or_copy(source_video, target_video)
    link_or_copy(source_prompts, target_prompts)
    if sha256(source_video) != sha256(target_video):
        raise RuntimeError("promoted video hash mismatch")

    runtime_payload = json.loads(source_runtime.read_text(encoding="utf-8"))
    runtime_payload["promotion"] = {
        "source": str(source_runtime.relative_to(ROOT)),
        "source_video": str(source_video.relative_to(ROOT)),
        "source_video_sha256": sha256(source_video),
        "reason": "selected rendered-alpha candidate is byte-reused as the identical final capsule run",
    }
    target_runtime = ROOT / run["runtime_dir"] / SCENE / "seed0/manifest.json"
    target_runtime.parent.mkdir(parents=True, exist_ok=True)
    target_runtime.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True), encoding="utf-8")

    run["status"] = "complete_reused_from_rendered_alpha_scan"
    run["render"] = str(target_video.relative_to(ROOT))
    run["runtime_manifest"] = str(target_runtime.relative_to(ROOT))
    run["promotion"] = runtime_payload["promotion"]
    manifest["actual_new_renders_after_alpha_scan"] = 11
    manifest["promoted_final_runs"] = 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"render": str(target_video.relative_to(ROOT)), "sha256": sha256(target_video)}, indent=2))


if __name__ == "__main__":
    main()
