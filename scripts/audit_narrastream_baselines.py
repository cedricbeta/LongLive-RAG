#!/usr/bin/env python3
"""Compare archived A/C baseline configs, authored inputs and decoded pixels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixels(path):
    result = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0",
        "-an", "-pix_fmt", "rgb24", "-threads", "1", "-f", "hash", "-hash", "sha256", "-"],
        check=True, capture_output=True, text=True)
    return result.stdout.strip()


def render_settings(config):
    inf = copy.deepcopy(config["inference"])
    for key in ("output_folder", "filename_prefix", "filename_from_sample_name"):
        inf.pop(key, None)
    return {"inference": inf, **{k: config.get(k) for k in ("model_kwargs", "checkpoints", "logging", "use_ema", "num_samples", "num_output_frames", "save_latents_only")},
            "shape": config["data"]["image_or_video_shape"]}


def audit_scene(root, scene):
    a_cfg_path = root / "configs" / f"ns3arm__{scene}__armA_baseline.yaml"
    c_configs = list((root / "armC_entity").glob(f"*/{scene}/configs/baseline.yaml"))
    if len(c_configs) != 1:
        return {"scene": scene, "error": f"expected one non-partial C baseline config, found {c_configs}"}
    c_cfg_path = c_configs[0]
    a_cfg = yaml.safe_load(a_cfg_path.read_text())
    c_cfg = yaml.safe_load(c_cfg_path.read_text())
    a_videos = list((root / "renders" / f"{scene}__armA_baseline").glob("*.mp4"))
    c_videos = list((c_cfg_path.parent.parent / "renders").glob("baseline-*.mp4"))
    if len(a_videos) != 1 or len(c_videos) != 1:
        return {"scene": scene, "error": f"expected one baseline per arm: A={a_videos}, C={c_videos}"}
    a, c = a_videos[0], c_videos[0]
    a_sha, c_sha = sha(a), sha(c)
    byte_equal = a_sha == c_sha
    pixel_a, pixel_c = (None, None) if byte_equal else (pixels(a), pixels(c))
    a_data = ROOT / a_cfg["data"]["data_path"] / scene
    c_data = ROOT / c_cfg["data"]["data_path"] / scene
    names = sorted({p.name for p in a_data.glob("*.json")} | {p.name for p in c_data.glob("*.json")} | {"shot_durations.txt"})
    input_checks = {name: (a_data / name).is_file() and (c_data / name).is_file() and sha(a_data / name) == sha(c_data / name) for name in names}
    return {"scene": scene, "a_video": str(a), "c_video": str(c), "a_sha256": a_sha, "c_sha256": c_sha,
            "a_config_sha256": sha(a_cfg_path), "c_config_sha256": sha(c_cfg_path),
            "a_kv_rag_enabled": a_cfg["inference"]["kv_rag"]["enabled"],
            "c_kv_rag_enabled": c_cfg["inference"]["kv_rag"]["enabled"],
            "render_settings_equal": render_settings(a_cfg) == render_settings(c_cfg),
            "authored_inputs_equal": all(input_checks.values()), "input_checks": input_checks,
            "file_bytes_equal": byte_equal, "decoded_pixels_equal": byte_equal or pixel_a == pixel_c,
            "decoded_a_hash": pixel_a, "decoded_c_hash": pixel_c}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "videos/narrastream_threearm_20260827")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    scenes = json.loads((args.root / "manifest.json").read_text())["scenes"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda s: audit_scene(args.root, s), scenes))
    summary = {"scenes": len(records), "errors": sum("error" in r for r in records),
        "c_kv_rag_enabled": sum(r.get("c_kv_rag_enabled", False) for r in records),
        **{k: sum(r.get(k, False) for r in records) for k in ("render_settings_equal", "authored_inputs_equal", "file_bytes_equal", "decoded_pixels_equal")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
