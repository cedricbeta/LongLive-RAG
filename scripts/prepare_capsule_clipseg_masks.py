#!/usr/bin/env python3
"""Precompute CLIPSeg fallback masks from the frozen stage-A videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as torch_f
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.capsule import entity_mask_from_attention


SCENES = {
    "bearded_watchmaker_workshop": {
        "query": "an elderly bearded watchmaker",
        "shot_queries": [
            "an elderly bearded man",
            "an elderly bearded man's face",
            "human hands",
            "an elderly bearded man",
            "an elderly bearded man",
            "human fingers",
            "an elderly man's head and shoulders",
            "an elderly bearded man",
        ],
        "identity_phrase": "watchmaker",
    },
    "tattooed_noodle_chef": {
        "query": "a tattooed male chef",
        "shot_queries": [
            "a tattooed male chef",
            "a man's face",
            "a tattooed human forearm",
            "tattooed human arms",
            "a bald man wearing an apron",
            "a tattooed human forearm",
            "a tattooed man's upper body and arms",
            "a bald male chef",
        ],
        "identity_phrase": "chef",
    },
    "brown_bear_river": {
        "query": "a brown bear",
        "shot_queries": ["a brown bear"] * 8,
        "identity_phrase": "bear",
    },
    "sandy_beach_driftwood": {
        "query": "weathered driftwood",
        "shot_queries": [
            "dead tree trunks and branches",
            "wooden logs",
            "weathered driftwood",
            "weathered wooden logs",
            "driftwood firewood",
            "driftwood firewood",
            "driftwood firewood",
            "wood logs in a campfire",
        ],
        "identity_phrase": "driftwood",
    },
}
MODEL_NAME = "CIDAS/clipseg-rd64-refined"


def manual_visibility_exclusions(seed: int, scene: str) -> list[dict]:
    """Conservative post-QC exclusions; never substitute a different object."""
    if scene != "sandy_beach_driftwood" or seed not in (1, 2):
        return []
    return [
        {
            "shot": 0,
            "frame_range": [48, 96],
            "reason": (
                "after the camera pan, driftwood is absent or too small for a "
                "22x40 token mask; CLIPSeg switches to coastline/sand"
            ),
        },
        {
            "shot": 1,
            "frame_range": [96, 192],
            "reason": (
                "driftwood is only a distant sub-token background detail; "
                "CLIPSeg localizes rock, fish, or person instead"
            ),
        },
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_video(seed: int, scene: str) -> Path:
    folder = ROOT / "videos/benchmark_hard_multishot/stage_A" / f"baseline_seed{seed}"
    matches = sorted(folder.glob(f"*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen video for seed{seed}/{scene}: {matches}")
    return matches[0]


def read_sampled_frames(path: Path, stride: int) -> tuple[list[int], list[Image.Image], int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    sampled_indices = []
    images = []
    last_rgb = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_rgb = rgb
        if frame_index % stride == 0:
            sampled_indices.append(frame_index)
            images.append(Image.fromarray(rgb))
        frame_index += 1
    capture.release()
    if frame_index == 0 or last_rgb is None:
        raise RuntimeError(f"decoded zero frames from {path}")
    if sampled_indices[-1] != frame_index - 1:
        sampled_indices.append(frame_index - 1)
        images.append(Image.fromarray(last_rgb))
    return sampled_indices, images, frame_index


def infer_soft_masks(
    images: list[Image.Image],
    queries: list[str],
    processor,
    model,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(
            text=queries[start : start + len(batch)],
            images=batch,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.sigmoid(logits.float()).unsqueeze(1)
            probabilities = torch_f.interpolate(
                probabilities,
                size=(22, 40),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        outputs.append(probabilities.cpu())
    return torch.cat(outputs)


def temporal_interpolate(
    sampled_indices: list[int],
    sampled: torch.Tensor,
    total_frames: int,
    ranges: list[tuple[int, int]],
) -> torch.Tensor:
    def shot_for(frame: int) -> int:
        return next(index for index, (start, end) in enumerate(ranges) if start <= frame < end)

    result = torch.empty(total_frames, sampled.shape[1], sampled.shape[2])
    for left_index in range(len(sampled_indices) - 1):
        start = sampled_indices[left_index]
        end = sampled_indices[left_index + 1]
        span = max(1, end - start)
        crosses_shot = shot_for(start) != shot_for(end)
        for frame in range(start, end):
            if crosses_shot:
                # Never blend masks across a hard shot boundary. The final few
                # unsampled frames inherit the last mask from their own shot.
                result[frame] = sampled[left_index]
            else:
                weight = (frame - start) / span
                result[frame] = sampled[left_index] * (1.0 - weight) + sampled[left_index + 1] * weight
    result[sampled_indices[-1]] = sampled[-1]
    return result


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(x)
        for x in (
            ROOT / "example/benchmark_hard_multishot" / scene / "shot_durations.txt"
        ).read_text().split()
    ]
    ranges = []
    start = 0
    for chunks in durations:
        end = start + chunks * 8
        ranges.append((start, end))
        start = end
    return ranges


def absent_identity_shots(scene: str) -> list[int]:
    phrase = SCENES[scene]["identity_phrase"]
    absent = []
    for shot, _ in enumerate(shot_ranges(scene)):
        payload = json.loads(
            (ROOT / "example/benchmark_hard_multishot" / scene / f"{shot}.json").read_text(
                encoding="utf-8"
            )
        )
        if phrase not in str(payload.get("caption", "")).lower():
            absent.append(shot)
    return absent


def build_one(
    seed: int,
    scene: str,
    output_root: Path,
    processor,
    model,
    *,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    source = locate_video(seed, scene)
    indices, images, total_frames = read_sampled_frames(source, stride)
    if total_frames != 480:
        raise RuntimeError(f"expected 480 frames in {source}, found {total_frames}")
    ranges = shot_ranges(scene)
    shot_queries = SCENES[scene]["shot_queries"]
    queries = [
        shot_queries[
            next(index for index, (start, end) in enumerate(ranges) if start <= frame < end)
        ]
        for frame in indices
    ]
    sampled = infer_soft_masks(
        images,
        queries,
        processor,
        model,
        batch_size=batch_size,
        device=device,
    )
    soft = temporal_interpolate(indices, sampled, total_frames, ranges)
    binary = entity_mask_from_attention(
        soft,
        top_fraction=0.25,
        min_relative=0.35,
        largest_component=True,
        dilation=1,
    )
    absent = absent_identity_shots(scene)
    for shot in absent:
        start, end = ranges[shot]
        soft[start:end] = 0.0
        binary[start:end] = False
    exclusions = manual_visibility_exclusions(seed, scene)
    for exclusion in exclusions:
        start, end = exclusion["frame_range"]
        soft[start:end] = 0.0
        binary[start:end] = False
    output_path = output_root / f"seed{seed}" / f"{scene}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "method": (
            "transformers_CLIPSeg_precomputed_with_manual_visibility_exclusions"
            if exclusions
            else "transformers_CLIPSeg_precomputed"
        ),
        "model": MODEL_NAME,
        "query": SCENES[scene]["query"],
        "shot_queries": shot_queries,
        "source_video": str(source.relative_to(ROOT)),
        "source_video_sha256": sha256(source),
        "source_seed": seed,
        "scene": scene,
        "sample_stride": stride,
        "temporal_fill": (
            "linear interpolation of soft masks within each shot; "
            "last in-shot mask held to hard boundary"
        ),
        "latent_grid": [22, 40],
        "threshold": {
            "top_fraction": 0.25,
            "min_relative": 0.35,
            "largest_component": True,
            "dilation": 1,
        },
        "explicit_absent_identity_shots": absent,
        "manual_visibility_exclusions": exclusions,
    }
    torch.save(
        {"binary": binary, "soft": soft.to(torch.float16), "metadata": metadata},
        output_path,
    )
    return {
        "seed": seed,
        "scene": scene,
        "path": str(output_path.relative_to(ROOT)),
        "mean_coverage": float(binary.float().mean().item()),
        "absent_identity_shots": absent,
        "manual_visibility_exclusions": exclusions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES), default=list(SCENES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--output_root",
        default="videos/capsule_pilot_20260715/clipseg_masks",
    )
    args = parser.parse_args()
    if any(seed not in (0, 1, 2) for seed in args.seeds):
        raise ValueError("seeds must be a subset of {0,1,2}")
    device = torch.device(args.device)
    processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME).to(device).eval()
    output_root = (ROOT / args.output_root).resolve()
    records = []
    for seed in args.seeds:
        for scene in args.scenes:
            record = build_one(
                seed,
                scene,
                output_root,
                processor,
                model,
                stride=max(1, args.stride),
                batch_size=max(1, args.batch_size),
                device=device,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    manifest_path = output_root / "manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"schema_version": 1, "model": MODEL_NAME, "records": []}
    )
    by_key = {
        (int(record["seed"]), record["scene"]): record
        for record in existing.get("records", [])
    }
    for record in records:
        by_key[(record["seed"], record["scene"])] = record
    existing["records"] = [by_key[key] for key in sorted(by_key)]
    manifest_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
