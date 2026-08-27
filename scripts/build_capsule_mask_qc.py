#!/usr/bin/env python3
"""Build the mandatory per-shot cross-attention mask QC sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)


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


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    frames = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
        index += 1
    capture.release()
    if set(frames) != wanted:
        raise RuntimeError(f"missing video frames {sorted(wanted - set(frames))}")
    return frames


def load_mask_index(mask_dir: Path) -> dict[int, dict]:
    indexed = {}
    for path in sorted(mask_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for local_index, frame in enumerate(payload["frame_indices"].tolist()):
            if int(frame) in indexed:
                raise RuntimeError(f"duplicate mask for global frame {frame} in {mask_dir}")
            indexed[int(frame)] = {
                "binary": payload["binary"][local_index].numpy().astype(bool),
                "soft": payload.get("soft", torch.zeros_like(payload["binary"], dtype=torch.float32))[
                    local_index
                ].float().numpy(),
                "shot_index": int(payload["shot_index"]),
                "prompt": str(payload.get("prompt") or ""),
                "subject_phrases": list(payload.get("subject_phrases") or []),
                "source": str(path.resolve().relative_to(ROOT)),
            }
    return indexed


def load_precomputed_index(path: Path, scene: str) -> dict[int, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    binary = payload["binary"].to(torch.bool)
    soft = payload.get("soft", binary).float()
    metadata = payload.get("metadata", {})
    absent_shots = set(metadata.get("explicit_absent_identity_shots") or [])
    exclusions = metadata.get("manual_visibility_exclusions") or []
    ranges = shot_ranges(scene)
    indexed = {}
    for frame in range(binary.shape[0]):
        shot = next(
            shot_index
            for shot_index, (start, end) in enumerate(ranges)
            if start <= frame < end
        )
        indexed[frame] = {
            "binary": binary[frame].numpy(),
            "soft": soft[frame].numpy(),
            "shot_index": shot,
            "prompt": str(
                (metadata.get("shot_queries") or [metadata.get("query") or ""] * 8)[shot]
            ),
            "subject_phrases": [
                str((metadata.get("shot_queries") or [metadata.get("query") or ""] * 8)[shot])
            ],
            "source": str(path.resolve().relative_to(ROOT)),
            "empty_reason": (
                "identity absent from frozen prompt/video"
                if shot in absent_shots
                else next(
                    (
                        str(exclusion.get("reason") or "manual visibility exclusion")
                        for exclusion in exclusions
                        if int(exclusion["frame_range"][0])
                        <= frame
                        < int(exclusion["frame_range"][1])
                    ),
                    None,
                )
            ),
        }
    return indexed


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    up = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    result = frame.copy()
    tint = np.zeros_like(result)
    tint[..., 1] = 255
    selected = up.astype(bool)
    if bool(selected.any()):
        result[selected] = cv2.addWeighted(
            result[selected], 0.55, tint[selected], 0.45, 0
        )
    contours, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 3)
    return result


def text_box(image: np.ndarray, lines: list[str], *, empty: bool) -> None:
    color = (40, 40, 230) if empty else (20, 180, 20)
    height = 25 + 28 * len(lines)
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (15, 15, 15), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, 27 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color if index == len(lines) - 1 else (240, 240, 240),
            2,
            cv2.LINE_AA,
        )


def build_scene(
    pilot_root: Path,
    scene: str,
    seed: int,
    mask_source: str,
    human_qc: str,
) -> dict:
    precomputed_metadata = {}
    if mask_source == "precomputed":
        render_dir = ROOT / f"videos/benchmark_hard_multishot/stage_A/baseline_seed{seed}"
    else:
        render_dir = pilot_root / f"probe_replays/seed{seed}/renders"
    videos = sorted(render_dir.glob(f"*-{scene}-*_regular.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"expected one replay video for {scene}, found {videos}")
    ranges = shot_ranges(scene)
    indices = [start + (end - start) // 2 for start, end in ranges]
    frames = read_frames(videos[0], indices)
    if mask_source == "precomputed":
        precomputed_path = pilot_root / f"clipseg_masks/seed{seed}" / f"{scene}.pt"
        masks = load_precomputed_index(precomputed_path, scene)
        precomputed_metadata = torch.load(
            precomputed_path, map_location="cpu", weights_only=False
        ).get("metadata", {})
        method_label = (
            "CLIPSeg + manual visibility exclusions"
            if precomputed_metadata.get("manual_visibility_exclusions")
            else "transformers CLIPSeg fallback"
        )
    else:
        mask_dir = (
            pilot_root
            / f"probe_replays/seed{seed}/dumps"
            / scene
            / f"seed{seed}/masks"
        )
        masks = load_mask_index(mask_dir)
        method_label = "runtime mask dump"
    records = []
    cells = []
    for shot, frame_index in enumerate(indices):
        if frame_index not in masks:
            raise RuntimeError(f"missing mask for {scene} shot {shot} frame {frame_index}")
        record = masks[frame_index]
        if record["shot_index"] != shot:
            raise RuntimeError(
                f"mask shot mismatch for {scene} frame {frame_index}: "
                f"{record['shot_index']} != {shot}"
            )
        mask = record["binary"]
        coverage = float(mask.mean())
        empty = not bool(mask.any())
        component_count = max(
            0, cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)[0] - 1
        )
        visual = overlay(frames[frame_index], mask)
        visual = cv2.resize(visual, (480, 264), interpolation=cv2.INTER_AREA)
        status = (
            f"EMPTY: {record.get('empty_reason') or 'no valid identity tokens'}"
            if empty
            else f"coverage={coverage:.1%}, components={component_count}"
        )
        text_box(
            visual,
            [f"shot {shot} | frame {frame_index}", method_label + ": " + status],
            empty=empty,
        )
        cells.append(visual)
        records.append(
            {
                "shot": shot,
                "frame": frame_index,
                "coverage": coverage,
                "components": component_count,
                "empty": empty,
                "empty_reason": record.get("empty_reason"),
                "subject_phrases": record["subject_phrases"],
                "prompt": record["prompt"],
                "mask_source": record["source"],
            }
        )
    rows = [np.concatenate(cells[index : index + 4], axis=1) for index in (0, 4)]
    sheet = np.concatenate(rows, axis=0)
    title = np.full((54, sheet.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        title,
        f"{scene} | seed{seed} | {method_label} identity mask | green fill + cyan contour",
        (18, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    sheet = np.concatenate([title, sheet], axis=0)
    output_dir = pilot_root / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{scene}_mask_qc.png" if seed == 0 else f"{scene}_seed{seed}_mask_qc.png"
    )
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"failed to write {output_path}")
    return {
        "scene": scene,
        "seed": seed,
        "method": (
            str(
                precomputed_metadata.get(
                    "method", "transformers_CLIPSeg_precomputed"
                )
            )
            if mask_source == "precomputed"
            else "runtime_mask_dump"
        ),
        "pixel_to_token_mapping": (
            "22x40 token masks are nearest-neighbor expanded to 704x1280; "
            "each token corresponds to one 32x32 decoded-pixel cell"
        ),
        "image": str(output_path.relative_to(ROOT)),
        "shots": records,
        "human_qc": human_qc,
        "human_qc_note": (
            "Accepted after visual inspection of all eight overlays; masks follow the "
            "visible identity or identity-bearing body part. Empty regions use the "
            "per-frame provenance reason; driftwood shot 2 is absent, while seed1/2 "
            "driftwood visibility exclusions prevent known coastline/rock/fish/person "
            "false positives."
            if human_qc == "accepted"
            else "Awaiting visual inspection of all eight shot overlays."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot_root", default="videos/capsule_pilot_20260715"
    )
    parser.add_argument(
        "--mask_source",
        choices=("precomputed", "probe_dump"),
        default="probe_dump",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--human_qc",
        choices=("pending", "accepted", "rejected"),
        default="pending",
    )
    args = parser.parse_args()
    if any(seed not in (0, 1, 2) for seed in args.seeds):
        raise ValueError("seeds must be a subset of {0,1,2}")
    pilot_root = (ROOT / args.pilot_root).resolve()
    result = {
        "schema_version": 1,
        "mask_priority": ["cross_attention", "CLIPSeg", "manual_box"],
        "selected_method": (
            (
                "CLIPSeg_accepted"
                if args.human_qc == "accepted"
                else f"CLIPSeg_{args.human_qc}_human_qc"
            )
            if args.mask_source == "precomputed"
            else f"runtime_dump_{args.human_qc}_human_qc"
        ),
        "scenes": [
            build_scene(pilot_root, scene, seed, args.mask_source, args.human_qc)
            for seed in args.seeds
            for scene in SCENES
        ],
    }
    output_path = pilot_root / "masks/mask_qc_stats.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
