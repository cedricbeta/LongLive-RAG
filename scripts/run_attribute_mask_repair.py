#!/usr/bin/env python
"""Attribute-level masks versus whole-person masks on the five fixed repair cases.

Same baseline, same repair prompt, same noise seed and entry sigma as the 2026-09-06
machine repair run; the only variable is the repaint region: the whole SAM2 person
track (09-06) versus person ∩ CLIPSeg part masks for the attributes that actually
conflict with the script (hair / clothing, decided by the pixel colour check).

Per case the report measures, against the archived baseline:
  collateral   mean |pixel change| outside the person track on the target shot
  inside_off   mean |pixel change| inside the person track but outside the part mask
  attribute    pixel colour conflicts of the repaired crops with the script (0 = fixed)
  identity     DINO cosine between baseline and repaired crops (higher = same person)
  downstream   mean |pixel change| per later shot (how much the replay moved)
for both the 09-06 whole-person render (read from disk) and the new part render.
No automatic judge, no best-of, one render per case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.agent_appearance_color import (check_crop_against_spec, load_rgb, parse_appearance_spec, set_part_device)
from utils.agent_edit import FRAMES_PER_CHUNK, LATENT_GRID, clipseg_chunk_masks, read_video_frames, save_mask_payload
from utils.agent_runtime import Action, Verdict

PART_PHRASE = {"hair": "hair", "garment": "clothing"}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def decide_parts(case, spec, device) -> dict:
    """Which parts conflict with the script on the baseline target crops."""
    set_part_device(device)
    from PIL import Image
    conflicts = {"hair": 0, "garment": 0}
    checked = 0
    crops_dir = Path(case["_out"]) / "baseline_crops"
    for i, sample in enumerate(case["samples"]):
        box = sample["boxes"][sample["selected_id"]]
        path = crops_dir / f"chunk{sample['chunk']:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(sample["frame"]) as im:
            im.crop(tuple(box)).save(path)
        result = check_crop_against_spec(path, spec)
        checked += 1
        for region in conflicts:
            if result["checks"].get(region, {}).get("conflict"):
                conflicts[region] += 1
    parts = [r for r, n in conflicts.items() if n >= max(1, checked // 2)]
    if not parts:  # colour check found nothing decisive: repaint every scripted part
        parts = [r for r in ("hair", "garment") if spec.get(r, {}).get("colors")]
    return {"conflict_votes": conflicts, "crops_checked": checked, "parts": parts}


def dilate_latent(mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Binary dilation on the [F, h, w] latent grid, ``steps`` cells."""
    out = mask
    for _ in range(int(steps)):
        out = torch.nn.functional.max_pool2d(out[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
    return out


def part_masks_for_case(case, parts, device, out_path: Path, *, min_coverage: float = 0.0, max_dilate: int = 2) -> dict:
    """person ∩ CLIPSeg parts; a part smaller than ``min_coverage`` of the grid is dilated
    (within the person track) up to ``max_dilate`` cells.  ns090's hair covered 0.4-0.6 %
    of the grid and repainted into a white cloth; ns190's 1.1 % repainted cleanly."""
    person = torch.load(case["mask_path"], map_location="cpu", weights_only=False)["masks"]
    chunks = sorted(int(c) for c in person)
    union = {c: torch.zeros_like(person[c]) for c in chunks}
    for part in parts:
        soft = clipseg_chunk_masks(case["video"], chunks, PART_PHRASE[part], device=device)
        for c in chunks:
            union[c] = torch.maximum(union[c], soft[c])
    # never repaint the face: the dilated hair mask on ns090 produced a white
    # label pasted over the face in every chunk (2026-09-11)
    face = clipseg_chunk_masks(case["video"], chunks, "face", device=device)
    masks, dilations = {}, {}
    for c in chunks:
        part_bin = (union[c] >= 0.35).float()
        steps = 0
        while min_coverage > 0 and float((person[c] * part_bin >= 0.35).float().mean()) < min_coverage and steps < max_dilate:
            part_bin = dilate_latent(part_bin, 1)
            steps += 1
        part_bin = part_bin * (face[c] < 0.35).float()
        masks[c] = person[c] * part_bin
        dilations[str(c)] = steps
    save_mask_payload(masks, out_path, phrase=json.dumps({"person": case["entity"], "parts": parts}), video=case["video"])
    coverage = {str(c): float((masks[c] >= 0.35).float().mean()) for c in chunks}
    person_cov = {str(c): float((person[c] >= 0.35).float().mean()) for c in chunks}
    return {"mask_path": str(out_path), "coverage": coverage, "person_coverage": person_cov,
            "min_coverage": min_coverage, "dilation_steps": dilations}


def upsample(mask: torch.Tensor, hw) -> np.ndarray:
    m = torch.nn.functional.interpolate(mask[:, None].float(), size=tuple(hw), mode="bilinear", align_corners=False)[:, 0]
    return (m >= 0.35).numpy()


def frames_np(video, start, count):
    return [np.asarray(f, dtype=np.float32) for f in read_video_frames(video, start, count)]


def measure(case, baseline_video, edited_video, person_payload, part_payload, parts, spec, dino, out_dir: Path) -> dict:
    fpc = FRAMES_PER_CHUNK
    chunks = sorted(int(c) for c in person_payload)
    c0, c1 = chunks[0], chunks[-1]
    base = frames_np(baseline_video, c0 * fpc, (c1 - c0 + 1) * fpc)
    edit = frames_np(edited_video, c0 * fpc, (c1 - c0 + 1) * fpc)
    hw = base[0].shape[:2]
    outside, inside_off, inside_part = [], [], []
    for c in chunks:
        pm = upsample(person_payload[c], hw)
        qm = upsample(part_payload[c], hw) if part_payload is not None else pm
        for j in range(fpc):
            diff = np.abs(edit[(c - c0) * fpc + j] - base[(c - c0) * fpc + j]).mean(-1)
            p, q = pm[j], qm[j]
            outside.append(float(diff[~p].mean()) if (~p).any() else 0.0)
            inside_off.append(float(diff[p & ~q].mean()) if (p & ~q).any() else 0.0)
            inside_part.append(float(diff[q].mean()) if q.any() else 0.0)
    from PIL import Image
    attr, identity = [], []
    for i, sample in enumerate(case["samples"]):
        box = tuple(sample["boxes"][sample["selected_id"]])
        frame_index = sample["chunk"] * fpc
        cur = Image.fromarray(edit[frame_index - c0 * fpc].astype(np.uint8)).crop(box)
        ref = Image.fromarray(base[frame_index - c0 * fpc].astype(np.uint8)).crop(box)
        path = out_dir / "crops" / f"chunk{sample['chunk']:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cur.save(path)
        result = check_crop_against_spec(path, spec)
        attr.append({region: result["checks"].get(region, {}).get("conflict") for region in parts})
        if dino is not None:
            feats = dino([np.asarray(ref.convert("RGB")), np.asarray(cur.convert("RGB"))])
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            identity.append(float(feats[0] @ feats[1]))
    # downstream shots
    total = int(case["frame_count"])
    downstream = {}
    for shot in range(1, total // (fpc * 8)):
        start = shot * 8 * fpc
        b = frames_np(baseline_video, start, 8 * fpc)
        e = frames_np(edited_video, start, 8 * fpc)
        n = min(len(b), len(e))
        downstream[str(shot)] = float(np.mean([np.abs(e[i] - b[i]).mean() for i in range(n)])) if n else None
    return {
        "collateral_outside_person": round(float(np.mean(outside)), 3),
        "inside_person_outside_part": round(float(np.mean(inside_off)), 3),
        "inside_part": round(float(np.mean(inside_part)), 3),
        "attribute_conflicts_per_crop": attr,
        "attribute_fixed_crops": sum(1 for a in attr if not any(a.values())),
        "dino_identity_cosine": [round(v, 3) for v in identity],
        "downstream_shot_mae": downstream,
    }


def run_case(packet_path: Path, machine_root: Path, output: Path, case_id: str, gpu: int, aux_device: str,
             *, min_coverage: float = 0.0, max_dilate: int = 2) -> dict:
    from scripts.run_agent_edit_demo import EditRenderBackend
    packet = json.loads(packet_path.read_text())
    case, = [c for c in packet["cases"] if c["id"] == case_id]
    case_dir = output / case_id
    case["_out"] = str(case_dir)
    started = case_dir / "started.json"
    if (case_dir / "report.json").exists():
        raise FileExistsError(f"{case_id} already has a report; choose a fresh output")
    (case_dir / "failure.json").unlink(missing_ok=True)
    write_json(started, {"case": case_id, "gpu": gpu, "time": time.time(), "packet": str(packet_path)})
    spec = parse_appearance_spec(case["spec"]["description"])
    decision = decide_parts(case, spec, aux_device)
    part_info = part_masks_for_case(case, decision["parts"], aux_device, case_dir / "part_masks.pt",
                                    min_coverage=min_coverage, max_dilate=max_dilate)
    write_json(case_dir / "parts.json", {"spec": spec, **decision, **part_info})

    machine_dir = machine_root / case_id
    scene_dir = Path(os.path.realpath(machine_dir / "data" / case["scene"]))
    # MultiTextConcatDataset renders one sample per scene folder under <case_dir>/data
    data_link = case_dir / "data" / case["scene"]
    data_link.parent.mkdir(parents=True, exist_ok=True)
    if not data_link.exists():
        data_link.symlink_to(scene_dir)
    backend = EditRenderBackend(scene=case["scene"], scene_dir=scene_dir, out_dir=case_dir,
                                base_config=Path(case["baseline_config"]), baseline_video=Path(case["video"]),
                                render_gpu=gpu, clipseg_device="cpu", noise_seed=packet["protocol"]["noise_seed"] - 1)
    action = Action("edit_shot", {"shot": 0, "mask_path": part_info["mask_path"],
                    "entry_sigma": packet["protocol"]["entry_sigma"], "prompt": case["repair_prompt"]},
                    reason="attribute_part_mask_repair").normalized(shot_index=0)
    _, result = backend.edit(0, 1, action, Verdict(False, "fixed_target_attribute_mask"))
    edited = result["edited_video"]

    from evaluation.vbench_consistency import build_dino_encoder
    dino = build_dino_encoder(device=aux_device)
    person_payload = torch.load(case["mask_path"], map_location="cpu", weights_only=False)["masks"]
    part_payload = torch.load(part_info["mask_path"], map_location="cpu", weights_only=False)["masks"]
    machine_result = json.loads((machine_dir / "result.json").read_text())["result"]
    report = {
        "case": case_id, "entity": case["entity"], "parts": decision["parts"],
        "whole_person_0906": measure(case, case["video"], machine_result["edited_video"], person_payload, None, decision["parts"], spec, dino, case_dir / "eval_whole"),
        "part_mask": measure(case, case["video"], edited, person_payload, part_payload, decision["parts"], spec, dino, case_dir / "eval_part"),
        "edited_video": edited, "whole_person_video": machine_result["edited_video"],
        "render": {k: result.get(k) for k in ("config", "edited_video_sha256", "prefix_check")},
    }
    write_json(case_dir / "report.json", report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=ROOT / "videos/machine_entity_repair_20260906/packet.json")
    ap.add_argument("--machine-root", type=Path, default=ROOT / "videos/machine_entity_repair_20260906")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--aux-device", default="cuda:0", help="CLIPSeg / DINO device (absolute index, CUDA_VISIBLE_DEVICES unset)")
    ap.add_argument("--min-part-coverage", type=float, default=0.0, help="dilate a part mask (within the person) until it covers this share of the latent grid")
    ap.add_argument("--max-dilate", type=int, default=2)
    args = ap.parse_args()
    for case_id in args.cases:
        try:
            report = run_case(args.packet, args.machine_root, args.output, case_id, args.gpu, args.aux_device,
                              min_coverage=args.min_part_coverage, max_dilate=args.max_dilate)
            print(json.dumps({"case": case_id, "parts": report["parts"],
                              "whole": {k: report["whole_person_0906"][k] for k in ("collateral_outside_person", "attribute_fixed_crops", "dino_identity_cosine")},
                              "part": {k: report["part_mask"][k] for k in ("collateral_outside_person", "inside_person_outside_part", "attribute_fixed_crops", "dino_identity_cosine")}}), flush=True)
        except Exception as error:
            write_json(args.output / case_id / "failure.json", {"error": f"{type(error).__name__}: {error}"})
            print(json.dumps({"case": case_id, "failed": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
