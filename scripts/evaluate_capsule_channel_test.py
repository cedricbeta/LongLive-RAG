#!/usr/bin/env python3
"""Evaluate reachability, identity direction, guards, and visual evidence for Track B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.vbench_consistency import build_clip_image_encoder, build_dino_encoder
from evaluation.video_consistency import (
    build_clip_adherence_scorer,
    build_raft_dynamic_scorer,
    cross_perspective_evaluate_video,
)
from evaluate_capsule_p2 import (
    decode,
    g0_reference_video,
    masked_shot_embeddings,
    prompt_spec,
    sampled_indices,
    shot_ranges,
)


GUARD_KEYS = (
    "dynamic_degree",
    "inter_shot_composition_diversity",
    "motion_profile_agreement",
    "within_shot_motion",
    "prompt_adherence_mean",
    "prompt_adherence_min",
    "invariant_margin_mean",
    "invariant_margin_min",
    "palette_agreement",
)


def locate_render(folder: Path, scene: str) -> Path:
    matches = sorted(folder.glob(f"*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one render for {scene} in {folder}, found {matches}")
    return matches[0]


def decoded_frame_diff(baseline: Path, edited: Path, boundary_frame: int) -> dict:
    left = cv2.VideoCapture(str(baseline))
    right = cv2.VideoCapture(str(edited))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError(f"could not decode paired videos {baseline} / {edited}")
    index = 0
    pre_equal = True
    post_different = 0
    post_total = 0
    first_different = None
    mean_abs = []
    max_abs = 0
    baseline_hash = hashlib.sha256()
    edited_hash = hashlib.sha256()
    while True:
        ok_left, frame_left = left.read()
        ok_right, frame_right = right.read()
        if ok_left != ok_right:
            raise RuntimeError("paired videos have different decoded frame counts")
        if not ok_left:
            break
        if frame_left.shape != frame_right.shape:
            raise RuntimeError("paired videos have different decoded frame shapes")
        baseline_hash.update(frame_left.tobytes())
        edited_hash.update(frame_right.tobytes())
        equal = bool(np.array_equal(frame_left, frame_right))
        if index < boundary_frame:
            pre_equal &= equal
        else:
            post_total += 1
            if not equal:
                post_different += 1
                if first_different is None:
                    first_different = index
            delta = np.abs(frame_left.astype(np.int16) - frame_right.astype(np.int16))
            mean_abs.append(float(delta.mean()))
            max_abs = max(max_abs, int(delta.max()))
        index += 1
    left.release()
    right.release()
    return {
        "frames": index,
        "boundary_frame": boundary_frame,
        "pre_boundary_decoded_frames_exact": pre_equal,
        "post_boundary_frames": post_total,
        "post_boundary_different_frames": post_different,
        "post_boundary_different_fraction": post_different / max(1, post_total),
        "first_different_frame": first_different,
        "post_boundary_mean_absolute_pixel_delta": float(np.mean(mean_abs)),
        "post_boundary_max_absolute_pixel_delta": max_abs,
        "baseline_decoded_sha256": baseline_hash.hexdigest(),
        "edited_decoded_sha256": edited_hash.hexdigest(),
        "edit_reached_graph": post_different > 0,
    }


def _annotated(image: np.ndarray, label: str, *, edited: bool) -> np.ndarray:
    target = cv2.resize(image, (480, 264), interpolation=cv2.INTER_AREA)
    target = cv2.copyMakeBorder(target, 32, 4, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    color = (255, 40, 40) if edited else (235, 235, 235)
    cv2.putText(target, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    if edited:
        cv2.rectangle(target, (1, 1), (target.shape[1] - 2, target.shape[0] - 2), (255, 0, 0), 4)
    return target


def build_visuals(
    output_dir: Path,
    arm_id: str,
    scene: str,
    baseline: Path,
    edited: Path,
    masks: torch.Tensor,
    first_boundary: int,
) -> dict:
    ranges = shot_ranges(scene)
    indices = [(start + end - 1) // 2 for start, end in ranges]
    baseline_frames = decode(baseline, indices)
    edited_frames = decode(edited, indices)
    rows = []
    crop_rows = []
    crop_shots = []
    for shot, (frame_index, baseline_frame, edited_frame) in enumerate(
        zip(indices, baseline_frames, edited_frames)
    ):
        is_edited = shot >= first_boundary
        rows.append(
            np.concatenate(
                [
                    _annotated(baseline_frame, f"baseline | shot {shot}", edited=False),
                    _annotated(
                        edited_frame,
                        f"edited | shot {shot}{' | EDIT' if is_edited else ''}",
                        edited=is_edited,
                    ),
                ],
                axis=1,
            )
        )
        if not is_edited:
            continue
        mask = masks[frame_index].numpy().astype(np.uint8)
        mask = cv2.resize(
            mask,
            (baseline_frame.shape[1], baseline_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        margin_x = max(16, int((x1 - x0) * 0.2))
        margin_y = max(16, int((y1 - y0) * 0.2))
        x0, x1 = max(0, x0 - margin_x), min(mask.shape[1], x1 + margin_x)
        y0, y1 = max(0, y0 - margin_y), min(mask.shape[0], y1 + margin_y)
        pair = []
        for frame, name in ((baseline_frame, "baseline"), (edited_frame, "edited")):
            crop = frame[y0:y1, x0:x1]
            crop = cv2.resize(crop, (320, 240), interpolation=cv2.INTER_AREA)
            pair.append(_annotated(crop, f"{name} subject | shot {shot}", edited=name == "edited"))
        crop_rows.append(np.concatenate(pair, axis=1))
        crop_shots.append(shot)
    output_dir.mkdir(parents=True, exist_ok=True)
    two_up = output_dir / f"{arm_id}_2up.png"
    cv2.imwrite(str(two_up), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    closeups = output_dir / f"{arm_id}_subject_closeups.png"
    if crop_rows:
        cv2.imwrite(
            str(closeups),
            cv2.cvtColor(np.concatenate(crop_rows, axis=0), cv2.COLOR_RGB2BGR),
        )
    return {
        "two_up": str(two_up.relative_to(ROOT)),
        "subject_closeups": str(closeups.relative_to(ROOT)) if crop_rows else None,
        "subject_closeup_shots": crop_shots,
        "edit_border": "red from the first edited shot onward",
    }


def runtime_integrity(pilot_root: Path, run: dict) -> dict:
    manifest_path = (
        ROOT / run["runtime_dir"] / run["scene"] / "seed0/manifest.json"
    )
    exit_path = pilot_root / f"logs/channel__{run['arm_id']}.exit_code"
    errors = []
    if not exit_path.exists() or exit_path.read_text().strip() != "0":
        errors.append("missing_or_nonzero_exit")
    if not manifest_path.exists():
        errors.append("missing_runtime_manifest")
        return {"passed": False, "errors": errors}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})
    edits = stats.get("edits", [])
    expected = [
        (boundary, layer)
        for boundary in run["boundaries"]
        for layer in run["layers"]
    ]
    observed = [(int(edit["boundary_shot"]), int(edit["layer"])) for edit in edits]
    if observed != expected:
        errors.append("boundary_layer_delivery_mismatch")
    if stats.get("errors"):
        errors.append("runtime_errors")
    if int(stats.get("cross_attention_calls", -1)) != 0:
        errors.append("unexpected_cross_attention")
    if int(stats.get("dump_files", -1)) != 0:
        errors.append("unexpected_v_dump")
    if int(stats.get("mask_files", -1)) != 60:
        errors.append("mask_file_count_not_60")
    if any(int(edit.get("edited_tokens", 0)) <= 0 for edit in edits):
        errors.append("zero_edited_tokens")
    return {
        "passed": not errors,
        "errors": errors,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "expected_boundary_layer_pairs": expected,
        "observed_boundary_layer_pairs": observed,
        "edited_tokens": [int(edit.get("edited_tokens", 0)) for edit in edits],
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    manifest_path = pilot_root / "channel_test/channel_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dino = build_dino_encoder(device=args.device)
    clip = build_clip_image_encoder(device=args.device)
    adherence = build_clip_adherence_scorer(device=args.device)
    raft = build_raft_dynamic_scorer(device=args.device)
    baseline_cache = {}
    arm_results = {}
    for run in manifest["runs"]:
        scene = run["scene"]
        arm_id = run["arm_id"]
        integrity = runtime_integrity(pilot_root, run)
        render = locate_render(ROOT / run["render_dir"], scene)
        baseline = g0_reference_video(pilot_root, scene)
        indices, shots = sampled_indices(scene)
        masks = torch.load(
            pilot_root / f"clipseg_masks/seed0/{scene}.pt",
            map_location="cpu",
            weights_only=False,
        )["binary"].to(torch.bool)
        if scene not in baseline_cache:
            baseline_frames = decode(baseline, indices)
            baseline_subject = masked_shot_embeddings(
                baseline_frames, masks, indices, shots, dino, subject=True
            )
            spec = prompt_spec(scene)
            baseline_guard = cross_perspective_evaluate_video(
                baseline,
                chunk_durations=spec["chunk_durations"],
                captions=spec["captions"],
                adherence_scorer=adherence,
                invariant_caption=spec["invariant_caption"],
                contrast_caption=spec["contrast_caption"],
                invariant_scorer=adherence,
                subject_encoder=dino,
                background_encoder=clip,
                dynamic_scorer=raft,
            )
            baseline_cache[scene] = {
                "subject": baseline_subject,
                "guard": baseline_guard,
            }
        edited_frames = decode(render, indices)
        edited_subject = masked_shot_embeddings(
            edited_frames, masks, indices, shots, dino, subject=True
        )
        spec = prompt_spec(scene)
        edited_guard = cross_perspective_evaluate_video(
            render,
            chunk_durations=spec["chunk_durations"],
            captions=spec["captions"],
            adherence_scorer=adherence,
            invariant_caption=spec["invariant_caption"],
            contrast_caption=spec["contrast_caption"],
            invariant_scorer=adherence,
            subject_encoder=dino,
            background_encoder=clip,
            dynamic_scorer=raft,
        )
        baseline_subject = baseline_cache[scene]["subject"]
        baseline_anchor = baseline_subject[0]
        edited_anchor = edited_subject[0]
        identity_delta = {}
        for shot in sorted(set(baseline_subject) & set(edited_subject)):
            if shot not in run["boundaries"]:
                continue
            baseline_value = float(baseline_subject[shot] @ baseline_anchor)
            edited_value = float(edited_subject[shot] @ edited_anchor)
            identity_delta[str(shot)] = edited_value - baseline_value
        max_abs_identity_delta = max(
            (abs(value) for value in identity_delta.values()), default=0.0
        )
        boundary_frame = shot_ranges(scene)[run["strongest_drift_shot"]][0]
        reachability = decoded_frame_diff(baseline, render, boundary_frame)
        guard_delta = {
            key: float(edited_guard.get(key, float("nan")))
            - float(baseline_cache[scene]["guard"].get(key, float("nan")))
            for key in GUARD_KEYS
        }
        visuals = build_visuals(
            pilot_root / "channel_test/visuals",
            arm_id,
            scene,
            baseline,
            render,
            masks,
            run["strongest_drift_shot"],
        )
        arm_results[arm_id] = {
            "scene": scene,
            "variant": run["variant"],
            "layers": run["layers"],
            "alpha": run["alpha"],
            "boundaries": run["boundaries"],
            "baseline": str(baseline.relative_to(ROOT)),
            "edited": str(render.relative_to(ROOT)),
            "runtime_integrity": integrity,
            "reachability": reachability,
            "identity_consistency_delta_by_shot": identity_delta,
            "max_abs_identity_consistency_delta": max_abs_identity_delta,
            "identity_metric_effect_threshold_abs": manifest[
                "identity_metric_effect_threshold_abs"
            ],
            "identity_metric_effect": bool(
                max_abs_identity_delta
                >= float(manifest["identity_metric_effect_threshold_abs"])
            ),
            "guard_delta": guard_delta,
            "guard_interpretation": "reported as causal evidence; not a failure gate",
            "visuals": visuals,
        }
    result = {
        "schema_version": 1,
        "track": "causal_channel_mini_test",
        "render_count": len(arm_results),
        "delivery_integrity_pass": all(
            arm["runtime_integrity"]["passed"] for arm in arm_results.values()
        ),
        "reachability_any": any(
            arm["reachability"]["edit_reached_graph"] for arm in arm_results.values()
        ),
        "identity_metric_effect_any": any(
            arm["identity_metric_effect"] for arm in arm_results.values()
        ),
        "human_verdict_complete": False,
        "channel_effective": None,
        "decision_rule": "any arm with identity-metric or human-visible subject appearance change => effective",
        "arms": arm_results,
    }
    output = pilot_root / "channel_test/channel_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "delivery_integrity_pass": result["delivery_integrity_pass"],
                "reachability_any": result["reachability_any"],
                "identity_metric_effect_any": result["identity_metric_effect_any"],
                "human_review": "pending",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
