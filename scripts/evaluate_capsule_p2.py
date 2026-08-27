#!/usr/bin/env python3
"""Evaluate P2 delivery integrity (G2) and efficacy/generalization/specificity (G3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vbench_consistency import build_clip_image_encoder, build_dino_encoder
from evaluation.video_consistency import (
    build_clip_adherence_scorer,
    build_raft_dynamic_scorer,
    composition_signature,
    cross_perspective_evaluate_video,
    read_video,
)


SCENES = (
    "bearded_watchmaker_workshop",
    "tattooed_noodle_chef",
    "brown_bear_river",
    "sandy_beach_driftwood",
)
ARMS = ("capsule", "full_vector", "random_subspace")


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(value)
        for value in (
            ROOT / f"example/benchmark_hard_multishot/{scene}/shot_durations.txt"
        ).read_text().split()
    ]
    ranges = []
    start = 0
    for chunks in durations:
        end = start + chunks * 8
        ranges.append((start, end))
        start = end
    return ranges


def sampled_indices(scene: str) -> tuple[list[int], list[int]]:
    frames, shots = [], []
    for shot, (start, end) in enumerate(shot_ranges(scene)):
        indices = np.linspace(start, end - 1, num=3).round().astype(int).tolist()
        frames.extend(indices)
        shots.extend([shot] * len(indices))
    return frames, shots


def locate_video(folder: Path, scene: str) -> Path:
    matches = sorted(folder.glob(f"*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {scene} video under {folder}, found {matches}")
    return matches[0]


def g0_reference_video(pilot_root: Path, scene: str) -> Path:
    g0 = json.loads((pilot_root / "g0/g0_results.json").read_text(encoding="utf-8"))
    if not g0.get("gate_pass"):
        raise RuntimeError("P2 evaluation requires a passing G0 v2 reference ledger")
    relative = g0["scenes"][scene].get("reference")
    if not relative:
        return locate_video(
            ROOT / "videos/benchmark_hard_multishot/stage_A/baseline_seed0", scene
        )
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def decode(path: Path, indices: list[int]) -> np.ndarray:
    wanted = set(indices)
    frames = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame in wanted:
            frames[frame] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame += 1
    capture.release()
    if set(frames) != wanted:
        raise RuntimeError(f"missing frames in {path}: {sorted(wanted - set(frames))}")
    return np.stack([frames[index] for index in indices])


def masked_shot_embeddings(
    frames: np.ndarray,
    masks: torch.Tensor,
    indices: list[int],
    shots: list[int],
    encoder,
    *,
    subject: bool,
) -> dict[int, np.ndarray]:
    prepared, prepared_shots = [], []
    for frame, frame_index, shot in zip(frames, indices, shots):
        mask = cv2.resize(
            masks[frame_index].numpy().astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if subject and not mask.any():
            continue
        image = frame.copy()
        image[~mask if subject else mask] = 127
        prepared.append(image)
        prepared_shots.append(shot)
    features = np.asarray(encoder(np.stack(prepared)), dtype=np.float64)
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    result = {}
    for shot in sorted(set(prepared_shots)):
        group = features[np.asarray(prepared_shots) == shot]
        mean = group.mean(axis=0)
        result[shot] = mean / max(np.linalg.norm(mean), 1e-8)
    return result


def face_shot_embeddings(
    frames: np.ndarray,
    masks: torch.Tensor,
    indices: list[int],
    shots: list[int],
    encoder,
) -> dict[int, np.ndarray]:
    """DINO face/upper-head crops for the two human scenes."""
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    crops, crop_shots = [], []
    for frame, frame_index, shot in zip(frames, indices, shots):
        mask = cv2.resize(
            masks[frame_index].numpy().astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if not mask.any():
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        detections = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
        )
        candidates = []
        for x, y, width, height in detections:
            center_y = min(mask.shape[0] - 1, y + height // 2)
            center_x = min(mask.shape[1] - 1, x + width // 2)
            if mask[center_y, center_x]:
                candidates.append((x, y, width, height))
        if candidates:
            x, y, width, height = max(candidates, key=lambda box: box[2] * box[3])
            margin = int(round(max(width, height) * 0.20))
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1 = min(frame.shape[1], x + width + margin)
            y1 = min(frame.shape[0], y + height + margin)
        else:
            ys, xs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            if (x1 - x0) * (y1 - y0) < frame.shape[0] * frame.shape[1] * 0.08:
                continue
            y1 = y0 + max(1, int(round((y1 - y0) * 0.45)))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA))
        crop_shots.append(shot)
    if not crops:
        return {}
    features = np.asarray(encoder(np.stack(crops)), dtype=np.float64)
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    result = {}
    for shot in sorted(set(crop_shots)):
        group = features[np.asarray(crop_shots) == shot]
        mean = group.mean(axis=0)
        result[shot] = mean / max(np.linalg.norm(mean), 1e-8)
    return result


def prompt_spec(scene: str) -> dict:
    scene_dir = ROOT / "example/benchmark_hard_multishot" / scene
    global_payload = json.loads((scene_dir / "global.json").read_text(encoding="utf-8"))
    return {
        "chunk_durations": [
            int(value) for value in (scene_dir / "shot_durations.txt").read_text().split()
        ],
        "captions": [
            str(json.loads((scene_dir / f"{shot}.json").read_text(encoding="utf-8"))["caption"])
            for shot in range(8)
        ],
        "invariant_caption": str(global_payload["invariant_caption"]),
        "contrast_caption": str(global_payload["contrast_caption"]),
    }


def runtime_integrity(pilot_root: Path) -> dict:
    records = {}
    token_sequences = {}
    delivery_configs = {}
    all_ok = True
    for arm in ARMS:
        for scene in SCENES:
            path = pilot_root / f"p2/{arm}/runtime/{scene}/seed0/manifest.json"
            if not path.exists():
                record = {"passed": False, "reason": f"missing {path}"}
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                stats = payload.get("stats", {})
                edits = stats.get("edits", [])
                edit_config = payload.get("edit_config", {})
                expected_boundaries = [
                    int(value) for value in edit_config.get("boundaries", [])
                ]
                passed = bool(
                    not stats.get("errors")
                    and len(edits) == len(expected_boundaries)
                    and [int(edit["boundary_shot"]) for edit in edits]
                    == expected_boundaries
                    and edit_config.get("arm") == arm
                    and edit_config.get("enabled") is True
                    and payload.get("probe_config", {}).get("enabled") is False
                    and int(stats.get("cross_attention_calls", -1)) == 0
                    and int(stats.get("dump_files", -1)) == 0
                    and int(stats.get("mask_files", -1)) == 60
                    and all(int(edit.get("edited_tokens", 0)) > 0 for edit in edits)
                    and edit_config.get("mask_method")
                    in {
                        "clipseg_precomputed",
                        "clipseg_precomputed_with_manual_visibility_overrides",
                    }
                )
                record = {
                    "passed": passed,
                    "manifest": str(path.relative_to(ROOT)),
                    "edited_tokens": [int(edit["edited_tokens"]) for edit in edits],
                    "stats": stats,
                }
                token_sequences[(arm, scene)] = record["edited_tokens"]
                delivery_configs[(arm, scene)] = {
                    "layer": int(edit_config.get("layer", -1)),
                    "rank": int(edit_config.get("rank", -1)),
                    "alpha": float(edit_config.get("alpha", float("nan"))),
                    "boundaries": expected_boundaries,
                    "mask_path": edit_config.get("mask_path"),
                    "artifact_path": edit_config.get("artifact_path"),
                }
            records[f"{arm}/{scene}"] = record
            all_ok &= bool(record["passed"])
    same_delivery = True
    for scene in SCENES:
        sequences = [token_sequences.get((arm, scene)) for arm in ARMS]
        same_delivery &= all(sequence == sequences[0] for sequence in sequences[1:])
        configs = [delivery_configs.get((arm, scene)) for arm in ARMS]
        if any(config is None for config in configs):
            same_delivery = False
            continue
        for config in configs[1:]:
            same_delivery &= all(
                config[key] == configs[0][key]
                for key in (
                    "layer",
                    "rank",
                    "alpha",
                    "boundaries",
                    "mask_path",
                    "artifact_path",
                )
            )
    return {
        "gate_pass": bool(all_ok and same_delivery),
        "same_edited_token_delivery_across_arms": same_delivery,
        "k_and_rope_invariant": (
            "code-path invariant plus tests/test_capsule.py: editor index-copies V only; "
            "K and RoPE are never passed to the editor"
        ),
        "runs": records,
    }


def guard_verdict(baseline: dict[str, float], modified: dict[str, float]) -> dict:
    """Apply the frozen Round-18 non-regression policy without an efficacy gate."""
    keys = (
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
    failures = []
    for side, metrics in (("baseline", baseline), ("modified", modified)):
        for key in keys:
            if key not in metrics or not np.isfinite(metrics[key]):
                failures.append(f"{side}:{key}:nonfinite_or_missing")
    baseline_values = {
        key: float(baseline.get(key, float("nan"))) for key in keys
    }
    modified_values = {
        key: float(modified.get(key, float("nan"))) for key in keys
    }
    checks = {
        "diversity_within_10pct": modified_values["inter_shot_composition_diversity"]
        >= 0.90 * baseline_values["inter_shot_composition_diversity"],
        "raft_motion_within_20pct": modified_values["dynamic_degree"]
        >= 0.80 * baseline_values["dynamic_degree"],
        "adherence_mean_within_0p02": modified_values["prompt_adherence_mean"]
        >= baseline_values["prompt_adherence_mean"] - 0.02,
        "adherence_min_within_0p02": modified_values["prompt_adherence_min"]
        >= baseline_values["prompt_adherence_min"] - 0.02,
        "invariant_mean_nonregression": modified_values["invariant_margin_mean"]
        >= baseline_values["invariant_margin_mean"],
        "invariant_min_nonregression": modified_values["invariant_margin_min"]
        >= baseline_values["invariant_margin_min"],
    }
    failures.extend(key for key, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "checks": {key: bool(value) for key, value in checks.items()},
        "failures": failures,
        "policy": {
            "diversity_tolerance_relative": 0.10,
            "motion_tolerance_relative": 0.20,
            "adherence_tolerance": 0.02,
            "invariant_tolerance": 0.0,
        },
    }


def composition_diversity_by_shot(frames: np.ndarray, ranges: list[tuple[int, int]]) -> dict[int, float]:
    signatures = []
    for start, end in ranges:
        signature = np.stack([composition_signature(frame) for frame in frames[start:end]]).mean(axis=0)
        signature /= max(np.linalg.norm(signature), 1e-8)
        signatures.append(signature)
    anchor = signatures[0]
    return {shot: float(1.0 - signature @ anchor) for shot, signature in enumerate(signatures)}


def raft_motion_by_shot(frames: np.ndarray, ranges: list[tuple[int, int]], scorer) -> dict[int, float]:
    return {shot: float(scorer(frames[start:end])) for shot, (start, end) in enumerate(ranges)}


def bootstrap_ci(values: list[float], *, repeats: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.5, 0.975])]


def paired_leakage_advantage(
    capsule_by_scene: dict,
    full_by_scene: dict,
    *,
    repeats: int,
) -> dict:
    """Positive values mean capsule leaks less than full-vector for a paired cell."""
    values = {"motion": [], "background": [], "diversity": []}
    cells = {"motion": [], "background": [], "diversity": []}
    for scene in SCENES:
        capsule = capsule_by_scene[scene]
        full = full_by_scene[scene]
        pairs = (
            (
                "motion",
                capsule["raft_motion_delta_by_shot"],
                full["raft_motion_delta_by_shot"],
                lambda cap, complete: abs(complete) - abs(cap),
            ),
            (
                "background",
                capsule["background_preservation_by_shot"],
                full["background_preservation_by_shot"],
                lambda cap, complete: cap - complete,
            ),
            (
                "diversity",
                capsule["inter_shot_diversity_delta_by_shot"],
                full["inter_shot_diversity_delta_by_shot"],
                lambda cap, complete: max(0.0, -complete) - max(0.0, -cap),
            ),
        )
        for component, cap_stream, full_stream, advantage in pairs:
            for shot in sorted(set(cap_stream) & set(full_stream), key=int):
                values[component].append(
                    float(advantage(float(cap_stream[shot]), float(full_stream[shot])))
                )
                cells[component].append(f"{scene}/shot{shot}")
    result = {}
    for index, component in enumerate(("motion", "background", "diversity")):
        component_values = values[component]
        if component_values:
            ci = bootstrap_ci(
                component_values,
                repeats=repeats,
                seed=20260730 + index,
            )
            mean = float(np.mean(component_values))
        else:
            ci = [float("nan")] * 3
            mean = float("nan")
        result[component] = {
            "paired_cells": cells[component],
            "advantage_values": component_values,
            "mean_advantage": mean,
            "bootstrap_ci95": ci,
            "significantly_less_leakage": bool(
                np.isfinite(mean) and mean > 0.0 and ci[0] > 0.0
            ),
        }
    return result


def write_report(output_dir: Path, result: dict) -> None:
    g2 = result["g2"]
    g3 = result["g3"]
    lines = [
        "# G2/G3 capsule intervention gate",
        "",
        f"G2 delivery integrity: **{'PASS' if g2['gate_pass'] else 'FAIL'}**  ",
        f"G3 efficacy/generalization/specificity: **{'PASS' if g3['gate_pass'] else 'FAIL'}**",
        "",
        "| arm | mean subject-consistency delta | bootstrap 95% CI | mean background preservation |",
        "|---|---:|---|---:|",
    ]
    for arm in ARMS:
        record = g3["arms"][arm]
        ci = record["identity_delta_bootstrap_ci95"]
        lines.append(
            f"| {arm} | {record['mean_identity_consistency_delta']:.4f} "
            f"| [{ci[0]:.4f}, {ci[2]:.4f}] "
            f"| {record['mean_background_preservation']:.4f} |"
        )
    capsule = g3["capsule_schema"]
    lines += [
        "",
        "Leakage vector (componentwise; no weighted scalar):",
        "",
        "| arm | RAFT motion Δ | CLIP background similarity Δ | inter-shot diversity Δ | Round-18 guards |",
        "|---|---:|---:|---:|:---:|",
    ]
    for arm in ARMS:
        record = g3["arms"][arm]
        vector = record["leakage_vector"]
        lines.append(
            f"| {arm} | {vector['raft_motion_delta_mean']:.4f} "
            f"| {vector['clip_background_similarity_delta_mean']:.4f} "
            f"| {vector['inter_shot_diversity_delta_mean']:.4f} "
            f"| {'pass' if record['existing_guards_passed'] else 'fail'} |"
        )
    lines += [
        "",
        f"- Efficacy: {'pass' if capsule['efficacy']['passed'] else 'fail'}.",
        f"- Generalization: {'pass' if capsule['generalization']['passed'] else 'fail'}.",
        f"- Specificity: {'pass' if capsule['specificity']['passed'] else 'fail'}.",
        f"- Scenes beating both baseline and random: {capsule['generalization']['scenes_improved']}.",
        f"- Edited boundaries beating both controls: {capsule['generalization']['boundaries_improved']}.",
        "",
        "Subject scores use DINO on the predeclared identity mask; specificity uses CLIP on its complement. "
        "All identity and leakage comparisons are restricted to declared edited drift shots and paired against "
        "the G0-selected probe-off reference at the same scene and shot (stage_A for prevalidated scenes; the same-code diagnostic reference when a stale archive is superseded).",
    ]
    (output_dir / "G2_G3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    output_dir = pilot_root / "p2/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    p2_manifest = json.loads(
        (pilot_root / "p2/p2_run_manifest.json").read_text(encoding="utf-8")
    )
    alpha_scan = json.loads(
        (pilot_root / "p2/alpha_scan.json").read_text(encoding="utf-8")
    )
    p2_audit = json.loads(
        (pilot_root / "p2/p2_run_audit.json").read_text(encoding="utf-8")
    )
    if not p2_audit.get("passed") or int(p2_audit.get("complete_cells", 0)) != 12:
        raise SystemExit("P2 run audit did not certify all 12 final arm×scene cells")
    boundaries_by_scene = {
        scene: [
            int(value)
            for value in next(
                record
                for record in p2_manifest["runs"]
                if record["arm"] == "capsule" and record["scene"] == scene
            )["boundaries"]
        ]
        for scene in SCENES
    }
    dino = build_dino_encoder(device=args.device)
    clip = build_clip_image_encoder(device=args.device)
    adherence = build_clip_adherence_scorer(device=args.device)
    raft = build_raft_dynamic_scorer(device=args.device)
    g2 = runtime_integrity(pilot_root)
    g2["p2_run_audit"] = str(
        (pilot_root / "p2/p2_run_audit.json").relative_to(ROOT)
    )
    g2["frozen_inputs_unchanged"] = bool(
        p2_audit.get("frozen_inputs_unchanged")
    )
    arm_pairs = {arm: [] for arm in ARMS}
    arm_background = {arm: [] for arm in ARMS}
    per_scene = {arm: {} for arm in ARMS}
    baseline_references = {}
    guard_keys = (
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

    for scene in SCENES:
        indices, shots = sampled_indices(scene)
        ranges = shot_ranges(scene)
        edited_shots = set(boundaries_by_scene[scene])
        mask_payload = torch.load(
            pilot_root / f"clipseg_masks/seed0/{scene}.pt",
            map_location="cpu",
            weights_only=False,
        )
        masks = mask_payload["binary"].to(torch.bool)
        baseline_path = g0_reference_video(pilot_root, scene)
        baseline_references[scene] = str(baseline_path.relative_to(ROOT))
        baseline_frames = decode(baseline_path, indices)
        baseline_subject = masked_shot_embeddings(
            baseline_frames, masks, indices, shots, dino, subject=True
        )
        baseline_background = masked_shot_embeddings(
            baseline_frames, masks, indices, shots, clip, subject=False
        )
        baseline_face = (
            face_shot_embeddings(
                baseline_frames, masks, indices, shots, dino
            )
            if scene in {"bearded_watchmaker_workshop", "tattooed_noodle_chef"}
            else {}
        )
        spec = prompt_spec(scene)
        baseline_guard = cross_perspective_evaluate_video(
            baseline_path,
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
        baseline_full_frames = read_video(baseline_path)
        baseline_raft_by_shot = raft_motion_by_shot(
            baseline_full_frames, ranges, raft
        )
        baseline_diversity_by_shot = composition_diversity_by_shot(
            baseline_full_frames, ranges
        )
        if 0 not in baseline_subject:
            raise RuntimeError(f"{scene} has no subject anchor in shot0")
        baseline_consistency = {
            shot: float(feature @ baseline_subject[0])
            for shot, feature in baseline_subject.items()
        }
        for arm in ARMS:
            render = locate_video(pilot_root / f"p2/{arm}/renders", scene)
            frames = decode(render, indices)
            subject_features = masked_shot_embeddings(
                frames, masks, indices, shots, dino, subject=True
            )
            background_features = masked_shot_embeddings(
                frames, masks, indices, shots, clip, subject=False
            )
            face_features = (
                face_shot_embeddings(frames, masks, indices, shots, dino)
                if baseline_face
                else {}
            )
            modified_guard = cross_perspective_evaluate_video(
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
            modified_full_frames = read_video(render)
            modified_raft_by_shot = raft_motion_by_shot(
                modified_full_frames, ranges, raft
            )
            modified_diversity_by_shot = composition_diversity_by_shot(
                modified_full_frames, ranges
            )
            consistency = {
                shot: float(feature @ subject_features[0])
                for shot, feature in subject_features.items()
            }
            scene_deltas = []
            scene_delta_by_shot = {}
            for shot in sorted(set(baseline_consistency) & set(consistency)):
                if shot not in edited_shots:
                    continue
                delta = consistency[shot] - baseline_consistency[shot]
                arm_pairs[arm].append(float(delta))
                scene_deltas.append(float(delta))
                scene_delta_by_shot[str(shot)] = float(delta)
            scene_background = []
            scene_background_by_shot = {}
            for shot in sorted(set(baseline_background) & set(background_features)):
                if shot not in edited_shots:
                    continue
                preservation = float(
                    background_features[shot] @ baseline_background[shot]
                )
                arm_background[arm].append(preservation)
                scene_background.append(preservation)
                scene_background_by_shot[str(shot)] = preservation
            face_deltas = []
            face_delta_by_shot = {}
            if baseline_face and 0 in baseline_face and 0 in face_features:
                baseline_face_consistency = {
                    shot: float(feature @ baseline_face[0])
                    for shot, feature in baseline_face.items()
                }
                modified_face_consistency = {
                    shot: float(feature @ face_features[0])
                    for shot, feature in face_features.items()
                }
                for shot in sorted(
                    (set(baseline_face_consistency) & set(modified_face_consistency))
                    & edited_shots
                ):
                    face_delta = (
                        modified_face_consistency[shot]
                        - baseline_face_consistency[shot]
                    )
                    face_deltas.append(face_delta)
                    face_delta_by_shot[str(shot)] = face_delta
            guard_delta = {
                key: float(
                    modified_guard.get(key, float("nan"))
                    - baseline_guard.get(key, float("nan"))
                )
                for key in guard_keys
            }
            round18_guard = guard_verdict(baseline_guard, modified_guard)
            raft_relative_delta_by_shot = {
                str(shot): float(
                    (modified_raft_by_shot[shot] - baseline_raft_by_shot[shot])
                    / max(abs(baseline_raft_by_shot[shot]), 1e-8)
                )
                for shot in sorted(edited_shots)
            }
            raft_delta_by_shot = {
                str(shot): float(
                    modified_raft_by_shot[shot] - baseline_raft_by_shot[shot]
                )
                for shot in sorted(edited_shots)
            }
            diversity_relative_delta_by_shot = {
                str(shot): float(
                    (
                        modified_diversity_by_shot[shot]
                        - baseline_diversity_by_shot[shot]
                    )
                    / max(abs(baseline_diversity_by_shot[shot]), 1e-8)
                )
                for shot in sorted(edited_shots)
            }
            diversity_delta_by_shot = {
                str(shot): float(
                    modified_diversity_by_shot[shot]
                    - baseline_diversity_by_shot[shot]
                )
                for shot in sorted(edited_shots)
            }
            clip_background_delta_by_shot = {
                shot: float(value - 1.0)
                for shot, value in scene_background_by_shot.items()
            }
            per_scene[arm][scene] = {
                "baseline_reference": str(baseline_path.relative_to(ROOT)),
                "edited_shots": sorted(edited_shots),
                "mean_identity_consistency_delta": float(np.mean(scene_deltas)),
                "identity_consistency_delta_by_shot": scene_delta_by_shot,
                "mean_face_crop_dino_delta": (
                    float(np.mean(face_deltas)) if face_deltas else None
                ),
                "face_crop_dino_delta_by_shot": face_delta_by_shot,
                "mean_background_preservation": float(np.mean(scene_background)),
                "background_preservation_by_shot": scene_background_by_shot,
                "clip_background_similarity_delta_by_shot": clip_background_delta_by_shot,
                "guard_baseline": {
                    key: float(baseline_guard.get(key, float("nan")))
                    for key in guard_keys
                },
                "guard_modified": {
                    key: float(modified_guard.get(key, float("nan")))
                    for key in guard_keys
                },
                "guard_delta": guard_delta,
                "round18_guard": round18_guard,
                "raft_motion_relative_delta_by_shot": raft_relative_delta_by_shot,
                "raft_motion_delta_by_shot": raft_delta_by_shot,
                "inter_shot_diversity_relative_delta_by_shot": diversity_relative_delta_by_shot,
                "inter_shot_diversity_delta_by_shot": diversity_delta_by_shot,
            }
            del modified_full_frames
        del baseline_full_frames

    arm_results = {}
    for arm_index, arm in enumerate(ARMS):
        motion_relative = []
        motion_delta = []
        diversity_relative = []
        diversity_delta = []
        guard_failures = []
        for scene, record in per_scene[arm].items():
            baseline_guard = record["guard_baseline"]
            modified_guard = record["guard_modified"]
            motion_relative.extend(
                record["raft_motion_relative_delta_by_shot"].values()
            )
            motion_delta.extend(record["raft_motion_delta_by_shot"].values())
            diversity_relative.extend(
                record["inter_shot_diversity_relative_delta_by_shot"].values()
            )
            diversity_delta.extend(
                record["inter_shot_diversity_delta_by_shot"].values()
            )
            guard_failures.extend(
                f"{scene}:{failure}"
                for failure in record["round18_guard"]["failures"]
            )
        background_loss = float(1.0 - np.mean(arm_background[arm]))
        leakage_vector = {
            "raft_motion_delta_mean": float(np.mean(motion_delta)),
            "clip_background_similarity_delta_mean": -background_loss,
            "inter_shot_diversity_delta_mean": float(np.mean(diversity_delta)),
            "raft_motion_relative_delta_mean": float(np.mean(motion_relative)),
            "clip_background_cosine_loss_mean": background_loss,
            "inter_shot_diversity_relative_delta_mean": float(
                np.mean(diversity_relative)
            ),
        }
        arm_results[arm] = {
            "mean_identity_consistency_delta": float(np.mean(arm_pairs[arm])),
            "identity_delta_bootstrap_ci95": bootstrap_ci(
                arm_pairs[arm],
                repeats=args.bootstrap_repeats,
                seed=20260715 + arm_index,
            ),
            "mean_background_preservation": float(np.mean(arm_background[arm])),
            "background_preservation_p05": float(
                np.quantile(arm_background[arm], 0.05)
            ),
            "paired_identity_shots": len(arm_pairs[arm]),
            "leakage_vector": leakage_vector,
            "existing_guard_failures": sorted(set(guard_failures)),
            "existing_guards_passed": not guard_failures,
            "numeric_finite": bool(
                np.isfinite(np.asarray(arm_pairs[arm], dtype=np.float64)).all()
                and np.isfinite(np.asarray(arm_background[arm], dtype=np.float64)).all()
                and np.isfinite(np.asarray(motion_delta, dtype=np.float64)).all()
                and np.isfinite(np.asarray(diversity_delta, dtype=np.float64)).all()
                and all(np.isfinite(value) for value in leakage_vector.values())
            ),
            "per_scene": per_scene[arm],
        }
    g2["existing_guard_failures"] = {
        arm: arm_results[arm]["existing_guard_failures"] for arm in ARMS
    }
    g2["existing_guard_results_reported_for_all_arms"] = True
    g2["numeric_no_nan"] = all(
        arm_results[arm]["numeric_finite"]
        and not any("nonfinite_or_missing" in failure for failure in arm_results[arm]["existing_guard_failures"])
        for arm in ARMS
    )
    g2["alpha_rendered_visual_stability"] = {
        "passed": bool(alpha_scan.get("gate_pass")),
        "selected_alpha": alpha_scan.get("selected_alpha"),
        "visual_review_complete": bool(alpha_scan.get("visual_review_complete")),
        "artifact": str((pilot_root / "p2/alpha_scan/alpha_4up.mp4").relative_to(ROOT)),
    }
    g2["gate_pass"] = bool(
        g2["gate_pass"]
        and g2["frozen_inputs_unchanged"]
        and g2["numeric_no_nan"]
        and g2["alpha_rendered_visual_stability"]["passed"]
        and g2["alpha_rendered_visual_stability"]["visual_review_complete"]
    )
    capsule = arm_results["capsule"]
    leakage_paired_vs_full = paired_leakage_advantage(
        capsule["per_scene"],
        arm_results["full_vector"]["per_scene"],
        repeats=args.bootstrap_repeats,
    )
    leakage_better_than_full = {
        component: record["significantly_less_leakage"]
        for component, record in leakage_paired_vs_full.items()
    }
    random_arm = arm_results["random_subspace"]
    paired_keys = [
        (scene, shot)
        for scene in SCENES
        for shot in capsule["per_scene"][scene]["identity_consistency_delta_by_shot"]
        if shot in random_arm["per_scene"][scene]["identity_consistency_delta_by_shot"]
    ]
    capsule_minus_random = [
        capsule["per_scene"][scene]["identity_consistency_delta_by_shot"][shot]
        - random_arm["per_scene"][scene]["identity_consistency_delta_by_shot"][shot]
        for scene, shot in paired_keys
    ]
    improved_scenes = [
        scene
        for scene in SCENES
        if capsule["per_scene"][scene]["mean_identity_consistency_delta"] > 0
        and capsule["per_scene"][scene]["mean_identity_consistency_delta"]
        > random_arm["per_scene"][scene]["mean_identity_consistency_delta"]
    ]
    improved_boundaries = [
        f"{scene}/shot{shot}"
        for scene, shot in paired_keys
        if capsule["per_scene"][scene]["identity_consistency_delta_by_shot"][shot] > 0
        and capsule["per_scene"][scene]["identity_consistency_delta_by_shot"][shot]
        > random_arm["per_scene"][scene]["identity_consistency_delta_by_shot"][shot]
    ]
    capsule_minus_random_ci = bootstrap_ci(
        capsule_minus_random,
        repeats=args.bootstrap_repeats,
        seed=20260718,
    )
    efficacy = {
        "passed": bool(
            capsule["mean_identity_consistency_delta"] > 0
            and float(np.mean(capsule_minus_random)) > 0
        ),
        "paired_delta_vs_baseline_mean": capsule["mean_identity_consistency_delta"],
        "paired_delta_vs_baseline_ci95": capsule["identity_delta_bootstrap_ci95"],
        "paired_delta_vs_random_mean": float(np.mean(capsule_minus_random)),
        "paired_delta_vs_random_ci95": capsule_minus_random_ci,
        "face_crop_dino_by_scene": {
            scene: capsule["per_scene"][scene]["face_crop_dino_delta_by_shot"]
            for scene in ("bearded_watchmaker_workshop", "tattooed_noodle_chef")
        },
        "criterion": "positive edited-shot paired effect versus baseline and rank-matched random",
    }
    generalization = {
        "passed": bool(
            len(improved_scenes) >= 2
            and capsule["identity_delta_bootstrap_ci95"][0] > 0
        ),
        "scenes_improved": improved_scenes,
        "boundaries_improved": improved_boundaries,
        "fixed_hyperparameters": {
            "layer": p2_manifest["runs"][0]["layer"],
            "rank": p2_manifest["runs"][0]["rank"],
            "alpha": p2_manifest["selected_alpha"],
            "mask_method": "fixed seed0 precomputed CLIPSeg with declared visibility overrides",
            "donor_rule": "same-scene seed0 shot0 clean-recache entity-token mean",
        },
        "criterion": ">=2/4 scenes better than baseline and random, with baseline-paired bootstrap lower CI > 0",
    }
    specificity = {
        "passed": bool(
            capsule["mean_background_preservation"] >= 0.95
            and capsule["background_preservation_p05"] >= 0.90
            and sum(leakage_better_than_full.values()) >= 2
            and capsule["existing_guards_passed"]
        ),
        "criterion": (
            "mean background CLIP >=0.95, fifth percentile >=0.90, and the "
            "reported motion/background/diversity leakage vector has paired-bootstrap "
            "lower confidence bound >0 versus full-vector on at least two of three "
            "components; components are not aggregated"
        ),
        "componentwise_better_than_full_vector": leakage_better_than_full,
        "paired_componentwise_advantage_vs_full_vector": leakage_paired_vs_full,
    }
    g3 = {
        "gate_pass": bool(
            g2["gate_pass"]
            and efficacy["passed"]
            and generalization["passed"]
            and specificity["passed"]
        ),
        "arms": arm_results,
        "capsule_schema": {
            "efficacy": efficacy,
            "generalization": generalization,
            "specificity": specificity,
        },
    }
    result = {
        "schema_version": 2,
        "baseline_references": baseline_references,
        "g2": g2,
        "g3": g3,
    }
    (output_dir / "g2_g3_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir, result)
    print(json.dumps({"g2_pass": g2["gate_pass"], "g3_pass": g3["gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
