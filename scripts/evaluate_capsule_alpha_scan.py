#!/usr/bin/env python3
"""Audit rendered alpha candidates and create mandatory visual-review assets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vbench_consistency import build_clip_image_encoder, build_dino_encoder
from evaluation.video_consistency import (
    build_clip_adherence_scorer,
    build_raft_dynamic_scorer,
    cross_perspective_evaluate_video,
)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def locate(folder: Path, scene: str) -> Path:
    matches = sorted(folder.glob(f"*-{scene}-*_regular.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {scene} video in {folder}, found {matches}")
    return matches[0]


def g0_reference_video(pilot_root: Path, scene: str) -> Path:
    g0 = json.loads((pilot_root / "g0/g0_results.json").read_text(encoding="utf-8"))
    if not g0.get("gate_pass"):
        raise RuntimeError("alpha evaluation requires a passing G0 v2 reference ledger")
    relative = g0["scenes"][scene].get("reference")
    if not relative:
        return locate(
            ROOT / "videos/benchmark_hard_multishot/stage_A/baseline_seed0", scene
        )
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return np.stack(frames), fps


def shot_ranges(scene: str) -> list[tuple[int, int]]:
    durations = [
        int(value)
        for value in (ROOT / f"example/benchmark_hard_multishot/{scene}/shot_durations.txt").read_text().split()
    ]
    result, start = [], 0
    for chunks in durations:
        end = start + chunks * 8
        result.append((start, end))
        start = end
    return result


def temporal_jumps(frames: np.ndarray, starts: list[int]) -> list[float]:
    values = []
    for start in starts:
        before = frames[start - 1].astype(np.float32)
        after = frames[start].astype(np.float32)
        values.append(float(np.mean(np.abs(after - before)) / 255.0))
    return values


def runtime_ok(path: Path, boundaries: list[int], alpha: float) -> tuple[bool, dict]:
    if not path.exists():
        return False, {"reason": f"missing {path.relative_to(ROOT)}"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})
    edits = stats.get("edits", [])
    passed = bool(
        not stats.get("errors")
        and [int(edit["boundary_shot"]) for edit in edits] == boundaries
        and all(abs(float(edit["alpha"]) - alpha) < 1e-8 for edit in edits)
        and all(edit["arm"] == "capsule" for edit in edits)
        and payload.get("probe_config", {}).get("enabled") is False
    )
    return passed, {
        "manifest": str(path.relative_to(ROOT)),
        "edited_tokens": [int(edit["edited_tokens"]) for edit in edits],
        "errors": stats.get("errors", []),
    }


def build_contact_sheet(
    output: Path,
    rows: list[tuple[str, np.ndarray]],
    indices: list[int],
) -> None:
    width, height = 256, 144
    header = np.full((38, 120 + len(indices) * width, 3), 24, dtype=np.uint8)
    for column, index in enumerate(indices):
        cv2.putText(header, f"f{index}", (120 + column * width + 94, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    assembled = [header]
    for label, frames in rows:
        name = np.full((height, 120, 3), 24, dtype=np.uint8)
        cv2.putText(name, label, (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
        cells = [cv2.resize(frames[index], (width, height), interpolation=cv2.INTER_AREA) for index in indices]
        assembled.append(np.concatenate([name, *cells], axis=1))
    cv2.imwrite(str(output), np.concatenate(assembled, axis=0))


def build_grid(output: Path, rows: list[tuple[str, Path]]) -> None:
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for _, path in rows:
        command += ["-i", str(path)]
    filters, labels = [], []
    for index, (label, _) in enumerate(rows):
        out = f"q{index}"
        filters.append(
            f"[{index}:v]scale=640:352,drawbox=x=0:y=0:w=260:h=38:color=black@0.75:t=fill,"
            f"drawtext=fontfile={FONT}:text='{label}':x=12:y=8:fontsize=24:fontcolor=white[{out}]"
        )
        labels.append(f"[{out}]")
    filters.append("".join(labels) + "xstack=inputs=4:layout=0_0|640_0|0_352|640_352[grid]")
    command += ["-filter_complex", ";".join(filters), "-map", "[grid]", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output)]
    subprocess.run(command, check=True)


def prompt_spec(scene: str) -> dict:
    scene_dir = ROOT / "example/benchmark_hard_multishot" / scene
    global_payload = json.loads((scene_dir / "global.json").read_text(encoding="utf-8"))
    return {
        "chunk_durations": [int(value) for value in (scene_dir / "shot_durations.txt").read_text().split()],
        "captions": [
            str(json.loads((scene_dir / f"{shot}.json").read_text(encoding="utf-8"))["caption"])
            for shot in range(8)
        ],
        "invariant_caption": str(global_payload["invariant_caption"]),
        "contrast_caption": str(global_payload["contrast_caption"]),
    }


def full_guard_metrics(path: Path, spec: dict, scorers: dict) -> dict[str, float]:
    return cross_perspective_evaluate_video(
        path,
        chunk_durations=spec["chunk_durations"],
        captions=spec["captions"],
        adherence_scorer=scorers["adherence"],
        invariant_caption=spec["invariant_caption"],
        contrast_caption=spec["contrast_caption"],
        invariant_scorer=scorers["adherence"],
        subject_encoder=scorers["dino"],
        background_encoder=scorers["clip"],
        dynamic_scorer=scorers["raft"],
    )


def full_guard_verdict(baseline: dict, modified: dict) -> dict:
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
        "policy": {
            "diversity_tolerance_relative": 0.10,
            "motion_tolerance_relative": 0.20,
            "adherence_tolerance": 0.02,
            "invariant_tolerance": 0.0,
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "failures": failures,
        "baseline": baseline_values,
        "modified": modified_values,
        "delta": {
            key: float(modified_values[key] - baseline_values[key])
            for key in keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_root", default="videos/capsule_pilot_20260715")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    pilot_root = (ROOT / args.pilot_root).resolve()
    scan_root = pilot_root / "p2/alpha_scan"
    manifest = json.loads((scan_root / "render_manifest.json").read_text(encoding="utf-8"))
    numeric = json.loads((pilot_root / "p2/alpha_numeric_prefilter.json").read_text(encoding="utf-8"))
    scene = manifest["scene"]
    boundaries = [int(value) for value in manifest["boundaries"]]
    ranges = shot_ranges(scene)
    boundary_frames = [ranges[shot][0] for shot in boundaries]
    baseline_path = g0_reference_video(pilot_root, scene)
    baseline, baseline_fps = read_video(baseline_path)
    baseline_jumps = temporal_jumps(baseline, boundary_frames)
    scorers = {
        "dino": build_dino_encoder(device=args.device),
        "clip": build_clip_image_encoder(device=args.device),
        "adherence": build_clip_adherence_scorer(device=args.device),
        "raft": build_raft_dynamic_scorer(device=args.device),
    }
    spec = prompt_spec(scene)
    baseline_guard = full_guard_metrics(baseline_path, spec, scorers)
    records = []
    visual_rows: list[tuple[str, np.ndarray]] = [("baseline", baseline)]
    video_rows: list[tuple[str, Path]] = [("baseline", baseline_path)]
    eligible_numeric = set(float(value) for value in numeric["eligible_alphas"])
    for run in manifest["runs"]:
        alpha = float(run["alpha"])
        render_path = locate(ROOT / run["render_dir"], scene)
        frames, fps = read_video(render_path)
        runtime_path = ROOT / run["runtime_dir"] / scene / "seed0/manifest.json"
        integrity, runtime = runtime_ok(runtime_path, boundaries, alpha)
        same_shape = frames.shape == baseline.shape
        prefix_end = ranges[min(boundaries)][0]
        prefix_mae = float(np.mean(np.abs(frames[:prefix_end].astype(np.float32) - baseline[:prefix_end].astype(np.float32)))) if same_shape and prefix_end else math.inf
        jumps = temporal_jumps(frames, boundary_frames) if same_shape else []
        ratios = [value / max(reference, 1e-6) for value, reference in zip(jumps, baseline_jumps)]
        black_fraction = float(np.mean(frames.mean(axis=(1, 2, 3)) < 2.0))
        clipped_fraction = float(np.mean((frames <= 1) | (frames >= 254)))
        modified_guard = full_guard_metrics(render_path, spec, scorers)
        guard = full_guard_verdict(baseline_guard, modified_guard)
        automatic_pass = bool(
            alpha in eligible_numeric
            and integrity
            and same_shape
            and abs(fps - baseline_fps) < 1e-3
            and prefix_mae <= 1.0
            and black_fraction == 0.0
            and clipped_fraction < 0.50
            and max(ratios, default=math.inf) <= 2.5
            and guard["passed"]
        )
        records.append(
            {
                "alpha": alpha,
                "render": str(render_path.relative_to(ROOT)),
                "runtime": runtime,
                "frame_count": int(frames.shape[0]),
                "resolution": [int(frames.shape[2]), int(frames.shape[1])],
                "fps": fps,
                "pre_edit_prefix_pixel_mae": prefix_mae,
                "boundary_temporal_l1": jumps,
                "boundary_jump_ratio_to_baseline": ratios,
                "black_frame_fraction": black_fraction,
                "clipped_pixel_fraction": clipped_fraction,
                "full_round18_guard": guard,
                "automatic_stability_pass": automatic_pass,
            }
        )
        visual_rows.append((f"alpha={alpha:g}", frames))
        video_rows.append((f"alpha={alpha:g}", render_path))
    if len(visual_rows) != 4:
        raise RuntimeError("the protocol requires baseline plus all three alpha candidates")
    indices = [start + (end - start) // 2 for start, end in ranges]
    sheet = scan_root / "alpha_contact_sheet.png"
    grid = scan_root / "alpha_4up.mp4"
    build_contact_sheet(sheet, visual_rows, indices)
    build_grid(grid, video_rows)
    result = {
        "schema_version": 1,
        "gate": "P2_rendered_alpha_stability_prefilter",
        "scene": scene,
        "baseline": str(baseline_path.relative_to(ROOT)),
        "boundaries": boundaries,
        "baseline_boundary_temporal_l1": baseline_jumps,
        "candidates": records,
        "automatic_gate_pass": any(record["automatic_stability_pass"] for record in records),
        "contact_sheet": str(sheet.relative_to(ROOT)),
        "four_up_video": str(grid.relative_to(ROOT)),
        "visual_review_required": True,
    }
    (scan_root / "rendered_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    verdict_path = scan_root / "visual_verdict.json"
    if not verdict_path.exists():
        verdict = {
            "schema_version": 1,
            "complete": False,
            "reviewer": None,
            "required_view_order": [
                str(sheet.relative_to(ROOT)),
                str(grid.relative_to(ROOT)) + " at real-time speed",
            ],
            "candidates": {
                str(record["alpha"]): {"accepted": None, "evidence": None}
                for record in records
            },
            "selected_alpha": None,
            "rationale": None,
        }
        verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"automatic_gate_pass": result["automatic_gate_pass"], "visual_verdict": str(verdict_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
