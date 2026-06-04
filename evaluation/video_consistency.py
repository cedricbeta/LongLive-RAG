# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Quantitative temporal-consistency metrics for generated videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency in some envs
    structural_similarity = None


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def read_video(path: str | Path, *, max_frames: int | None = None, stride: int = 1) -> np.ndarray:
    """Read a video into uint8 RGB frames with shape [T, H, W, 3]."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def discover_videos(directory: str | Path, extensions: Iterable[str] = VIDEO_EXTENSIONS) -> list[Path]:
    directory = Path(directory)
    exts = {ext.lower() for ext in extensions}
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in exts)


def pair_video_dirs(baseline_dir: str | Path, rag_dir: str | Path) -> list[tuple[Path, Path]]:
    baseline = discover_videos(baseline_dir)
    rag = discover_videos(rag_dir)
    baseline_by_stem = {path.stem: path for path in baseline}
    rag_by_stem = {path.stem: path for path in rag}
    common = sorted(set(baseline_by_stem) & set(rag_by_stem))
    if common:
        return [(baseline_by_stem[stem], rag_by_stem[stem]) for stem in common]
    return list(zip(baseline, rag))


def evaluate_video(path: str | Path, *, max_frames: int | None = None, stride: int = 1) -> dict[str, float]:
    frames = read_video(path, max_frames=max_frames, stride=stride)
    return compute_video_metrics(frames)


def compute_video_metrics(frames: np.ndarray) -> dict[str, float]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video [T,H,W,3], got shape={frames.shape}")
    if frames.shape[0] < 2:
        raise ValueError("Need at least two frames to compute temporal metrics")

    frame_f = frames.astype(np.float32) / 255.0
    adjacent_l1 = []
    adjacent_psnr = []
    adjacent_ssim = []
    flow_l1 = []
    flow_ssim = []
    feature_adjacent = []
    feature_to_first = []
    appearance_to_first = []

    features = np.stack([frame_feature(frame) for frame in frames], axis=0)
    first_feature = features[0]
    first_frame = frame_f[0]

    for idx in range(frames.shape[0] - 1):
        prev = frame_f[idx]
        nxt = frame_f[idx + 1]
        diff = np.abs(prev - nxt)
        mse = float(np.mean((prev - nxt) ** 2))
        adjacent_l1.append(float(diff.mean()))
        adjacent_psnr.append(_psnr_from_mse(mse))
        adjacent_ssim.append(_ssim(prev, nxt))
        flow_metrics = optical_flow_warp_metrics(frames[idx], frames[idx + 1])
        flow_l1.append(flow_metrics["flow_warp_l1"])
        flow_ssim.append(flow_metrics["flow_warp_ssim"])
        feature_adjacent.append(cosine_similarity(features[idx], features[idx + 1]))

    for idx in range(1, frames.shape[0]):
        feature_to_first.append(cosine_similarity(first_feature, features[idx]))
        appearance_to_first.append(_ssim(first_frame, frame_f[idx]))

    return {
        "temporal_adjacent_l1": _mean(adjacent_l1),
        "temporal_adjacent_psnr": _mean(adjacent_psnr),
        "temporal_adjacent_ssim": _mean(adjacent_ssim),
        "optical_flow_warp_l1": _mean(flow_l1),
        "optical_flow_warp_ssim": _mean(flow_ssim),
        "frame_feature_adjacent_cosine": _mean(feature_adjacent),
        "frame_feature_to_first_cosine": _mean(feature_to_first),
        "appearance_to_first_ssim": _mean(appearance_to_first),
        "num_frames": float(frames.shape[0]),
    }


def compare_video_dirs(
    baseline_dir: str | Path,
    rag_dir: str | Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, object]:
    pairs = pair_video_dirs(baseline_dir, rag_dir)
    if not pairs:
        raise ValueError(f"No comparable videos found in {baseline_dir} and {rag_dir}")

    records = []
    for baseline_path, rag_path in pairs:
        baseline_metrics = evaluate_video(baseline_path, max_frames=max_frames, stride=stride)
        rag_metrics = evaluate_video(rag_path, max_frames=max_frames, stride=stride)
        deltas = {
            key: rag_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if key in rag_metrics and key != "num_frames"
        }
        records.append(
            {
                "baseline": str(baseline_path),
                "kv_rag": str(rag_path),
                "baseline_metrics": baseline_metrics,
                "kv_rag_metrics": rag_metrics,
                "delta": deltas,
            }
        )

    return {
        "num_pairs": len(records),
        "baseline_summary": summarize_records(records, "baseline_metrics"),
        "kv_rag_summary": summarize_records(records, "kv_rag_metrics"),
        "delta_summary": summarize_records(records, "delta"),
        "records": records,
    }


def summarize_records(records: list[dict[str, object]], key: str) -> dict[str, dict[str, float | int]]:
    metric_names = sorted(records[0][key].keys())
    summary = {}
    for name in metric_names:
        values = np.array([record[key][name] for record in records], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "count": int(values.size),
        }
    return summary


def save_metrics_json(result: dict[str, object], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Cross-perspective scene consistency (single scene, multiple camera angles)
# ---------------------------------------------------------------------------
#
# A multi-shot video can depict ONE scene from DIFFERENT viewpoints. We want
# two things to be true at once and we report them separately so neither can
# hide a failure of the other:
#   * scene consistency  -- successive shots read as the SAME place (high),
#     measured with a framing-robust global color signature;
#   * viewpoint variation -- the shots are genuinely DIFFERENT framings (not a
#     frozen copy of shot 0), measured with a framing-sensitive layout signature.
# A model that cheats by repeating shot 0 maxes out scene consistency but
# collapses viewpoint variation and within-shot motion, so the collapse is
# visible rather than rewarded.


def scene_color_signature(frame_rgb: np.ndarray) -> np.ndarray:
    """Framing-robust global HSV color histogram (normalized).

    Camera moves rearrange where things are but largely preserve the palette of
    the scene, so this is a viewpoint-robust "same place" signal.
    """
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]).astype(np.float32)
    hist = hist.reshape(-1)
    hist /= max(float(hist.sum()), 1e-6)
    return hist


def composition_signature(frame_rgb: np.ndarray) -> np.ndarray:
    """Framing-SENSITIVE spatial layout: downsampled luminance + edge map.

    A different camera angle changes the spatial arrangement, so this signal
    separates genuine viewpoint changes from a repeated frame.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.resize(np.sqrt(sobel_x ** 2 + sobel_y ** 2), (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    feature = np.concatenate([small, edge], axis=0).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def shot_ranges(num_frames: int, *, num_shots: int | None = None,
                boundaries: list[int] | None = None) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` frame ranges per shot.

    ``boundaries`` (sorted frame indices where each shot starts, first need not
    be 0) takes precedence; otherwise the frames are split into ``num_shots``
    contiguous, near-equal segments (the vendored multi-view prompts use equal
    shot durations).
    """
    if num_frames <= 0:
        return []
    if boundaries:
        starts = sorted({0, *[b for b in boundaries if 0 < b < num_frames]})
        ends = starts[1:] + [num_frames]
        return list(zip(starts, ends))
    shots = max(1, int(num_shots or 1))
    shots = min(shots, num_frames)
    edges = [round(i * num_frames / shots) for i in range(shots + 1)]
    return [(edges[i], edges[i + 1]) for i in range(shots) if edges[i + 1] > edges[i]]


def _mean_pairwise(features: np.ndarray) -> float:
    """Mean pairwise cosine similarity over rows of ``features`` ([N, D])."""
    n = features.shape[0]
    if n < 2:
        return float("nan")
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(features[i], features[j]))
    return float(np.mean(sims)) if sims else float("nan")


def cross_perspective_consistency(scene_feats: np.ndarray, comp_feats: np.ndarray) -> dict[str, float]:
    """Pure scoring over per-shot feature means (no video decode).

    Args:
        scene_feats: ``[num_shots, D_scene]`` per-shot framing-robust signatures.
        comp_feats:  ``[num_shots, D_comp]`` per-shot framing-sensitive signatures.
    """
    scene_feats = np.asarray(scene_feats, dtype=np.float64)
    comp_feats = np.asarray(comp_feats, dtype=np.float64)
    scene_consistency = _mean_pairwise(scene_feats)
    comp_similarity = _mean_pairwise(comp_feats)
    # diversity is high when shots are framed differently; ~0 means copy-collapse
    diversity = float("nan") if np.isnan(comp_similarity) else 1.0 - comp_similarity
    return {
        "cross_shot_scene_consistency": scene_consistency,
        "inter_shot_composition_diversity": diversity,
        "num_shots": float(scene_feats.shape[0]),
    }


def cross_perspective_metrics(
    frames: np.ndarray,
    *,
    num_shots: int | None = None,
    boundaries: list[int] | None = None,
) -> dict[str, float]:
    """Cross-perspective scene-consistency metrics for one multi-shot video.

    Reports scene consistency together with two anti-cheating companions
    (viewpoint variation and within-shot motion) so a degenerate "copy shot 0"
    output is visible instead of scoring a fake consistency win.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video [T,H,W,3], got shape={frames.shape}")
    ranges = shot_ranges(frames.shape[0], num_shots=num_shots, boundaries=boundaries)
    if len(ranges) < 2:
        raise ValueError("Need at least two shots to measure cross-perspective consistency")

    scene_per_frame = np.stack([scene_color_signature(f) for f in frames], axis=0)
    comp_per_frame = np.stack([composition_signature(f) for f in frames], axis=0)

    scene_feats, comp_feats, within_motion = [], [], []
    for start, end in ranges:
        scene_feats.append(scene_per_frame[start:end].mean(axis=0))
        comp_feats.append(comp_per_frame[start:end].mean(axis=0))
        if end - start >= 2:
            seg = comp_per_frame[start:end]
            within_motion.append(float(np.mean(np.abs(np.diff(seg, axis=0)))))

    result = cross_perspective_consistency(np.stack(scene_feats), np.stack(comp_feats))
    result["within_shot_motion"] = float(np.mean(within_motion)) if within_motion else 0.0
    return result


def cross_perspective_evaluate_video(
    path: str | Path,
    *,
    num_shots: int | None = None,
    boundaries: list[int] | None = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, float]:
    frames = read_video(path, max_frames=max_frames, stride=stride)
    return cross_perspective_metrics(frames, num_shots=num_shots, boundaries=boundaries)


def compare_cross_perspective_dirs(
    baseline_dir: str | Path,
    modified_dir: str | Path,
    *,
    shots_for: "callable | int | None" = None,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, object]:
    """Baseline-vs-modified cross-perspective comparison over paired videos.

    ``shots_for`` resolves the shot count per video: an int (same for all), a
    callable ``stem -> int``, or None (auto = no split, which is rejected). The
    delta on ``cross_shot_scene_consistency`` is the consistency win; the
    companion deltas expose any viewpoint-variation / motion collapse.
    """
    pairs = pair_video_dirs(baseline_dir, modified_dir)
    if not pairs:
        raise ValueError(f"No comparable videos found in {baseline_dir} and {modified_dir}")

    def resolve_shots(stem: str) -> int | None:
        if callable(shots_for):
            return shots_for(stem)
        if isinstance(shots_for, int):
            return shots_for
        return None

    records = []
    for baseline_path, modified_path in pairs:
        stem = baseline_path.stem
        n = resolve_shots(stem)
        baseline_metrics = cross_perspective_evaluate_video(
            baseline_path, num_shots=n, max_frames=max_frames, stride=stride
        )
        modified_metrics = cross_perspective_evaluate_video(
            modified_path, num_shots=n, max_frames=max_frames, stride=stride
        )
        deltas = {
            key: modified_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if key in modified_metrics and key != "num_shots"
        }
        records.append(
            {
                "baseline": str(baseline_path),
                "modified": str(modified_path),
                "baseline_metrics": baseline_metrics,
                "modified_metrics": modified_metrics,
                "delta": deltas,
            }
        )

    return {
        "num_pairs": len(records),
        "baseline_summary": summarize_records(records, "baseline_metrics"),
        "modified_summary": summarize_records(records, "modified_metrics"),
        "delta_summary": summarize_records(records, "delta"),
        "records": records,
    }


def frame_feature(frame_rgb: np.ndarray) -> np.ndarray:
    """Extract a deterministic appearance feature without model downloads."""
    small = cv2.resize(frame_rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256]).astype(np.float32)
    hist = hist.reshape(-1)
    hist /= max(float(hist.sum()), 1e-6)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.resize(np.sqrt(sobel_x ** 2 + sobel_y ** 2), (16, 16), interpolation=cv2.INTER_AREA)
    feature = np.concatenate([small.reshape(-1), hist, edge_mag.reshape(-1)], axis=0)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature.astype(np.float32)


def optical_flow_warp_metrics(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> dict[str, float]:
    prev = prev_rgb.astype(np.float32) / 255.0
    nxt = next_rgb.astype(np.float32) / 255.0
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)

    # Backward flow maps each next-frame pixel to its source in prev.
    flow_back = cv2.calcOpticalFlowFarneback(
        next_gray,
        prev_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    h, w = prev_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_back[..., 0]
    map_y = grid_y + flow_back[..., 1]
    warped_prev = cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return {
        "flow_warp_l1": float(np.abs(warped_prev - nxt).mean()),
        "flow_warp_ssim": _ssim(warped_prev, nxt),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    if structural_similarity is None:
        return 1.0 - float(np.mean(np.abs(a - b)))
    return float(
        structural_similarity(
            a,
            b,
            channel_axis=-1,
            data_range=1.0,
        )
    )


def _psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.array(values, dtype=np.float64)))
