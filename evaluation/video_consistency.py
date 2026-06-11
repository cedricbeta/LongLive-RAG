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
import re
from pathlib import Path
from typing import Callable, Iterable

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


# A generated stem is "<prefix>-rank<R>-<name>-seed<S>_<model>"; the variant
# prefix (baseline/kv_rag) is the ONLY part that differs between the two sides,
# so the pair key drops it (negative lookahead so a "rank0-" with no prefix is
# not mistaken for the prefix).
_PAIR_KEY_RE = re.compile(r"^(?:(?!rank\d)[^-]+-)?(rank\d+-.+-seed\d+_[^-]*)$")


def cross_perspective_pair_key(stem: str) -> str:
    """Variant-independent pairing key for a generated video stem.

    Strips the leading ``baseline-``/``kv_rag-`` (or any non-``rank`` prefix) so
    the matching baseline and modified renders of the SAME prompt/rank/seed share
    a key. Falls back to the full stem when the name is not in the generated
    format (then pairing degrades to exact-stem matching, never silent zip).
    """
    m = _PAIR_KEY_RE.match(stem)
    return m.group(1) if m else stem


def pair_cross_perspective_dirs(
    baseline_dir: str | Path, modified_dir: str | Path
) -> list[tuple[Path, Path]]:
    """Pair baseline/modified videos by variant-independent key.

    Unlike ``pair_video_dirs`` there is NO zip fallback: the official gate
    outputs differ only by prefix, so zip pairing could silently compare
    different prompts after a missing/extra render. Raises on a duplicate key
    within a directory (ambiguous) or any key present on only one side
    (missing/extra).
    """
    def index(videos: list[Path], label: str) -> dict[str, Path]:
        by_key: dict[str, Path] = {}
        for path in videos:
            key = cross_perspective_pair_key(path.stem)
            if key in by_key:
                raise ValueError(
                    f"Ambiguous {label} videos for pair key {key!r}: "
                    f"{by_key[key].name} and {path.name}"
                )
            by_key[key] = path
        return by_key

    baseline = index(discover_videos(baseline_dir), "baseline")
    modified = index(discover_videos(modified_dir), "modified")
    only_baseline = sorted(set(baseline) - set(modified))
    only_modified = sorted(set(modified) - set(baseline))
    if only_baseline or only_modified:
        raise ValueError(
            "Unpaired cross-perspective videos (a missing or extra render would "
            f"silently misalign the gate): baseline-only={only_baseline}, "
            f"modified-only={only_modified}"
        )
    if not baseline:
        raise ValueError(
            f"No videos found to pair in {baseline_dir} and {modified_dir}"
        )
    return [(baseline[k], modified[k]) for k in sorted(baseline)]


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


def chunk_durations_to_boundaries(
    num_frames: int, chunk_durations: list[int]
) -> list[int] | None:
    """Map per-shot chunk counts to decoded-frame shot-start boundaries.

    The prompt sets specify each shot as a number of generation chunks (e.g.
    ``shot_durations.txt`` = ``7 7 7 7 7 7 6``), which need not be equal. After
    decode the video has ``num_frames`` frames; a shot that owns a fraction of
    the chunks owns the same fraction of the frames, so shot ``i`` starts at
    ``round(cumulative_chunks_before_i * num_frames / total_chunks)``. Returns
    the interior shot-start indices (shot 0 always starts at 0); ``None`` when
    there is nothing to split.
    """
    durations = [int(d) for d in (chunk_durations or []) if int(d) > 0]
    total = sum(durations)
    if total <= 0 or len(durations) < 2 or num_frames <= 0:
        return None
    boundaries: list[int] = []
    cumulative = 0
    for d in durations[:-1]:
        cumulative += d
        boundaries.append(int(round(cumulative * num_frames / total)))
    return boundaries


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


def _centroid_cosine(features: np.ndarray) -> float:
    """EntityBench-style mean cosine of each unit embedding to the scene centroid."""
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.clip(norms, 1e-8, None)
    centroid = feats.mean(axis=0)
    n = np.linalg.norm(centroid)
    if n <= 1e-12:
        return float("nan")
    centroid = centroid / n
    return float(np.mean(feats @ centroid))


def _shot_sample(frames: np.ndarray, start: int, end: int, count: int = 3) -> np.ndarray:
    """Evenly sample representative frames from one shot range."""
    if end <= start:
        return frames[start:start]
    k = max(1, min(int(count), end - start))
    idx = np.linspace(start, end - 1, num=k).round().astype(int)
    return frames[idx]


def shot_anchor_centroid_consistency(
    subject_embeddings: np.ndarray,
    background_embeddings: np.ndarray,
) -> dict[str, float]:
    """Cross-shot centroid score from per-shot DINO subject + CLIP background anchors."""
    subject = _centroid_cosine(subject_embeddings)
    background = _centroid_cosine(background_embeddings)
    vals = [v for v in (subject, background) if not np.isnan(v)]
    aggregate = float(np.mean(vals)) if len(vals) == 2 else float("nan")
    return {
        "subject_anchor_consistency": subject,
        "background_anchor_consistency": background,
        "anchor_centroid_consistency": aggregate,
    }


def shot_anchor_metrics(
    frames: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    subject_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
    background_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
    frames_per_shot: int = 3,
) -> dict[str, float]:
    """Per-shot anchor embeddings and centroid agreement.

    ``subject_encoder`` is the DINO image encoder and ``background_encoder`` is
    the CLIP image encoder used by the VBench dimensions. Each shot is reduced
    to one mean-normalized embedding per backbone, then scored against the
    per-scene centroid. If either backbone is absent the gate treats the result
    as unscorable and fails closed.
    """
    if subject_encoder is None or background_encoder is None:
        return {
            "subject_anchor_consistency": float("nan"),
            "background_anchor_consistency": float("nan"),
            "anchor_centroid_consistency": float("nan"),
        }

    def encode_video_shot(encoder, sample):
        feats = np.asarray(encoder(sample), dtype=np.float64)
        if feats.ndim != 2 or feats.shape[0] == 0:
            return np.full((1,), np.nan, dtype=np.float64)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-8, None)
        emb = feats.mean(axis=0)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    subject, background = [], []
    for start, end in ranges:
        sample = _shot_sample(frames, start, end, frames_per_shot)
        if sample.shape[0] == 0:
            continue
        subject.append(encode_video_shot(subject_encoder, sample))
        background.append(encode_video_shot(background_encoder, sample))
    if len(subject) < 2 or len(background) < 2:
        return {
            "subject_anchor_consistency": float("nan"),
            "background_anchor_consistency": float("nan"),
            "anchor_centroid_consistency": float("nan"),
        }
    return shot_anchor_centroid_consistency(np.stack(subject), np.stack(background))


def motion_profile_signature(frames: np.ndarray, *, bins: int = 8) -> np.ndarray:
    """Per-shot motion-profile proxy: histogram of frame-to-frame luminance change."""
    if frames.ndim != 4 or frames.shape[0] < 2:
        return np.zeros(bins + 2, dtype=np.float64)
    grays = np.stack([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (32, 32),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for f in frames
    ], axis=0)
    mags = np.abs(np.diff(grays, axis=0)).reshape(-1)
    hist, _ = np.histogram(mags, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1e-8)
    sig = np.concatenate([hist, [float(mags.mean()), float(mags.std())]])
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 0 else sig


def farneback_dynamic_degree(frames: np.ndarray, *, sample: int = 12) -> float:
    """CPU optical-flow dynamic-degree proxy used by unit tests.

    The rendered gate can pass a RAFT scorer via ``dynamic_scorer``; this
    Farneback path is deterministic and dependency-light for GPU-free tests.
    """
    if frames.ndim != 4 or frames.shape[0] < 2:
        return 0.0
    idx = np.linspace(0, frames.shape[0] - 1, num=min(sample + 1, frames.shape[0]))
    idx = np.unique(idx.round().astype(int))
    mags = []
    for a, b in zip(idx[:-1], idx[1:]):
        g0 = cv2.cvtColor(frames[a], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[b], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))))
    return float(np.mean(mags)) if mags else 0.0


def build_raft_dynamic_scorer(*, device: str | None = None, sample: int = 12):
    """Return ``scorer(frames_rgb) -> dynamic_degree`` using torchvision RAFT."""
    import torch
    import torch.nn.functional as torch_f
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(dev).eval()
    transforms = weights.transforms()

    def _prep(frame: np.ndarray):
        x = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        x = x.to(dev) / 255.0
        h, w = x.shape[-2:]
        nh = max(64, int(np.ceil(h / 8.0) * 8))
        nw = max(64, int(np.ceil(w / 8.0) * 8))
        if (nh, nw) != (h, w):
            x = torch_f.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        return x

    def _score(frames: np.ndarray) -> float:
        if frames.ndim != 4 or frames.shape[0] < 2:
            return 0.0
        idx = np.linspace(0, frames.shape[0] - 1, num=min(sample + 1, frames.shape[0]))
        idx = np.unique(idx.round().astype(int))
        mags: list[float] = []
        with torch.no_grad():
            for a, b in zip(idx[:-1], idx[1:]):
                img1, img2 = transforms(_prep(frames[a]), _prep(frames[b]))
                flow = model(img1, img2)[-1]
                mags.append(float(flow.norm(dim=1).mean().item()))
        return float(np.mean(mags)) if mags else 0.0

    return _score


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
        "palette_agreement": scene_consistency,
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

    scene_feats, comp_feats, motion_feats, within_motion = [], [], [], []
    for start, end in ranges:
        scene_feats.append(scene_per_frame[start:end].mean(axis=0))
        comp_feats.append(comp_per_frame[start:end].mean(axis=0))
        motion_feats.append(motion_profile_signature(frames[start:end]))
        if end - start >= 2:
            seg = comp_per_frame[start:end]
            within_motion.append(float(np.mean(np.abs(np.diff(seg, axis=0)))))

    result = cross_perspective_consistency(np.stack(scene_feats), np.stack(comp_feats))
    result["motion_profile_agreement"] = _mean_pairwise(np.stack(motion_feats))
    result["within_shot_motion"] = float(np.mean(within_motion)) if within_motion else 0.0
    result["dynamic_degree"] = farneback_dynamic_degree(frames)
    return result


# ---------------------------------------------------------------------------
# Prompt-adherence guard (milestone-only; needs a frozen CLIP backbone)
# ---------------------------------------------------------------------------
#
# The cross-perspective metric rewards "same place across shots", which a
# degenerate model can game by freezing on shot 0. The companion diversity /
# motion metrics expose a *copy* collapse, but not a model that stays coherent
# while drifting away from each shot's prompt. The adherence guard scores each
# shot's frames against its caption with a frozen CLIP encoder so the milestone
# gate can require that the modified pipeline does not regress per-shot prompt
# adherence. It is OPT-IN and never imported on the per-round CPU test path.


def build_clip_adherence_scorer(
    *, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None
):
    """Return ``scorer(frames_rgb, caption) -> mean image/text cosine``.

    Tries ``open_clip`` then HuggingFace ``transformers`` CLIP. Raises a clear
    error if neither backend (or its checkpoint) is available, so a milestone
    gate fails loudly instead of silently skipping the guard.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - torch is present in real runs
        raise RuntimeError(
            "Prompt-adherence scoring requires PyTorch, which is not importable."
        ) from exc

    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Preferred backend: open_clip. Fall back to transformers on ANY failure
    # (not just ImportError) so an offline box with a cached transformers CLIP
    # but no open_clip weights still scores adherence instead of hard-failing.
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(resolved_device).eval()

        def _score_open_clip(frames_rgb: np.ndarray, caption: str) -> float:
            from PIL import Image

            images = torch.stack(
                [preprocess(Image.fromarray(f.astype(np.uint8))) for f in frames_rgb]
            ).to(resolved_device)
            text = tokenizer([caption]).to(resolved_device)
            with torch.no_grad():
                img_feat = model.encode_image(images)
                txt_feat = model.encode_text(text)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ txt_feat.T).squeeze(-1)
            return float(sims.mean().item())

        return _score_open_clip
    except Exception:
        pass

    # Fallback backend: transformers CLIP.
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Prompt-adherence scoring needs a CLIP backend. Install `open_clip_torch` "
            "or `transformers` (plus a downloaded CLIP checkpoint)."
        ) from exc

    hf_name = "openai/clip-vit-base-patch32"
    try:
        model = CLIPModel.from_pretrained(hf_name).to(resolved_device).eval()
        processor = CLIPProcessor.from_pretrained(hf_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the CLIP checkpoint {hf_name!r} for prompt-adherence "
            "scoring; ensure it is downloaded/cached."
        ) from exc

    def _score_transformers(frames_rgb: np.ndarray, caption: str) -> float:
        from PIL import Image

        images = [Image.fromarray(f.astype(np.uint8)) for f in frames_rgb]
        inputs = processor(
            text=[caption], images=images, return_tensors="pt", padding=True, truncation=True
        ).to(resolved_device)
        with torch.no_grad():
            out = model(**inputs)
            img_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_feat = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T).squeeze(-1)
        return float(sims.mean().item())

    return _score_transformers


def build_clip_text_encoder(
    *, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None
):
    """Return ``encoder(texts) -> [N, D]`` CLIP text embeddings for prompt lint."""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CLIP text lint requires PyTorch.") from exc

    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(resolved_device).eval()

        def _encode_open_clip(texts):
            text = tokenizer(list(texts)).to(resolved_device)
            with torch.no_grad():
                feat = model.encode_text(text)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.float().cpu().numpy()

        return _encode_open_clip
    except Exception:
        pass

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "CLIP text lint needs `open_clip_torch` or `transformers`."
        ) from exc

    hf_name = "openai/clip-vit-base-patch32"
    try:
        model = CLIPModel.from_pretrained(hf_name).to(resolved_device).eval()
        processor = CLIPProcessor.from_pretrained(hf_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {hf_name!r} for CLIP text lint; ensure it is cached."
        ) from exc

    def _encode_transformers(texts):
        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(resolved_device)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().cpu().numpy()

    return _encode_transformers


def prompt_text_similarity_lint(
    captions: list[str],
    text_encoder: Callable[[list[str]], np.ndarray] | None,
    *,
    floor: float,
) -> dict[str, object]:
    """Pairwise CLIP text-similarity matrix for shot captions."""
    if text_encoder is None or len(captions) < 2:
        return {
            "matrix": [],
            "min_pairwise": float("nan"),
            "floor": float(floor),
            "passed": False,
            "blocked_reason": "CLIP text encoder unavailable" if text_encoder is None else "fewer than two captions",
        }
    emb = np.asarray(text_encoder(captions), dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-8, None)
    matrix = emb @ emb.T
    off_diag = [float(matrix[i, j]) for i in range(matrix.shape[0])
                for j in range(matrix.shape[1]) if i < j]
    min_pairwise = float(np.min(off_diag)) if off_diag else float("nan")
    passed = not np.isnan(min_pairwise) and min_pairwise >= floor
    out = {
        "matrix": [[float(v) for v in row] for row in matrix],
        "min_pairwise": min_pairwise,
        "floor": float(floor),
        "passed": bool(passed),
    }
    if not passed:
        out["blocked_reason"] = "shot captions below CLIP text-similarity floor"
    return out


def prompt_adherence_for_video(
    frames: np.ndarray,
    ranges: list[tuple[int, int]],
    captions: list[str],
    scorer,
    *,
    frames_per_shot: int = 3,
) -> dict[str, float]:
    """Mean/min per-shot CLIP adherence of a video to its shot captions.

    A few evenly-spaced representative frames per shot are scored against that
    shot's caption; ``prompt_adherence_min`` is the weakest shot (so a single
    drifting shot is visible, not averaged away).
    """
    per_shot: list[float] = []
    for (start, end), caption in zip(ranges, captions):
        if not caption or end <= start:
            continue
        k = max(1, min(int(frames_per_shot), end - start))
        idx = np.linspace(start, end - 1, num=k).round().astype(int)
        per_shot.append(float(scorer(frames[idx], caption)))
    if not per_shot:
        return {"prompt_adherence_mean": float("nan"), "prompt_adherence_min": float("nan")}
    return {
        "prompt_adherence_mean": float(np.mean(per_shot)),
        "prompt_adherence_min": float(np.min(per_shot)),
    }


def invariant_probe_for_video(
    frames: np.ndarray,
    ranges: list[tuple[int, int]],
    invariant_caption: str | None,
    contrast_caption: str | None,
    scorer,
    *,
    frames_per_shot: int = 3,
) -> dict[str, float]:
    """Per-shot CLIP margin: score(invariant) - score(contrast)."""
    if scorer is None or not invariant_caption or not contrast_caption:
        return {"invariant_margin_mean": float("nan"), "invariant_margin_min": float("nan")}
    margins: list[float] = []
    for start, end in ranges:
        sample = _shot_sample(frames, start, end, frames_per_shot)
        if sample.shape[0] == 0:
            continue
        inv = float(scorer(sample, invariant_caption))
        con = float(scorer(sample, contrast_caption))
        margins.append(inv - con)
    if not margins:
        return {"invariant_margin_mean": float("nan"), "invariant_margin_min": float("nan")}
    return {
        "invariant_margin_mean": float(np.mean(margins)),
        "invariant_margin_min": float(np.min(margins)),
    }


def cross_perspective_evaluate_video(
    path: str | Path,
    *,
    num_shots: int | None = None,
    boundaries: list[int] | None = None,
    chunk_durations: list[int] | None = None,
    captions: list[str] | None = None,
    adherence_scorer=None,
    invariant_caption: str | None = None,
    contrast_caption: str | None = None,
    invariant_scorer=None,
    subject_encoder=None,
    background_encoder=None,
    dynamic_scorer=None,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, float]:
    """Cross-perspective metrics (plus optional adherence) for one video.

    ``chunk_durations`` (per-shot generation-chunk counts) are converted to
    exact decoded-frame boundaries after the decode, taking precedence over
    ``boundaries``/``num_shots`` so uneven shot lengths are honored. When an
    ``adherence_scorer`` and per-shot ``captions`` are supplied the prompt-
    adherence guard metrics are merged in.
    """
    frames = read_video(path, max_frames=max_frames, stride=stride)
    if chunk_durations:
        derived = chunk_durations_to_boundaries(frames.shape[0], chunk_durations)
        if derived is not None:
            boundaries = derived
            num_shots = None
    metrics = cross_perspective_metrics(frames, num_shots=num_shots, boundaries=boundaries)
    ranges = shot_ranges(frames.shape[0], num_shots=num_shots, boundaries=boundaries)
    metrics.update(
        shot_anchor_metrics(
            frames, ranges,
            subject_encoder=subject_encoder,
            background_encoder=background_encoder,
        )
    )
    if dynamic_scorer is not None:
        metrics["dynamic_degree"] = float(dynamic_scorer(frames))
    if adherence_scorer is not None and captions:
        metrics.update(
            prompt_adherence_for_video(frames, ranges, captions, adherence_scorer)
        )
    if invariant_scorer is not None:
        metrics.update(
            invariant_probe_for_video(
                frames, ranges, invariant_caption, contrast_caption, invariant_scorer
            )
        )
    return metrics


def compare_cross_perspective_dirs(
    baseline_dir: str | Path,
    modified_dir: str | Path,
    *,
    shots_for: "callable | int | None" = None,
    adherence_scorer=None,
    invariant_scorer=None,
    subject_encoder=None,
    background_encoder=None,
    dynamic_scorer=None,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, object]:
    """Baseline-vs-modified cross-perspective comparison over paired videos.

    ``shots_for`` resolves how to split each video, by stem. It may be an int
    (same shot count for all), or a callable ``stem -> spec`` where ``spec`` is
    one of: an int (shot count); a mapping with ``chunk_durations`` (exact,
    possibly uneven shot lengths) and optional ``captions``; or ``None`` (no
    match -> the pair is skipped). The delta on ``anchor_centroid_consistency``
    is the gated consistency win; proxy/companion deltas expose palette,
    viewpoint-variation, and motion collapse, and (when ``adherence_scorer`` is
    given) prompt-adherence deltas drive the milestone non-regression guard.
    """
    pairs = pair_cross_perspective_dirs(baseline_dir, modified_dir)

    def resolve_spec(stem: str):
        spec = shots_for(stem) if callable(shots_for) else shots_for
        if spec is None:
            return None
        if isinstance(spec, int):
            return {"num_shots": spec}
        if isinstance(spec, dict):
            return spec
        raise TypeError(f"Unsupported shot spec for {stem!r}: {type(spec)!r}")

    records = []
    skipped: list[str] = []
    for baseline_path, modified_path in pairs:
        stem = baseline_path.stem
        spec = resolve_spec(stem)
        if spec is None:
            skipped.append(stem)
            continue
        kwargs = dict(
            num_shots=spec.get("num_shots"),
            chunk_durations=spec.get("chunk_durations"),
            captions=spec.get("captions"),
            adherence_scorer=adherence_scorer,
            invariant_caption=spec.get("invariant_caption"),
            contrast_caption=spec.get("contrast_caption"),
            invariant_scorer=invariant_scorer,
            subject_encoder=subject_encoder,
            background_encoder=background_encoder,
            dynamic_scorer=dynamic_scorer,
            max_frames=max_frames,
            stride=stride,
        )
        baseline_metrics = cross_perspective_evaluate_video(baseline_path, **kwargs)
        modified_metrics = cross_perspective_evaluate_video(modified_path, **kwargs)
        deltas = {
            key: modified_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if key in modified_metrics and key != "num_shots"
        }
        records.append(
            {
                "stem": stem,
                "baseline": str(baseline_path),
                "modified": str(modified_path),
                "theme": spec.get("theme"),
                "negative_control": bool(spec.get("negative_control", False)),
                "baseline_metrics": baseline_metrics,
                "modified_metrics": modified_metrics,
                "delta": deltas,
            }
        )

    if not records:
        raise ValueError(
            "No videos could be matched to a shot spec. Check --prompts_dir / "
            f"--num_shots and the generated filenames (skipped stems: {skipped})."
        )

    return {
        "num_pairs": len(records),
        "skipped_stems": skipped,
        "baseline_summary": summarize_records(records, "baseline_metrics"),
        "modified_summary": summarize_records(records, "modified_metrics"),
        "delta_summary": summarize_records(records, "delta"),
        "records": records,
    }


def evaluate_cross_perspective_gate(
    result: dict[str, object],
    *,
    metric: str = "anchor_centroid_consistency",
    min_consistency_wins: int = 2,
    adherence_tolerance: float = 0.0,
    diversity_tolerance: float = 0.0,
    motion_tolerance: float | None = None,
    invariant_tolerance: float = 0.0,
    require_adherence: bool = False,
    require_invariant: bool = True,
) -> dict[str, object]:
    """Decide the guarded long-multishot gate from a comparison ``result``."""
    records = result.get("records", [])
    if not records:
        return {
            "passed": False,
            "is_null_result": True,
            "blocked_reason": "no comparable long_multishot records",
            "metric": metric,
            "num_pairs": 0,
            "consistency_wins": 0,
            "min_consistency_wins": min_consistency_wins,
            "consistency_ok": False,
            "require_adherence": bool(require_adherence),
            "adherence_tolerance": adherence_tolerance,
            "adherence_ok": False if require_adherence else True,
            "adherence_failures": [],
            "diversity_tolerance_relative": diversity_tolerance,
            "diversity_ok": False,
            "diversity_failures": [],
            "motion_tolerance_relative": motion_tolerance,
            "motion_ok": False,
            "motion_failures": [],
            "require_invariant": bool(require_invariant),
            "invariant_tolerance": invariant_tolerance,
            "invariant_ok": False if require_invariant else True,
            "invariant_failures": [],
            "scorable_ok": False,
            "unscorable": ["records"],
            "negative_control_ok": True,
            "negative_control_failures": [],
            "per_prompt": [],
        }
    per_prompt = []
    consistency_wins = 0
    adherence_failures = []
    diversity_failures = []
    motion_failures = []
    invariant_failures = []
    unscorable = []
    negative_control_failures = []
    for rec in records:
        b = rec["baseline_metrics"]
        m = rec["modified_metrics"]
        stem = rec.get("stem", rec.get("modified", "?"))
        negative_control = bool(rec.get("negative_control", False))
        b_metric, m_metric = b.get(metric, float("nan")), m.get(metric, float("nan"))
        metric_scored = not (np.isnan(b_metric) or np.isnan(m_metric))
        win = metric_scored and m_metric > b_metric
        if win and not negative_control:
            consistency_wins += 1
        if not metric_scored:
            unscorable.append(f"{stem}:{metric}")
        entry = {
            "stem": stem,
            "theme": rec.get("theme"),
            "negative_control": negative_control,
            "metric": metric,
            "consistency_baseline": b_metric,
            "consistency_modified": m_metric,
            "subject_anchor_baseline": b.get("subject_anchor_consistency", float("nan")),
            "subject_anchor_modified": m.get("subject_anchor_consistency", float("nan")),
            "background_anchor_baseline": b.get("background_anchor_consistency", float("nan")),
            "background_anchor_modified": m.get("background_anchor_consistency", float("nan")),
            "palette_agreement_baseline": b.get("palette_agreement", float("nan")),
            "palette_agreement_modified": m.get("palette_agreement", float("nan")),
            "motion_profile_agreement_baseline": b.get("motion_profile_agreement", float("nan")),
            "motion_profile_agreement_modified": m.get("motion_profile_agreement", float("nan")),
            "consistency_win": bool(win),
        }

        b_div, m_div = b.get("inter_shot_composition_diversity"), m.get("inter_shot_composition_diversity")
        div_ok = False
        if b_div is None or m_div is None or np.isnan(b_div) or np.isnan(m_div):
            unscorable.append(f"{stem}:inter_shot_composition_diversity")
        else:
            div_ok = m_div >= b_div * (1.0 - diversity_tolerance)
            if not div_ok:
                diversity_failures.append(stem)
        entry.update(
            diversity_baseline=b_div if b_div is not None else float("nan"),
            diversity_modified=m_div if m_div is not None else float("nan"),
            diversity_ok=bool(div_ok),
        )

        b_dyn, m_dyn = b.get("dynamic_degree"), m.get("dynamic_degree")
        motion_ok = False
        if b_dyn is None or m_dyn is None or np.isnan(b_dyn) or np.isnan(m_dyn) or motion_tolerance is None:
            reason = "dynamic_degree" if motion_tolerance is not None else "motion_tolerance"
            unscorable.append(f"{stem}:{reason}")
        else:
            motion_ok = m_dyn >= b_dyn * (1.0 - motion_tolerance)
            if not motion_ok:
                motion_failures.append(stem)
        entry.update(
            dynamic_degree_baseline=b_dyn if b_dyn is not None else float("nan"),
            dynamic_degree_modified=m_dyn if m_dyn is not None else float("nan"),
            motion_tolerance_relative=motion_tolerance,
            motion_ok=bool(motion_ok),
        )

        if require_adherence:
            b_mean = b.get("prompt_adherence_mean", float("nan"))
            m_mean = m.get("prompt_adherence_mean", float("nan"))
            b_min = b.get("prompt_adherence_min", float("nan"))
            m_min = m.get("prompt_adherence_min", float("nan"))
            # A NaN (unscored) prompt cannot certify non-regression -> it fails.
            ok = (
                not (np.isnan(b_mean) or np.isnan(m_mean) or np.isnan(b_min) or np.isnan(m_min))
                and m_mean >= b_mean - adherence_tolerance
                and m_min >= b_min - adherence_tolerance
            )
            entry.update(
                adherence_baseline_mean=b_mean,
                adherence_modified_mean=m_mean,
                adherence_baseline_min=b_min,
                adherence_modified_min=m_min,
                adherence_ok=bool(ok),
            )
            if not ok:
                adherence_failures.append(stem)
        else:
            ok = True

        inv_ok = True
        if require_invariant:
            b_inv_mean = b.get("invariant_margin_mean", float("nan"))
            m_inv_mean = m.get("invariant_margin_mean", float("nan"))
            b_inv_min = b.get("invariant_margin_min", float("nan"))
            m_inv_min = m.get("invariant_margin_min", float("nan"))
            inv_ok = (
                not (np.isnan(b_inv_mean) or np.isnan(m_inv_mean) or np.isnan(b_inv_min) or np.isnan(m_inv_min))
                and m_inv_mean >= b_inv_mean - invariant_tolerance
                and m_inv_min >= b_inv_min - invariant_tolerance
            )
            entry.update(
                invariant_margin_baseline_mean=b_inv_mean,
                invariant_margin_modified_mean=m_inv_mean,
                invariant_margin_baseline_min=b_inv_min,
                invariant_margin_modified_min=m_inv_min,
                invariant_ok=bool(inv_ok),
            )
            if not inv_ok:
                invariant_failures.append(stem)

        if negative_control and win and (not require_adherence or ok):
            negative_control_failures.append(stem)
            entry["negative_control_ok"] = False
            entry["negative_control_reason"] = (
                "consistency win without adherence loss is text-override evidence, not a pass"
            )
        elif negative_control:
            entry["negative_control_ok"] = True
        per_prompt.append(entry)

    consistency_ok = consistency_wins >= min_consistency_wins
    adherence_ok = (not require_adherence) or not adherence_failures
    diversity_ok = not diversity_failures
    motion_ok = not motion_failures and not any(":motion_tolerance" in x for x in unscorable)
    invariant_ok = (not require_invariant) or not invariant_failures
    scorable_ok = not unscorable
    negative_control_ok = not negative_control_failures
    passed = bool(
        consistency_ok and adherence_ok and diversity_ok and motion_ok
        and invariant_ok and scorable_ok and negative_control_ok
    )
    blocked_reasons = []
    if unscorable:
        blocked_reasons.append(f"unscorable input(s): {sorted(set(unscorable))}")
    if negative_control_failures:
        blocked_reasons.append(
            f"negative control indicates text override: {negative_control_failures}"
        )
    blocked_reason = "; ".join(blocked_reasons) if blocked_reasons else None
    return {
        "passed": passed,
        "is_null_result": bool(not passed),
        "blocked_reason": blocked_reason,
        "metric": metric,
        "num_pairs": len(per_prompt),
        "consistency_wins": consistency_wins,
        "min_consistency_wins": min_consistency_wins,
        "consistency_ok": bool(consistency_ok),
        "require_adherence": bool(require_adherence),
        "adherence_tolerance": adherence_tolerance,
        "adherence_ok": bool(adherence_ok),
        "adherence_failures": adherence_failures,
        "diversity_tolerance_relative": diversity_tolerance,
        "diversity_ok": bool(diversity_ok),
        "diversity_failures": diversity_failures,
        "motion_tolerance_relative": motion_tolerance,
        "motion_ok": bool(motion_ok),
        "motion_failures": motion_failures,
        "require_invariant": bool(require_invariant),
        "invariant_tolerance": invariant_tolerance,
        "invariant_ok": bool(invariant_ok),
        "invariant_failures": invariant_failures,
        "scorable_ok": bool(scorable_ok),
        "unscorable": sorted(set(unscorable)),
        "negative_control_ok": bool(negative_control_ok),
        "negative_control_failures": negative_control_failures,
        "per_prompt": per_prompt,
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
