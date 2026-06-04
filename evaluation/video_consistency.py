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


def cross_perspective_evaluate_video(
    path: str | Path,
    *,
    num_shots: int | None = None,
    boundaries: list[int] | None = None,
    chunk_durations: list[int] | None = None,
    captions: list[str] | None = None,
    adherence_scorer=None,
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
    if adherence_scorer is not None and captions:
        ranges = shot_ranges(frames.shape[0], num_shots=num_shots, boundaries=boundaries)
        metrics.update(
            prompt_adherence_for_video(frames, ranges, captions, adherence_scorer)
        )
    return metrics


def compare_cross_perspective_dirs(
    baseline_dir: str | Path,
    modified_dir: str | Path,
    *,
    shots_for: "callable | int | None" = None,
    adherence_scorer=None,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, object]:
    """Baseline-vs-modified cross-perspective comparison over paired videos.

    ``shots_for`` resolves how to split each video, by stem. It may be an int
    (same shot count for all), or a callable ``stem -> spec`` where ``spec`` is
    one of: an int (shot count); a mapping with ``chunk_durations`` (exact,
    possibly uneven shot lengths) and optional ``captions``; or ``None`` (no
    match -> the pair is skipped). The delta on ``cross_shot_scene_consistency``
    is the consistency win; the companion deltas expose any viewpoint-variation
    / motion collapse, and (when ``adherence_scorer`` is given) prompt-adherence
    deltas drive the milestone non-regression guard.
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
    min_consistency_wins: int = 2,
    adherence_tolerance: float = 0.0,
    require_adherence: bool = False,
) -> dict[str, object]:
    """Decide the milestone quality gate from a comparison ``result``.

    Passes only when the modified pipeline beats the baseline on
    ``cross_shot_scene_consistency`` for at least ``min_consistency_wins``
    prompts AND (when ``require_adherence``) the modified per-shot prompt
    adherence does not drop below baseline minus ``adherence_tolerance`` on any
    evaluated prompt -- the anti-cheating guard that blocks a consistency win
    bought by ignoring the prompts.
    """
    per_prompt = []
    consistency_wins = 0
    adherence_failures = []
    for rec in result.get("records", []):
        b = rec["baseline_metrics"]
        m = rec["modified_metrics"]
        stem = rec.get("stem", rec.get("modified", "?"))
        win = m["cross_shot_scene_consistency"] > b["cross_shot_scene_consistency"]
        consistency_wins += int(win)
        entry = {
            "stem": stem,
            "consistency_baseline": b["cross_shot_scene_consistency"],
            "consistency_modified": m["cross_shot_scene_consistency"],
            "consistency_win": bool(win),
        }
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
        per_prompt.append(entry)

    consistency_ok = consistency_wins >= min_consistency_wins
    adherence_ok = (not require_adherence) or not adherence_failures
    passed = bool(consistency_ok and adherence_ok)
    return {
        "passed": passed,
        "num_pairs": len(per_prompt),
        "consistency_wins": consistency_wins,
        "min_consistency_wins": min_consistency_wins,
        "consistency_ok": bool(consistency_ok),
        "require_adherence": bool(require_adherence),
        "adherence_tolerance": adherence_tolerance,
        "adherence_ok": bool(adherence_ok),
        "adherence_failures": adherence_failures,
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
