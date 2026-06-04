# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""VBench-style cross-perspective consistency for single-scene multi-view sets.

Unlike a single concatenated multi-shot video, a scene here is a *set of
independently generated videos*, one per prompt/perspective. We measure whether
those separate videos depict the SAME place/subject, reusing VBench's backbones
and similarity definition (https://github.com/Vchitect/VBench/tree/master/vbench):

  * subject consistency  -- DINO (``facebook/dino-vits16``) per-frame features,
    cosine similarity (``vbench/subject_consistency.py``);
  * background consistency-- CLIP (``ViT-B/32``) image features, cosine
    (``vbench/background_consistency.py``);

lifted from VBench's adjacent-frame granularity to *cross-video* granularity:
each video is reduced to one mean (L2-normalized) embedding, then we take the
mean pairwise cosine across the perspective videos (and each vs. perspective 0).

Anti-cheating companions so "every perspective is an identical / frozen clip"
cannot score a fake win:
  * dynamic_degree   -- optical-flow magnitude per video (VBench RAFT proxy via
    Farneback; no extra weights);
  * inter_video_diversity -- 1 - mean pairwise cosine of a framing-sensitive
    layout signature, so near-duplicate perspectives are visible.

Prompt adherence (CLIP image-text) is the per-perspective guard. The DINO / CLIP
backbones are milestone-only (GPU + a one-time download); the cross-video
aggregation math is pure NumPy and unit-tested on CPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from evaluation.video_consistency import (
    composition_signature,
    cosine_similarity,
    discover_videos,
    read_video,
)

# Per-perspective output stem: "<prefix>-rank<R>-<scene>-p<P>-seed<S>_<model>"
# (scene tokens are filename-safe, no hyphens). Captures scene + perspective.
_PERSPECTIVE_STEM = re.compile(
    r"^(?:.*?-)?rank\d+-(?P<scene>.+)-p(?P<persp>\d+)-seed\d+_[^-]*$"
)


# ---------------------------------------------------------------------------
# Pure cross-video aggregation (GPU-free, unit-tested)
# ---------------------------------------------------------------------------


def video_mean_embedding(per_frame_feats: np.ndarray) -> np.ndarray:
    """Reduce ``[T, D]`` per-frame features to one L2-normalized video embedding.

    Each frame feature is L2-normalized (so no single high-norm frame dominates),
    averaged over time, then renormalized -- matching how VBench pools a clip.
    """
    feats = np.asarray(per_frame_feats, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[0] == 0:
        raise ValueError(f"expected [T,D] features, got {feats.shape}")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.clip(norms, 1e-8, None)
    mean = feats.mean(axis=0)
    n = np.linalg.norm(mean)
    return mean / n if n > 0 else mean


def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine over all unordered pairs of rows ``[N, D]`` (N>=2)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    sims = [cosine_similarity(emb[i], emb[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))


def mean_to_first_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine of every perspective vs. perspective 0 (VBench's to-first term)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.shape[0] < 2:
        return float("nan")
    return float(np.mean([cosine_similarity(emb[0], emb[i]) for i in range(1, emb.shape[0])]))


def cross_video_consistency(video_embeddings: np.ndarray) -> dict[str, float]:
    """Cross-video consistency from per-video embeddings ``[N_videos, D]``.

    VBench averages a to-first term and an adjacent term; across independent
    perspectives there is no temporal order, so we report the symmetric mean
    pairwise cosine and the mean-to-first, plus VBench's 0.5*(pairwise+to_first).
    """
    emb = np.asarray(video_embeddings, dtype=np.float64)
    pairwise = mean_pairwise_cosine(emb)
    to_first = mean_to_first_cosine(emb)
    score = float(np.nanmean([pairwise, to_first]))
    return {"pairwise": pairwise, "to_first": to_first, "score": score,
            "num_videos": float(emb.shape[0])}


def optical_flow_dynamics(frames: np.ndarray, *, stride: int = 1, sample: int = 12) -> float:
    """Mean optical-flow magnitude over a video (VBench dynamic-degree proxy).

    Uses Farneback flow (no RAFT weights) on up to ``sample`` evenly spaced frame
    pairs; ~0 means a frozen/near-static clip, exposing a degenerate copy.
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


def layout_signature_video(frames: np.ndarray, *, sample: int = 8) -> np.ndarray:
    """One framing-sensitive layout embedding per video (for diversity check)."""
    idx = np.unique(np.linspace(0, frames.shape[0] - 1, num=min(sample, frames.shape[0])).round().astype(int))
    feats = np.stack([composition_signature(frames[i]) for i in idx], axis=0)
    return video_mean_embedding(feats)


# ---------------------------------------------------------------------------
# Milestone-only backbones (DINO / CLIP); lazy, clear errors, never on CPU tests
# ---------------------------------------------------------------------------


def build_dino_encoder(*, device: str | None = None):
    """Return ``fn(frames_rgb) -> [T, D]`` DINO features (VBench subject backbone).

    Tries torch.hub ``facebookresearch/dino:dino_vits16`` then HF
    ``facebook/dino-vits16``; raises a clear error if neither is available.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    import torchvision.transforms as T

    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = T.Compose([T.ToTensor(), T.Resize(224), T.CenterCrop(224), norm])

    try:
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(dev).eval()

        def _embed(frames_rgb):
            from PIL import Image
            batch = torch.stack([resize(Image.fromarray(f.astype(np.uint8))) for f in frames_rgb]).to(dev)
            with torch.no_grad():
                feats = model(batch)
            return feats.float().cpu().numpy()

        return _embed
    except Exception:
        pass

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "VBench subject consistency needs a DINO backbone (torch.hub "
            "facebookresearch/dino or transformers facebook/dino-vits16)."
        ) from exc
    try:
        model = AutoModel.from_pretrained("facebook/dino-vits16").to(dev).eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not load facebook/dino-vits16 for VBench subject consistency; "
            "ensure it is downloaded/cached."
        ) from exc

    def _embed_hf(frames_rgb):
        from PIL import Image
        batch = torch.stack([resize(Image.fromarray(f.astype(np.uint8))) for f in frames_rgb]).to(dev)
        with torch.no_grad():
            out = model(pixel_values=batch)
        return out.last_hidden_state[:, 0].float().cpu().numpy()  # CLS token

    return _embed_hf


def build_clip_image_encoder(*, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str | None = None):
    """Return ``fn(frames_rgb) -> [T, D]`` CLIP image features (VBench background backbone)."""
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(dev).eval()

        def _embed(frames_rgb):
            from PIL import Image
            batch = torch.stack([preprocess(Image.fromarray(f.astype(np.uint8))) for f in frames_rgb]).to(dev)
            with torch.no_grad():
                feats = model.encode_image(batch)
            return feats.float().cpu().numpy()

        return _embed
    except Exception:
        pass

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("VBench background consistency needs open_clip or transformers CLIP.") from exc
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _embed_hf(frames_rgb):
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames_rgb]
        inputs = processor(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        return feats.float().cpu().numpy()

    return _embed_hf


# ---------------------------------------------------------------------------
# Scene scoring over a set of per-perspective videos
# ---------------------------------------------------------------------------


def _sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    if frames.shape[0] <= n:
        return frames
    idx = np.unique(np.linspace(0, frames.shape[0] - 1, num=n).round().astype(int))
    return frames[idx]


def group_perspectives_by_scene(directory: str | Path) -> dict[str, list[Path]]:
    """Group per-perspective video files in a dir by scene, ordered by perspective.

    Matches ``<prefix>-rank<R>-<scene>-p<P>-seed<S>_<model>``; non-matching files
    are ignored. Returns ``{scene: [p0, p1, ...]}``.
    """
    scenes: dict[str, list[tuple[int, Path]]] = {}
    for path in discover_videos(directory):
        m = _PERSPECTIVE_STEM.match(path.stem)
        if not m:
            continue
        scenes.setdefault(m.group("scene"), []).append((int(m.group("persp")), path))
    return {s: [p for _, p in sorted(v)] for s, v in scenes.items()}


def score_scene(
    perspective_videos: list[str | Path],
    *,
    dino_encoder=None,
    clip_encoder=None,
    adherence_scorer=None,
    captions: list[str] | None = None,
    frames_per_video: int = 16,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, float]:
    """VBench-style cross-perspective scores for one scene's perspective videos.

    Returns subject (DINO) and background (CLIP) cross-video consistency, the
    dynamic-degree / inter-video-diversity anti-cheats, and (optional) per-shot
    prompt adherence. Decodes each video once; the heavy encoders are optional so
    a partial (e.g. dynamics-only) score still works without the weights.
    """
    paths = [Path(p) for p in perspective_videos]
    if len(paths) < 2:
        raise ValueError("cross-perspective scoring needs >= 2 perspective videos")

    dino_feats, clip_feats, layout_feats = [], [], []
    dynamics, adherence = [], []
    for i, path in enumerate(paths):
        frames = read_video(path, max_frames=max_frames, stride=stride)
        sub = _sample_frames(frames, frames_per_video)
        if dino_encoder is not None:
            dino_feats.append(video_mean_embedding(dino_encoder(sub)))
        if clip_encoder is not None:
            clip_feats.append(video_mean_embedding(clip_encoder(sub)))
        layout_feats.append(layout_signature_video(frames))
        dynamics.append(optical_flow_dynamics(frames, stride=stride))
        if adherence_scorer is not None and captions and i < len(captions) and captions[i]:
            adherence.append(float(adherence_scorer(sub, captions[i])))

    out: dict[str, float] = {"num_videos": float(len(paths))}
    if dino_feats:
        s = cross_video_consistency(np.stack(dino_feats))
        out["subject_consistency"] = s["score"]
        out["subject_consistency_pairwise"] = s["pairwise"]
        out["subject_consistency_to_first"] = s["to_first"]
    if clip_feats:
        s = cross_video_consistency(np.stack(clip_feats))
        out["background_consistency"] = s["score"]
        out["background_consistency_pairwise"] = s["pairwise"]
    out["inter_video_diversity"] = 1.0 - mean_pairwise_cosine(np.stack(layout_feats))
    out["dynamic_degree"] = float(np.mean(dynamics))
    out["dynamic_degree_min"] = float(np.min(dynamics))
    if adherence:
        out["prompt_adherence_mean"] = float(np.mean(adherence))
        out["prompt_adherence_min"] = float(np.min(adherence))
    return out


def compare_multiview_vbench_dirs(
    baseline_dir: str | Path,
    modified_dir: str | Path,
    *,
    dino_encoder=None,
    clip_encoder=None,
    adherence_scorer=None,
    captions_for=None,
    frames_per_video: int = 16,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, object]:
    """Baseline-vs-modified VBench-style cross-perspective comparison.

    Each directory holds one video per perspective; videos are grouped by scene
    (``group_perspectives_by_scene``) and each scene with >= 2 perspectives is
    scored on both sides with ``score_scene``. ``captions_for`` is an optional
    ``scene -> [captions]`` map for the adherence guard. Returns per-scene
    records with subject/background consistency, dynamics, diversity, adherence
    and modified-minus-baseline deltas.
    """
    base = group_perspectives_by_scene(baseline_dir)
    mod = group_perspectives_by_scene(modified_dir)
    scenes = sorted(set(base) & set(mod))
    if not scenes:
        raise ValueError(
            f"No common multi-perspective scenes in {baseline_dir} and {modified_dir}"
        )

    records = []
    for scene in scenes:
        if len(base[scene]) < 2 or len(mod[scene]) < 2:
            continue
        caps = captions_for.get(scene) if captions_for else None
        kw = dict(
            dino_encoder=dino_encoder, clip_encoder=clip_encoder,
            adherence_scorer=adherence_scorer, captions=caps,
            frames_per_video=frames_per_video, max_frames=max_frames, stride=stride,
        )
        b = score_scene(base[scene], **kw)
        m = score_scene(mod[scene], **kw)
        delta = {k: m[k] - b[k] for k in b if k in m and k != "num_videos"}
        records.append({"scene": scene, "baseline_metrics": b,
                        "modified_metrics": m, "delta": delta})
    if not records:
        raise ValueError("No scene had >= 2 perspectives on both sides.")
    return {"num_scenes": len(records), "records": records}
