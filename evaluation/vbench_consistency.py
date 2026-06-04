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
  * subject-IDENTITY consistency -- a DISCRIMINATIVE instance embedding (ArcFace
    for human subjects, DINO patch-tokens for non-human), so the SAME subject
    across videos scores above a merely similar-looking one (identity != mere
    similarity); the subject kind is selected per scene;
  * adapted VBench dims -- ``temporal_style`` + ``appearance_style`` cross-video
    consistency (GPU-free motion/palette signatures), ``overall_consistency``
    (CLIP text-video), and ``motion_smoothness`` (GPU-free AMT proxy).

lifted from VBench's adjacent-frame granularity to *cross-video* granularity:
each video is reduced to one mean (L2-normalized) embedding, then we take the
mean pairwise cosine across the perspective videos (and each vs. perspective 0).
``aggregate_consistency`` is the mean of the available consistency dims.

Anti-cheating companions so "every perspective is an identical / frozen clip"
cannot score a fake win:
  * dynamic_degree   -- optical-flow magnitude per video (VBench RAFT proxy via
    Farneback; no extra weights);
  * inter_video_diversity -- 1 - mean pairwise cosine of a framing-sensitive
    layout signature, so near-duplicate perspectives are visible.

Prompt adherence (CLIP image-text) is the per-perspective guard. The DINO / CLIP
/ ArcFace backbones are milestone-only (GPU + a one-time download); the
cross-video aggregation math is pure NumPy and unit-tested on CPU.
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
# (scene tokens are filename-safe, no hyphens). Captures scene + perspective + seed.
_PERSPECTIVE_STEM = re.compile(
    r"^(?:.*?-)?rank\d+-(?P<scene>.+)-p(?P<persp>\d+)-seed(?P<seed>\d+)_[^-]*$"
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


def identity_consistency(identity_embeddings: np.ndarray) -> dict[str, float]:
    """Cross-video subject-IDENTITY consistency from per-video identity embeddings.

    The aggregation is the same symmetric cross-video cosine as
    ``cross_video_consistency``; what makes this *identity* (the SAME subject
    across videos) rather than mere *similarity* (a similar-looking subject) is
    the ENCODER -- a discriminative face/instance embedding (ArcFace for human
    subjects, DINO patch features for non-human ones) whose cosine separates two
    different individuals even when their global appearance is close. The
    non-vacuity of that separation is asserted on synthetic embeddings in the
    unit tests (same-subject scores strictly higher than different-subject AT
    EQUAL global-feature distance).
    """
    out = cross_video_consistency(identity_embeddings)
    return {"identity": out["score"], "identity_pairwise": out["pairwise"],
            "identity_to_first": out["to_first"], "num_videos": out["num_videos"]}


#: Cross-video consistency dimensions that feed the AC-4 aggregate. Each is a
#: "are the separate perspective videos the same scene" score in roughly [-1, 1];
#: quality/anti-cheat companions (dynamics, diversity, motion_smoothness,
#: adherence) are reported separately and are NOT averaged into the aggregate.
CONSISTENCY_DIMS = (
    "subject_consistency",
    "background_consistency",
    "subject_identity_consistency",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
)


def aggregate_consistency(metrics: dict[str, float]) -> float:
    """Mean of the available cross-video consistency dimensions (AC-4 aggregate).

    Averages only the ``CONSISTENCY_DIMS`` actually present (a backbone-gated dim
    is simply absent when its encoder was not loaded), so the aggregate is
    comparable as long as baseline and modified are scored with the same encoders.
    Returns NaN when no consistency dim is available.
    """
    vals = [float(metrics[k]) for k in CONSISTENCY_DIMS
            if k in metrics and not np.isnan(metrics[k])]
    return float(np.mean(vals)) if vals else float("nan")


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
# GPU-free per-video VBench-adapted signatures (task2)
# ---------------------------------------------------------------------------
#
# VBench's temporal_style / appearance_style / motion_smoothness use heavy
# backbones (CLIP-temporal, AMT interpolation). For a cross-VIDEO consistency
# read we only need a per-video signature whose *agreement across perspectives*
# is meaningful; these GPU-free signatures give that without weights, so the
# aggregation stays CPU unit-testable. ``overall_consistency`` (CLIP text-video)
# is the one dim that genuinely needs a backbone and is milestone-only.


def motion_smoothness(frames: np.ndarray, *, sample: int = 16) -> float:
    """Per-video motion smoothness in [0, 1] (VBench AMT proxy, GPU-free).

    Smooth motion has a roughly CONSTANT speed over time; jitter/teleport-y
    motion has a wildly varying speed. We take the per-frame-pair mean luminance
    change as a scalar "speed", and define smoothness as ``1 - std(speed) /
    mean(speed)`` (one minus the coefficient of variation of speed), clipped to
    [0, 1]. A static clip (no motion) and a steady pan both read as smooth
    (-> ~1.0); erratic motion drops toward 0.
    """
    if frames.ndim != 4 or frames.shape[0] < 3:
        return 1.0  # too short to be non-smooth
    idx = np.unique(np.linspace(0, frames.shape[0] - 1, num=min(sample, frames.shape[0])).round().astype(int))
    grays = np.stack([
        cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), (32, 32),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for i in idx
    ], axis=0)
    if grays.shape[0] < 3:
        return 1.0
    speed = np.abs(np.diff(grays, axis=0)).mean(axis=(1, 2))  # scalar speed per pair
    mean_speed = float(speed.mean())
    if mean_speed < 1e-4:
        return 1.0  # no motion -> trivially smooth
    cv = float(speed.std() / (mean_speed + 1e-6))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def temporal_dynamics_signature(frames: np.ndarray, *, bins: int = 8, sample: int = 16) -> np.ndarray:
    """GPU-free per-video temporal-style signature (motion-magnitude profile).

    A normalized histogram of per-pixel temporal-gradient magnitude over sampled
    adjacent frame pairs, plus its mean/std. Two perspectives of one scene shot
    with a similar motion character (pace, amount of movement) produce similar
    signatures, so their cross-video cosine is high -- the cross-video reading of
    VBench's temporal_style.
    """
    if frames.ndim != 4 or frames.shape[0] < 2:
        return np.zeros(bins + 2, dtype=np.float64)
    idx = np.unique(np.linspace(0, frames.shape[0] - 1, num=min(sample, frames.shape[0])).round().astype(int))
    grays = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), (32, 32),
                        interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for i in idx]
    mags = np.abs(np.diff(np.stack(grays, axis=0), axis=0)).reshape(-1)
    hist, _ = np.histogram(mags, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1e-8)
    sig = np.concatenate([hist, [float(mags.mean()), float(mags.std())]])
    return sig


def appearance_style_signature(frame_rgb: np.ndarray) -> np.ndarray:
    """GPU-free per-frame appearance-style signature (palette + texture energy).

    A framing-ROBUST "look" descriptor: an HSV color histogram (palette) plus an
    edge-magnitude histogram (texture energy). Both are global distributions, so
    a camera move that rearranges the frame barely shifts them -- two
    perspectives of the same place share a palette/texture and score consistent.
    Distinct from CLIP ``background_consistency`` (semantic) and from
    ``composition_signature`` (framing-SENSITIVE layout).
    """
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    color = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).astype(np.float64).reshape(-1)
    color /= max(color.sum(), 1e-8)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(-1)
    texture, _ = np.histogram(mag, bins=8, range=(0.0, 4.0))
    texture = texture.astype(np.float64)
    texture /= max(texture.sum(), 1e-8)
    return np.concatenate([color, texture])


def temporal_style_video(frames: np.ndarray) -> np.ndarray:
    """One L2-normalized temporal-style embedding per video."""
    return video_mean_embedding(temporal_dynamics_signature(frames)[None, :])


def appearance_style_video(frames: np.ndarray, *, sample: int = 8) -> np.ndarray:
    """One L2-normalized appearance-style embedding per video."""
    idx = np.unique(np.linspace(0, frames.shape[0] - 1, num=min(sample, frames.shape[0])).round().astype(int))
    feats = np.stack([appearance_style_signature(frames[i]) for i in idx], axis=0)
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


def build_identity_encoder(*, subject_kind: str = "auto", device: str | None = None):
    """Return ``fn(frames_rgb) -> [T, D]`` subject-IDENTITY features.

    Identity is the SAME individual across videos, not a similar-looking one, so
    the backbone is discriminative, not a generic appearance encoder:

      * human subjects -> ArcFace face embedding (``insightface`` buffalo_l), one
        embedding per frame from the largest detected face; frames with no face
        contribute no embedding (the per-video mean uses the rest).
      * non-human subjects -> DINO patch-token features (mean over patch tokens,
        not just CLS) -- finer-grained instance identity than the CLS appearance
        vector used for ``subject_consistency``.

    ``subject_kind`` is "human", "object", or "auto" (try a face first, fall back
    to DINO patches per frame). Milestone-only: needs weights + a one-time
    download; never imported on the per-round CPU tests. Raises a clear error if
    the requested backbone is unavailable.
    """
    import numpy as _np

    kind = (subject_kind or "auto").lower()
    face_embed = None
    if kind in ("human", "auto"):
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]
                               if (device or "").startswith("cpu") else None)
            app.prepare(ctx_id=0 if not (device or "").startswith("cpu") else -1, det_size=(320, 320))

            def face_embed(frame_rgb):  # noqa: ANN001
                faces = app.get(frame_rgb[..., ::-1])  # insightface expects BGR
                if not faces:
                    return None
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                return _np.asarray(face.normed_embedding, dtype=_np.float64)
        except Exception as exc:
            if kind == "human":
                raise RuntimeError(
                    "subject_kind='human' identity needs the `insightface` ArcFace "
                    "backbone (buffalo_l); install it or use subject_kind='object'."
                ) from exc
            face_embed = None  # auto -> fall back to DINO patches

    # DINO patch fallback. Build it EAGERLY only when it is the sole identity path
    # ("object", or "auto" with no face backbone). For "auto" WITH a working face
    # backbone, build it LAZILY (only when a frame has no detectable face), so an
    # environment with working ArcFace but no DINO can still score faces.
    _dino_state: dict = {}

    def _dino_patch():
        if "enc" not in _dino_state:
            _dino_state["enc"] = _build_dino_patch_encoder(device=device)
        return _dino_state["enc"]

    if kind == "object" or (kind == "auto" and face_embed is None):
        _dino_patch()  # fail fast: DINO is the only available identity path here
    allow_dino = kind != "human"  # 'human' is ArcFace-only; no-face frames are skipped

    def _encode(frames_rgb):
        embs = []
        for f in frames_rgb:
            emb = face_embed(f) if face_embed is not None else None
            if emb is None and allow_dino:
                emb = _dino_patch()(f[None, ...])[0]  # lazy build on first no-face frame
            if emb is not None:
                embs.append(_np.asarray(emb, dtype=_np.float64))
        if not embs:
            raise RuntimeError(
                "Identity encoder produced no embeddings (no face detected and no "
                "DINO-patch fallback available) for this video."
            )
        return _np.stack(embs, axis=0)

    return _encode


def _build_dino_patch_encoder(*, device: str | None = None):
    """Return ``fn(frames_rgb) -> [T, D]`` DINO PATCH-token features (mean over
    patches), a finer instance-identity signal than the CLS appearance vector.

    Prefers the HuggingFace ``facebook/dino-vits16`` (its ``last_hidden_state``
    patch tokens), because the torch.hub DINO repo does ``from utils import
    trunc_normal_`` which collides with this repository's own ``utils`` package
    once it is in ``sys.modules`` (the hub import silently picks the wrong
    ``utils``). torch.hub is only the fallback.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    import torchvision.transforms as T

    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = T.Compose([T.ToTensor(), T.Resize(224), T.CenterCrop(224), norm])

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("facebook/dino-vits16").to(dev).eval()

        def _embed_hf(frames_rgb):
            from PIL import Image
            batch = torch.stack([resize(Image.fromarray(f.astype(np.uint8))) for f in frames_rgb]).to(dev)
            with torch.no_grad():
                out = model(pixel_values=batch)
                patches = out.last_hidden_state[:, 1:, :].mean(dim=1)  # mean over patch tokens
            return patches.float().cpu().numpy()

        return _embed_hf
    except Exception:
        pass

    try:
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(dev).eval()
    except Exception as exc:
        raise RuntimeError(
            "Non-human identity needs a DINO backbone (transformers "
            "facebook/dino-vits16 or torch.hub facebookresearch/dino) for "
            "patch-token features."
        ) from exc

    def _embed(frames_rgb):
        from PIL import Image
        batch = torch.stack([resize(Image.fromarray(f.astype(np.uint8))) for f in frames_rgb]).to(dev)
        with torch.no_grad():
            tokens = model.get_intermediate_layers(batch, n=1)[0]  # [B, 1+N, D]
            patches = tokens[:, 1:, :].mean(dim=1)  # mean over patch tokens
        return patches.float().cpu().numpy()

    return _embed


def select_subject_kind(scene: str, *, scene_subject_kinds: dict | None = None,
                        default: str = "auto") -> str:
    """Per-scene subject-kind selector for the identity backbone.

    Looks ``scene`` up in an optional ``scene -> {human|object|auto}`` map and
    falls back to ``default`` ("auto" tries a face then DINO patches). Kept as an
    explicit hook so a scene's subject type can be pinned without code changes.
    """
    if scene_subject_kinds and scene in scene_subject_kinds:
        kind = str(scene_subject_kinds[scene]).lower()
        if kind not in ("human", "object", "auto"):
            raise ValueError(f"Unknown subject_kind {kind!r} for scene {scene!r}")
        return kind
    return default


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

    Cross-VIEW consistency is scored within a SINGLE seed, so a scene must not
    carry the same perspective index from more than one seed (e.g. ``num_samples
    > 1`` or mixed-seed outputs). Such a directory is rejected with a clear error
    rather than silently scoring seed-to-seed instead of cross-view consistency.
    """
    scenes: dict[str, dict[int, tuple[int, Path]]] = {}
    for path in discover_videos(directory):
        m = _PERSPECTIVE_STEM.match(path.stem)
        if not m:
            continue
        scene, persp, seed = m.group("scene"), int(m.group("persp")), int(m.group("seed"))
        bucket = scenes.setdefault(scene, {})
        if persp in bucket:
            prev_seed = bucket[persp][0]
            raise ValueError(
                f"Scene {scene!r} has perspective p{persp} from multiple seeds "
                f"(seed{prev_seed} and seed{seed}); cross-view consistency is scored "
                "within ONE seed -- render num_samples=1 or evaluate a single seed's "
                "outputs per directory."
            )
        bucket[persp] = (seed, path)
    return {s: [path for _persp, (_seed, path) in sorted(b.items())] for s, b in scenes.items()}


def score_scene(
    perspective_videos: list[str | Path],
    *,
    dino_encoder=None,
    clip_encoder=None,
    identity_encoder=None,
    adherence_scorer=None,
    captions: list[str] | None = None,
    frames_per_video: int = 16,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, float]:
    """VBench-style cross-perspective scores for one scene's perspective videos.

    Reports, across the scene's separately-generated perspective videos:

      * ``subject_consistency`` (DINO CLS), ``background_consistency`` (CLIP),
        ``subject_identity_consistency`` (ArcFace/DINO-patch) -- backbone-gated;
      * GPU-free ``temporal_style`` + ``appearance_style`` cross-video
        consistency, ``motion_smoothness`` (mean + min);
      * ``overall_consistency`` (CLIP text-video, == adherence mean) when a scorer
        + captions are given;
      * the anti-cheat companions ``inter_video_diversity`` / ``dynamic_degree``
        and the ``prompt_adherence`` guard;
      * ``aggregate_consistency`` -- the mean of the available consistency dims.

    Decodes each video once; the heavy encoders are optional so a partial (e.g.
    GPU-free-only) score still works without any weights.
    """
    paths = [Path(p) for p in perspective_videos]
    if len(paths) < 2:
        raise ValueError("cross-perspective scoring needs >= 2 perspective videos")

    dino_feats, clip_feats, identity_feats, layout_feats = [], [], [], []
    temporal_feats, appearance_feats = [], []
    dynamics, smoothness, adherence = [], [], []
    for i, path in enumerate(paths):
        frames = read_video(path, max_frames=max_frames, stride=stride)
        sub = _sample_frames(frames, frames_per_video)
        if dino_encoder is not None:
            dino_feats.append(video_mean_embedding(dino_encoder(sub)))
        if clip_encoder is not None:
            clip_feats.append(video_mean_embedding(clip_encoder(sub)))
        if identity_encoder is not None:
            identity_feats.append(video_mean_embedding(identity_encoder(sub)))
        layout_feats.append(layout_signature_video(frames))
        temporal_feats.append(temporal_style_video(frames))
        appearance_feats.append(appearance_style_video(frames))
        dynamics.append(optical_flow_dynamics(frames, stride=stride))
        smoothness.append(motion_smoothness(frames))
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
    if identity_feats:
        s = identity_consistency(np.stack(identity_feats))
        out["subject_identity_consistency"] = s["identity"]
        out["subject_identity_consistency_pairwise"] = s["identity_pairwise"]
    # GPU-free adapted VBench dims (always available).
    out["temporal_style"] = cross_video_consistency(np.stack(temporal_feats))["score"]
    out["appearance_style"] = cross_video_consistency(np.stack(appearance_feats))["score"]
    out["motion_smoothness"] = float(np.mean(smoothness))
    out["motion_smoothness_min"] = float(np.min(smoothness))
    # Anti-cheat companions.
    out["inter_video_diversity"] = 1.0 - mean_pairwise_cosine(np.stack(layout_feats))
    out["dynamic_degree"] = float(np.mean(dynamics))
    out["dynamic_degree_min"] = float(np.min(dynamics))
    if adherence:
        out["prompt_adherence_mean"] = float(np.mean(adherence))
        out["prompt_adherence_min"] = float(np.min(adherence))
        # overall_consistency (VBench CLIP text-video dim) shares the text-video
        # signal; it feeds the consistency aggregate while adherence_min stays the
        # gate guard.
        out["overall_consistency"] = out["prompt_adherence_mean"]
    out["aggregate_consistency"] = aggregate_consistency(out)
    return out


def compare_multiview_vbench_dirs(
    baseline_dir: str | Path,
    modified_dir: str | Path,
    *,
    dino_encoder=None,
    clip_encoder=None,
    identity_encoder=None,
    identity_encoder_for=None,
    scene_subject_kinds: dict | None = None,
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
    ``scene -> [captions]`` map for the adherence guard.

    Subject IDENTITY is per-scene: pass either a single ``identity_encoder`` for
    every scene, or ``identity_encoder_for`` -- a factory ``fn(subject_kind) ->
    encoder`` built once per distinct kind, with ``scene_subject_kinds`` (a
    ``scene -> human|object|auto`` map) selecting the kind via
    ``select_subject_kind``. Returns per-scene records with every dimension,
    the ``aggregate_consistency``, and modified-minus-baseline deltas.
    """
    base = group_perspectives_by_scene(baseline_dir)
    mod = group_perspectives_by_scene(modified_dir)
    scenes = sorted(set(base) & set(mod))
    if not scenes:
        raise ValueError(
            f"No common multi-perspective scenes in {baseline_dir} and {modified_dir}"
        )

    encoder_cache: dict[str, object] = {}

    def _identity_for(scene: str):
        if identity_encoder is not None:
            return identity_encoder
        if identity_encoder_for is None:
            return None
        kind = select_subject_kind(scene, scene_subject_kinds=scene_subject_kinds)
        if kind not in encoder_cache:
            encoder_cache[kind] = identity_encoder_for(kind)
        return encoder_cache[kind]

    records = []
    for scene in scenes:
        if len(base[scene]) < 2 or len(mod[scene]) < 2:
            continue
        caps = captions_for.get(scene) if captions_for else None
        kw = dict(
            dino_encoder=dino_encoder, clip_encoder=clip_encoder,
            identity_encoder=_identity_for(scene),
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


def evaluate_multiview_vbench_gate(
    result: dict,
    *,
    metric: str = "aggregate_consistency",
    min_scene_wins: int | None = None,
    adherence_tolerance: float = 0.0,
    diversity_tolerance: float = 0.0,
    motion_tolerance: float | None = None,
    require_adherence: bool = True,
) -> dict:
    """Decide the AC-4 cross-video milestone gate from a comparison ``result``.

    Passes only when the modified config raises the cross-video consistency
    ``metric`` (default ``aggregate_consistency``) over baseline on at least
    ``min_scene_wins`` scenes (default ``ceil(N/2)``) WITHOUT (a) collapsing
    inter-video diversity below baseline minus ``diversity_tolerance``, (b) --
    when ``motion_tolerance`` is set -- dropping ``dynamic_degree`` below baseline
    minus ``motion_tolerance`` on any scene (a motion-collapse guard; OFF by
    default so a known motion-reduction is reported, not silently failed), or
    (c) -- when ``require_adherence`` -- regressing per-perspective prompt
    adherence below baseline minus ``adherence_tolerance`` on any scene. A
    consistency gain bought by a diversity/motion collapse or an adherence
    regression does NOT count (AC-4 anti-cheat). When it does not pass, the
    returned record IS the honest null/partial result -- the caller commits it.
    """
    records = result.get("records", [])
    n = len(records)
    if min_scene_wins is None:
        min_scene_wins = -(-n // 2)  # ceil(n/2)

    per_scene = []
    wins = 0
    adherence_failures: list[str] = []
    diversity_failures: list[str] = []
    motion_failures: list[str] = []
    for rec in records:
        b, m = rec["baseline_metrics"], rec["modified_metrics"]
        scene = rec.get("scene", "?")
        b_agg, m_agg = b.get(metric, float("nan")), m.get(metric, float("nan"))
        win = not (np.isnan(b_agg) or np.isnan(m_agg)) and m_agg > b_agg
        wins += int(win)
        entry = {
            "scene": scene,
            "metric": metric,
            "baseline": b_agg,
            "modified": m_agg,
            "delta": (m_agg - b_agg) if not (np.isnan(b_agg) or np.isnan(m_agg)) else float("nan"),
            "win": bool(win),
        }

        b_div, m_div = b.get("inter_video_diversity"), m.get("inter_video_diversity")
        if b_div is not None and m_div is not None:
            div_ok = m_div >= b_div - diversity_tolerance
            entry["diversity_baseline"] = b_div
            entry["diversity_modified"] = m_div
            entry["diversity_ok"] = bool(div_ok)
            if not div_ok:
                diversity_failures.append(scene)

        b_dyn, m_dyn = b.get("dynamic_degree"), m.get("dynamic_degree")
        if b_dyn is not None and m_dyn is not None:
            entry["dynamic_degree_baseline"] = b_dyn
            entry["dynamic_degree_modified"] = m_dyn
            if motion_tolerance is not None:
                motion_ok = m_dyn >= b_dyn - motion_tolerance
                entry["motion_ok"] = bool(motion_ok)
                if not motion_ok:
                    motion_failures.append(scene)

        if require_adherence:
            b_mean, m_mean = b.get("prompt_adherence_mean", float("nan")), m.get("prompt_adherence_mean", float("nan"))
            b_min, m_min = b.get("prompt_adherence_min", float("nan")), m.get("prompt_adherence_min", float("nan"))
            adh_ok = (
                not (np.isnan(b_mean) or np.isnan(m_mean) or np.isnan(b_min) or np.isnan(m_min))
                and m_mean >= b_mean - adherence_tolerance
                and m_min >= b_min - adherence_tolerance
            )
            entry.update(adherence_baseline_mean=b_mean, adherence_modified_mean=m_mean,
                         adherence_baseline_min=b_min, adherence_modified_min=m_min,
                         adherence_ok=bool(adh_ok))
            if not adh_ok:
                adherence_failures.append(scene)
        per_scene.append(entry)

    consistency_ok = wins >= min_scene_wins
    adherence_ok = (not require_adherence) or not adherence_failures
    diversity_ok = not diversity_failures
    motion_ok = (motion_tolerance is None) or not motion_failures
    passed = bool(consistency_ok and adherence_ok and diversity_ok and motion_ok)
    return {
        "passed": passed,
        "metric": metric,
        "num_scenes": n,
        "scene_wins": wins,
        "min_scene_wins": min_scene_wins,
        "consistency_ok": bool(consistency_ok),
        "require_adherence": bool(require_adherence),
        "adherence_tolerance": adherence_tolerance,
        "adherence_ok": bool(adherence_ok),
        "adherence_failures": adherence_failures,
        "diversity_tolerance": diversity_tolerance,
        "diversity_ok": bool(diversity_ok),
        "diversity_failures": diversity_failures,
        "motion_tolerance": motion_tolerance,
        "motion_ok": bool(motion_ok),
        "motion_failures": motion_failures,
        "is_null_result": bool(not passed),
        "per_scene": per_scene,
    }
