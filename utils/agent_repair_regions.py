"""Parent-aligned region alternatives and auditable visual previews."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from utils.agent_edit import read_video_frames
from utils.agent_runtime import file_sha256


def mask_content_sha256(masks: Mapping[int, torch.Tensor]) -> str:
    """Hash geometry, including frame/chunk indices, rather than file metadata."""
    digest = hashlib.sha256()
    for chunk, mask in sorted(masks.items()):
        value = mask.detach().float().cpu().contiguous()
        digest.update(json.dumps([int(chunk), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def attribute_region_variants(base, part, person, protected):
    """Three parent-coordinate choices, with the same protected pixels excluded."""
    allowed = person * (protected < 0.35).float()
    safe_base = torch.minimum(base, allowed)
    expanded = torch.nn.functional.max_pool2d((base > 0)[:, None].float(), 3, 1, 1)[:, 0]
    return {
        "detected": safe_base,
        "expanded": torch.maximum(safe_base, expanded * allowed),
        "sensitive": torch.maximum(safe_base, (part >= 0.20).float() * allowed),
    }


def region_preview(option, *, video: str, output_dir: Path):
    """Show exact mask frame indices on their source video, plus enlarged details.

    This is a diagnostic plot of stored arrays, not a new segmentation. The
    renderer receives the same mask artifact that is visualised here.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    mask_path = option["action"]["params"]["mask_path"]
    if file_sha256(mask_path) != option["mask_sha256"]:
        raise RuntimeError("offered parent mask changed before preview")
    payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    if Path(payload["video"]).resolve() != Path(video).resolve():
        raise ValueError("mask preview must use the same parent video as its coordinates")
    masks = payload["masks"]
    frames_per_chunk = int(payload["frames_per_chunk"])
    protected = {}
    if option.get("protected_mask_path"):
        if file_sha256(option["protected_mask_path"]) != option["protected_mask_sha256"]:
            raise RuntimeError("protected mask changed before preview")
        protected = torch.load(option["protected_mask_path"], map_location="cpu", weights_only=False)["masks"]
    for chunk, mask in masks.items():
        if mask.ndim != 3 or len(mask) != frames_per_chunk or not torch.isfinite(mask).all():
            raise ValueError("invalid per-frame repaint mask")
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("repaint mask values must be in [0, 1]")
    digest = mask_content_sha256(masks)
    indices = sorted(int(c) for c in masks)
    selected = sorted({indices[0], indices[len(indices) // 2], indices[-1]})
    source_hash = file_sha256(video)
    key = hashlib.sha256(json.dumps([source_hash, digest, option.get("protected_mask_sha256"), 1]).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{key}.jpg"
    frame_indices = [c * frames_per_chunk for c in selected]
    if not target.exists():
        canvas = Image.new("RGB", (1000, 35 + 255 * len(selected)), "#181818")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "Parent frame | Parent detail | RED: repaint weight; BLUE: protected. Latent-grid preview.", fill="white")
        for row, chunk in enumerate(selected):
            frame_index = chunk * frames_per_chunk
            frames = read_video_frames(video, frame_index, 1)
            if len(frames) != 1:
                raise ValueError(f"cannot read mask source frame {frame_index}")
            original = frames[0].convert("RGB")
            width, height = original.size
            mask = torch.nn.functional.interpolate(masks[chunk][:1, None].float(), (height, width), mode="nearest")[0, 0].numpy()
            protect = np.zeros_like(mask)
            if chunk in protected:
                protect = torch.nn.functional.interpolate(protected[chunk][:1, None].float(), (height, width), mode="nearest")[0, 0].numpy()
            rgb = np.asarray(original).astype(np.float32)
            alpha = np.clip(mask, 0, 1)[..., None] * 0.55
            overlay = rgb * (1 - alpha) + np.array([255, 40, 40]) * alpha
            alpha = (protect >= 0.35)[..., None] * 0.35
            overlay = Image.fromarray(np.uint8(np.clip(overlay * (1 - alpha) + np.array([40, 160, 255]) * alpha, 0, 255)))
            yy, xx = np.where(mask > 0.05)
            box = (0, 0, width, height)
            if len(xx):
                margin = max(24, int(max(xx.max() - xx.min(), yy.max() - yy.min()) * 0.25))
                box = (max(0, int(xx.min()) - margin), max(0, int(yy.min()) - margin),
                       min(width, int(xx.max()) + margin + 1), min(height, int(yy.max()) + margin + 1))
            context = original.copy()
            ImageDraw.Draw(context).rectangle(box, outline="#ffff00", width=3)
            y = 35 + row * 255
            for x, panel, size in ((0, context, (400, 225)), (400, original.crop(box), (300, 225)),
                                    (700, overlay.crop(box), (300, 225))):
                canvas.paste(ImageOps.pad(panel, size, color="#181818"), (x, y))
            draw.text((8, y + 230), f"Source frame {frame_index}; chunk {chunk}; local frame 0", fill="white")
        canvas.save(target, quality=92)
    return {"path": str(target), "sha256": file_sha256(target), "source_video_sha256": source_hash,
            "frame_indices": frame_indices, "mask_content_sha256": digest,
            "coverage": sum(float((m >= 0.35).float().mean()) for m in masks.values()) / len(masks)}
