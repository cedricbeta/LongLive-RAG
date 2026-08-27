"""Runtime glue for the edit_shot masked-renoise repair action.

Two pieces live here so that both the agent session and demo drivers can call
them without shelling out to the offline scripts:

* :func:`build_repair_prompt` -- the default repair-prompt construction: the
  shot's authored caption joined with the judge's drift description through a
  configurable template (``AgentRuntimeConfig.edit_prompt_template`` is the
  config hook).
* :func:`clipseg_soft_masks` / :func:`clipseg_chunk_masks` -- the CLIPSeg
  subject-mask flow of ``scripts/prepare_renoise_edit_masks.py`` as callable
  functions.  Heavy dependencies (transformers, cv2) are imported lazily so
  importing this module stays cheap for CPU-only tests.

Mask payloads use the exact on-disk format of the offline script
(``{"masks": {chunk: FloatTensor [F, 22, 40]}, ...}``) so files produced either
way are interchangeable with the pipeline's ``edit_renoise`` mask loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from utils.agent_runtime import DEFAULT_EDIT_PROMPT_TEMPLATE

CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"
FRAMES_PER_CHUNK = 8
LATENT_GRID = (22, 40)


def build_repair_prompt(
    caption: str,
    drift_description: str,
    *,
    template: str | None = None,
) -> str:
    """Join the authored shot caption with the judged drift description."""
    caption = str(caption or "").strip()
    drift = str(drift_description or "").strip()
    if not caption and not drift:
        raise ValueError("repair prompt needs an authored caption or a drift description")
    if not drift:
        return caption
    chosen = DEFAULT_EDIT_PROMPT_TEMPLATE if template is None else str(template)
    return chosen.format(caption=caption, drift=drift).strip()


def read_video_frames(video: str | Path, start: int, count: int) -> list[Any]:
    """Read ``count`` RGB PIL frames from ``video`` starting at frame ``start``."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    frames: list[Any] = []
    for _ in range(int(count)):
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(bgr[:, :, ::-1]))
    capture.release()
    return frames


class _ClipSegRunner:
    """Lazily constructed CLIPSeg model shared across calls."""

    _cache: dict[str, "_ClipSegRunner"] = {}

    def __init__(self, device: str) -> None:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = str(device)
        self.processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL_NAME)
        self.model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_NAME)
        self.model.to(self.device).eval()

    @classmethod
    def get(cls, device: str) -> "_ClipSegRunner":
        key = str(device)
        if key not in cls._cache:
            cls._cache[key] = cls(key)
        return cls._cache[key]


def clipseg_soft_masks(
    frames: Sequence[Any],
    phrase: str,
    *,
    device: str = "cuda:0",
    grid: tuple[int, int] = LATENT_GRID,
) -> torch.Tensor:
    """Soft CLIPSeg masks for ``frames``, resampled to the latent token grid.

    Returns a float tensor of shape ``[len(frames), grid_h, grid_w]`` on CPU.
    Identical math to scripts/prepare_renoise_edit_masks.py.
    """
    if not frames:
        raise ValueError("clipseg_soft_masks requires at least one frame")
    runner = _ClipSegRunner.get(device)
    with torch.no_grad():
        inputs = runner.processor(
            text=[str(phrase)] * len(frames),
            images=list(frames),
            return_tensors="pt",
            padding=True,
        ).to(runner.device)
        logits = runner.model(**inputs).logits  # [F, h, w]
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        soft = torch.sigmoid(logits)
        soft = torch.nn.functional.interpolate(
            soft.unsqueeze(1), size=tuple(grid), mode="bilinear", align_corners=False
        ).squeeze(1).cpu()
    return soft


def clipseg_chunk_masks(
    video: str | Path,
    chunks: Iterable[int],
    phrase: str,
    *,
    device: str = "cuda:0",
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    grid: tuple[int, int] = LATENT_GRID,
) -> dict[int, torch.Tensor]:
    """Per-chunk CLIPSeg subject masks for the requested chunks of ``video``."""
    masks: dict[int, torch.Tensor] = {}
    for chunk in sorted({int(item) for item in chunks}):
        frames = read_video_frames(video, chunk * frames_per_chunk, frames_per_chunk)
        if len(frames) < frames_per_chunk:
            raise RuntimeError(
                f"chunk {chunk}: only {len(frames)} of {frames_per_chunk} frames readable "
                f"from {video}"
            )
        masks[chunk] = clipseg_soft_masks(frames, phrase, device=device, grid=grid)
    return masks


def mask_coverage(
    masks: Mapping[int, torch.Tensor], *, threshold: float = 0.35
) -> dict[int, float]:
    """Fraction of latent tokens above ``threshold`` per chunk (QC metric)."""
    return {
        int(chunk): float((soft.float() >= float(threshold)).float().mean())
        for chunk, soft in masks.items()
    }


def save_mask_payload(
    masks: Mapping[int, torch.Tensor],
    out_path: str | Path,
    *,
    phrase: str,
    video: str | Path = "",
    grid: tuple[int, int] = LATENT_GRID,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
) -> Path:
    """Persist masks in the prepare_renoise_edit_masks.py payload format."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "masks": {int(k): v for k, v in masks.items()},
            "phrase": str(phrase),
            "video": str(video),
            "grid": list(grid),
            "frames_per_chunk": int(frames_per_chunk),
        },
        out,
    )
    return out
