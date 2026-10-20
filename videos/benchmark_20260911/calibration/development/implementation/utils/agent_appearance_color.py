"""Pixel-level colour checks for scripted hair / garment attributes.

The VLM comparer is nearly colour-blind in both directions: on ns090 it judged
dark hair compatible with "curly red hair" and, after the repair, red hair
compatible with the dark-haired reference.  This module measures colour
directly on the crop and can only veto: a clear, well-supported colour-family
conflict with the script (or with the frozen reference crop) marks the crop as
not matching.  It never turns a VLM "no" into a "yes".

Region heuristics are deliberately crude (hair = upper band of a person crop,
garment = torso band) so they hold for the tight person boxes the judge already
produces; they are validated against the frozen calibration labels rather than
tuned per case.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.agent_appearance_evidence import EvidenceAppearanceComparer

# colour word -> colour family.  Families are what the pixel classifier emits.
COLOR_WORDS: dict[str, str] = {
    **{w: "red" for w in ("red", "auburn", "ginger", "copper", "crimson", "scarlet", "wine", "burgundy", "maroon", "cherry", "rust")},
    **{w: "orange" for w in ("orange", "tangerine", "apricot", "peach")},
    **{w: "yellow" for w in ("yellow", "blonde", "blond", "golden", "gold", "mustard", "lemon")},
    **{w: "green" for w in ("green", "moss", "olive", "emerald", "sage", "mint", "lime", "forest")},
    **{w: "blue" for w in ("blue", "navy", "indigo", "denim", "cobalt", "azure", "sky", "teal", "cyan")},
    **{w: "purple" for w in ("purple", "violet", "lavender", "plum", "lilac", "mauve")},
    **{w: "pink" for w in ("pink", "magenta", "rose", "fuchsia", "salmon", "coral")},
    **{w: "brown" for w in ("brown", "chestnut", "mocha", "coffee", "hazel", "bronze", "chocolate", "walnut", "umber")},
    **{w: "tan" for w in ("tan", "beige", "khaki", "camel", "caramel", "taupe", "sand", "sandy", "cream", "ivory", "oatmeal", "ecru", "buff")},
    **{w: "black" for w in ("black", "dark", "jet", "ebony", "raven", "charcoal")},
    **{w: "white" for w in ("white", "silver", "platinum", "snowy")},
    **{w: "grey" for w in ("grey", "gray", "ash", "slate")},
}
HAIR_STYLE_WORDS = ("curly", "straight", "wavy", "short", "long", "bob", "shaved", "buzz", "bald", "ponytail", "braids", "braided", "cropped", "tidy", "messy", "slicked")
GARMENT_WORDS = ("coat", "jacket", "shirt", "sweater", "dress", "blouse", "t-shirt", "tee", "hoodie", "blazer", "cardigan", "turtleneck", "vest", "apron", "uniform", "top", "jumper", "overcoat", "trench", "parka", "suit", "gown", "robe", "tunic", "polo", "pullover")

# hue centres (degrees) for chromatic families
_HUE_CENTRES = {"red": 0.0, "orange": 25.0, "yellow": 50.0, "green": 120.0, "blue": 220.0, "purple": 280.0, "pink": 330.0}
ACHROMATIC = ("black", "white", "grey")


def _family_of(word: str) -> str | None:
    return COLOR_WORDS.get(word.lower().strip("-,."))


def parse_appearance_spec(description: str) -> dict[str, dict[str, Any]]:
    """Extract hair and garment colour families (and hair style) from a script description.

    "a middle-aged woman with curly red hair and a moss green coat" ->
    {"hair": {"colors": ["red"], "styles": ["curly"], "phrase": "curly red hair"},
     "garment": {"colors": ["green"], "type": "coat", "phrase": "moss green coat"}}
    Only what is explicitly written is returned; nothing is inferred.
    """
    text = description.lower().replace("-coloured", "").replace("-colored", "")
    tokens = re.findall(r"[a-z][a-z\-]*", text)
    out: dict[str, dict[str, Any]] = {}

    def window_before(index: int, size: int = 4) -> list[str]:
        return tokens[max(0, index - size):index]

    for i, tok in enumerate(tokens):
        if tok == "hair" and "hair" not in out:
            before = window_before(i)
            colors = [f for f in (_family_of(w) for w in before) if f]
            styles = [w for w in before if w in HAIR_STYLE_WORDS]
            # "dark hair" alone is a colour; "dark green coat" belongs to the garment branch
            if colors or styles:
                out["hair"] = {"colors": list(dict.fromkeys(colors)), "styles": styles, "phrase": " ".join(before + ["hair"])}
        elif tok.rstrip("s") in GARMENT_WORDS and "garment" not in out:
            before = window_before(i)
            colors = [f for f in (_family_of(w) for w in before) if f]
            if colors:
                out["garment"] = {"colors": list(dict.fromkeys(colors)), "type": tok, "phrase": " ".join(before + [tok])}
    return out


@dataclass(frozen=True)
class ColorMeasurement:
    family: str          # dominant family: red/orange/yellow/green/blue/purple/pink/brown/tan/black/white/grey/unknown
    fraction: float      # share of region pixels in the dominant family
    pixels: int
    median_hsv: tuple[float, float, float]
    shares: Mapping[str, float] = None  # share of every family (unknown excluded from the denominator)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "fraction": round(self.fraction, 3), "pixels": self.pixels,
                "median_hsv": [round(v, 3) for v in self.median_hsv],
                "shares": {k: round(v, 3) for k, v in (self.shares or {}).items() if v > 0.02}}


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    rc = np.where(mask, (mx - r) / np.where(mask, diff, 1), 0)
    gc = np.where(mask, (mx - g) / np.where(mask, diff, 1), 0)
    bc = np.where(mask, (mx - b) / np.where(mask, diff, 1), 0)
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = np.where(mask, h, 0.0)
    s = np.where(mx > 1e-6, diff / np.where(mx > 1e-6, mx, 1), 0.0)
    return np.stack([h * 360.0, s, mx], -1)


def classify_pixels(hsv: np.ndarray) -> np.ndarray:
    """Per-pixel colour family labels (ints indexing FAMILIES)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    labels = np.full(h.shape, FAMILIES.index("unknown"), dtype=np.int16)
    # "dark" means dark regardless of hue only when really dark; a saturated
    # burgundy coat (v ~0.3, s ~0.5) must stay chromatic
    dark = (v < 0.14) | ((v < 0.26) & (s < 0.35))
    achrom = (s < 0.16) & ~dark
    white = achrom & (v > 0.72)
    grey = achrom & ~white
    labels[dark] = FAMILIES.index("black")
    labels[white] = FAMILIES.index("white")
    labels[grey] = FAMILIES.index("grey")
    chrom = ~dark & ~achrom
    hue = h % 360.0
    warm = chrom & (hue >= 8) & (hue < 52)
    # brown: warm and darkish; tan: warm, light and weakly saturated (beige, khaki, camel)
    brown = warm & (v < 0.6)
    tan = warm & ~brown & (s < 0.45)
    labels[brown] = FAMILIES.index("brown")
    labels[tan] = FAMILIES.index("tan")
    rest = chrom & ~brown & ~tan
    bins = [("red", 345, 375), ("orange", 15, 42), ("yellow", 42, 72), ("green", 72, 170), ("blue", 170, 262), ("purple", 262, 300), ("pink", 300, 345)]
    hue2 = np.where(hue < 15, hue + 360.0, hue)
    for name, lo, hi in bins:
        sel = rest & (hue2 >= lo) & (hue2 < hi) if name == "red" else rest & (hue >= lo) & (hue < hi)
        labels[sel] = FAMILIES.index(name)
    return labels


FAMILIES = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "tan", "black", "white", "grey", "unknown"]

REGIONS = {
    # fallback bands of the person crop when no segmentation is available: (row0, row1, col0, col1)
    "hair": (0.0, 0.22, 0.2, 0.8),
    "garment": (0.36, 0.78, 0.22, 0.78),
}
PART_PHRASES = {"hair": "hair", "garment": "clothing"}
PART_THRESHOLD = 0.4
_PART_DEVICE = {"device": None}


def set_part_device(device: str | None) -> None:
    """Device for the CLIPSeg part segmenter (None = fixed crop bands only)."""
    _PART_DEVICE["device"] = device


def part_masks(image: "np.ndarray", *, device: str | None = None) -> dict[str, np.ndarray] | None:
    """CLIPSeg soft masks for hair / clothing at crop resolution, or None without a device."""
    device = device or _PART_DEVICE["device"]
    if device is None:
        return None
    import torch
    from PIL import Image
    from utils.agent_edit import _ClipSegRunner

    runner = _ClipSegRunner.get(device)
    pil = Image.fromarray(image)
    phrases = list(PART_PHRASES.values())
    with torch.no_grad():
        inputs = runner.processor(text=phrases, images=[pil] * len(phrases), return_tensors="pt", padding=True).to(runner.device)
        logits = runner.model(**inputs).logits
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        soft = torch.sigmoid(logits).unsqueeze(1)
        soft = torch.nn.functional.interpolate(soft, size=image.shape[:2], mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    return {name: soft[i] for i, name in enumerate(PART_PHRASES)}


def measure_region(image: "np.ndarray", region: str, masks: Mapping[str, np.ndarray] | None = None) -> ColorMeasurement:
    """Dominant colour family of a part; ``image`` is HxWx3 RGB uint8.

    With CLIPSeg ``masks`` the part is the mask above ``PART_THRESHOLD``
    (hair additionally restricted to the upper half so "hair" never grabs a
    fur collar); otherwise the fixed crop band is used.
    """
    H, W = image.shape[:2]
    if masks is not None and region in masks:
        sel = masks[region] >= PART_THRESHOLD
        if region == "hair":
            sel[int(0.55 * H):, :] = False
        if region == "garment":
            sel[: int(0.12 * H), :] = False
        band = image[sel]
    else:
        r0, r1, c0, c1 = REGIONS[region]
        band = image[int(r0 * H):max(int(r0 * H) + 1, int(r1 * H)), int(c0 * W):max(int(c0 * W) + 1, int(c1 * W))]
    if band.size == 0:
        return ColorMeasurement("unknown", 0.0, 0, (0.0, 0.0, 0.0))
    hsv = _rgb_to_hsv(band.reshape(-1, 3))
    labels = classify_pixels(hsv)
    counts = np.bincount(labels, minlength=len(FAMILIES)).astype(np.float64)
    counts[FAMILIES.index("unknown")] = 0
    total = int(labels.size)
    known = counts.sum()
    if known == 0:
        return ColorMeasurement("unknown", 0.0, total, tuple(float(x) for x in np.median(hsv, 0)), {})
    best = int(counts.argmax())
    shares = {FAMILIES[i]: float(counts[i] / known) for i in range(len(FAMILIES)) if counts[i] > 0}
    return ColorMeasurement(FAMILIES[best], float(counts[best] / known), total, tuple(float(x) for x in np.median(hsv, 0)), shares)


def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# Measured families that a spec family tolerates (besides itself and hue neighbours).
# Garments: dark dyes (burgundy, navy, moss) photograph as black, so black is
# tolerated for saturated spec colours.  Hair: "red hair" measured black is the
# ns090 failure itself, so hair tolerances are tighter (see _HAIR_TOLERATED).
_TOLERATED: dict[str, set[str]] = {
    "red": {"orange", "brown", "black"},          # auburn photographs brown; burgundy can go dark
    "orange": {"red", "yellow", "brown", "tan"},
    "yellow": {"orange", "tan", "white"},          # blonde hair reads tan/white in light
    "green": {"black", "grey"},                   # moss / dark green coats read dark
    "blue": {"black", "purple", "grey"},           # navy reads dark
    "purple": {"blue", "pink", "black"},
    "pink": {"red", "purple", "tan"},
    "brown": {"tan", "black", "red", "orange"},
    "tan": {"brown", "white", "grey", "yellow", "orange"},
    "black": {"brown", "grey", "blue", "purple", "green"},
    "white": {"grey", "tan", "yellow"},
    "grey": {"white", "black", "tan", "blue"},
}


_HAIR_TOLERATED: dict[str, set[str]] = {
    "red": {"orange", "brown"},
    "orange": {"red", "yellow", "brown", "tan"},
    "yellow": {"orange", "tan", "white"},
    "brown": {"tan", "black", "red", "orange"},
    "tan": {"brown", "white", "grey", "yellow"},
    "black": {"brown", "grey"},
    "white": {"grey", "tan", "yellow"},
    "grey": {"white", "black", "tan"},
}


def families_conflict(expected: Sequence[str], measured: str, region: str = "garment") -> bool:
    """True only for a far conflict; neighbouring hues and dark/warm ambiguity are tolerated."""
    if measured == "unknown" or not expected:
        return False
    table = _HAIR_TOLERATED if region == "hair" else _TOLERATED
    for exp in expected:
        if exp == measured or measured in table.get(exp, set()):
            return False
        if exp in _HUE_CENTRES and measured in _HUE_CENTRES:
            d = abs(_HUE_CENTRES[exp] - _HUE_CENTRES[measured]) % 360
            if min(d, 360 - d) < 60:
                return False
    return True


MIN_CONFLICT_SHARE = 0.5  # conflicting families must own this share of the part's classified pixels
MIN_PIXELS = 400


def conflict_share(measurement: ColorMeasurement, expected: Sequence[str], region: str) -> float:
    """Share of classified part pixels whose family conflicts with the spec."""
    return float(sum(share for fam, share in (measurement.shares or {}).items()
                     if families_conflict(expected, fam, region)))


def check_crop_against_spec(crop: str | Path, spec: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Measure hair/garment colour on ``crop`` and compare with parsed spec families."""
    image = load_rgb(crop)
    masks = part_masks(image)
    checks: dict[str, Any] = {"segmentation": "clipseg" if masks is not None else "fixed_band"}
    contradictions: list[str] = []
    for region in ("hair", "garment"):
        want = spec.get(region)
        if not want or not want.get("colors"):
            continue
        m = measure_region(image, region, masks)
        share = conflict_share(m, want["colors"], region)
        conflict = m.pixels >= MIN_PIXELS and share >= MIN_CONFLICT_SHARE
        checks[region] = {**m.to_dict(), "expected": list(want["colors"]), "conflict_share": round(share, 3), "conflict": bool(conflict)}
        if conflict:
            contradictions.append(f"{region} measured {m.family} ({share:.0%} of part conflicts) vs script {'/'.join(want['colors'])}")
    return {"checks": checks, "contradictions": contradictions}


def check_crop_against_reference(crop: str | Path, reference: str | Path) -> dict[str, Any]:
    """Same measurement on both crops; far colour-family changes are contradictions."""
    cur, ref = load_rgb(crop), load_rgb(reference)
    cur_masks, ref_masks = part_masks(cur), part_masks(ref)
    checks: dict[str, Any] = {"segmentation": "clipseg" if cur_masks is not None else "fixed_band"}
    contradictions: list[str] = []
    for region in ("hair", "garment"):
        a, b = measure_region(cur, region, cur_masks), measure_region(ref, region, ref_masks)
        share = conflict_share(a, [b.family], region) if b.fraction >= 0.5 else 0.0
        conflict = a.pixels >= MIN_PIXELS and b.pixels >= MIN_PIXELS and share >= MIN_CONFLICT_SHARE
        checks[region] = {"current": a.to_dict(), "reference": b.to_dict(), "conflict_share": round(share, 3), "conflict": bool(conflict)}
        if conflict:
            contradictions.append(f"{region} measured {a.family} vs reference {b.family}")
    return {"checks": checks, "contradictions": contradictions}


class ColorCheckedAppearanceComparer(EvidenceAppearanceComparer):
    """Evidence comparer plus a pixel colour veto against the script and the reference."""

    def _compare_crop(self, entity, reference_crop, crop, *, shot_index, inputs):
        result = dict(super()._compare_crop(entity, reference_crop, crop, shot_index=shot_index, inputs=inputs))
        spec = parse_appearance_spec(entity.description)
        pixel = {"spec": spec, "against_script": check_crop_against_spec(crop, spec)}
        contradictions = list(pixel["against_script"]["contradictions"])
        if reference_crop:
            pixel["against_reference"] = check_crop_against_reference(crop, reference_crop)
            contradictions += pixel["against_reference"]["contradictions"]
        if contradictions:
            result["matches_reference"] = False
            result["differences"] = "; ".join([d for d in (result.get("differences", ""),) if d] + contradictions)
        result["pixel_color"] = {**pixel, "vetoed": bool(contradictions)}
        return result


__all__ = [
    "COLOR_WORDS", "FAMILIES", "ColorCheckedAppearanceComparer", "ColorMeasurement",
    "check_crop_against_reference", "check_crop_against_spec", "conflict_share", "families_conflict",
    "measure_region", "parse_appearance_spec", "part_masks", "set_part_device",
]
