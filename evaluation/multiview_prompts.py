# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Resolve vendored multi-view prompt sets to per-theme shot specs.

A theme folder (e.g. ``example/multiview_prompts/frying_egg_closeup/``) holds
one ``<i>.json`` caption file per shot plus an optional ``shot_durations.txt``
giving the per-shot generation-chunk counts (which need not be equal). The
generated videos are named ``<prefix>-rank<R>-<theme>-seed<S>_<model>.mp4`` by
``inference.py``, so matching a video back to its theme means stripping those
wrappers, not requiring ``stem == theme``.

The resolver returns, per theme, the ordered shot ``captions`` and the per-shot
``chunk_durations`` so the evaluator can convert them to exact decoded-frame
shot boundaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# inference.py names outputs "<prefix>-rank<rank>-<name>-seed<seed>_<model>".
_GENERATED_STEM = re.compile(r"^(?:.*?-)?rank\d+-(?P<name>.+)-seed\d+_[^-]*$")


def _filename_token(value: str) -> str:
    """Mirror inference.safe_filename_token so theme names match generated stems."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:120] or "sample"


def _shot_json_files(folder: Path) -> list[Path]:
    files = [f for f in folder.glob("*.json") if f.name != "global.json"]
    return sorted(
        files,
        key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem),
    )


def _load_durations_txt(folder: Path) -> list[int] | None:
    txt = folder / "shot_durations.txt"
    if not txt.exists():
        return None
    try:
        parts = txt.read_text(encoding="utf-8").replace(",", " ").split()
        durations = [int(x) for x in parts if x.strip()]
    except Exception:
        return None
    return durations or None


def _durations_from_json(captions_meta: list[dict]) -> list[int]:
    """Per-shot chunk counts from each JSON: num_blocks, else block_indices, else 1."""
    durations: list[int] = []
    for meta in captions_meta:
        n = meta.get("num_blocks")
        if not n:
            indices = meta.get("block_indices")
            n = len(indices) if isinstance(indices, (list, tuple)) and indices else 1
        durations.append(max(1, int(n)))
    return durations


def load_shot_specs(prompts_dir: str | Path, *, caption_field: str = "caption") -> dict[str, dict]:
    """Map ``theme -> {"captions": [...], "chunk_durations": [...]}``.

    Chunk durations use a three-level fallback: ``shot_durations.txt`` (truncated
    to the number of shots), else per-JSON ``num_blocks``/``block_indices``, else
    one chunk per shot. Themes with fewer than two shots are skipped (nothing to
    measure across).
    """
    root = Path(prompts_dir)
    caption_root = root / "caption" if (root / "caption").is_dir() else root
    specs: dict[str, dict] = {}
    for sub in sorted(p for p in caption_root.iterdir() if p.is_dir()):
        json_files = _shot_json_files(sub)
        if len(json_files) < 2:
            continue
        captions: list[str] = []
        metas: list[dict] = []
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            captions.append(str(data.get(caption_field, "")))
            metas.append(data if isinstance(data, dict) else {})

        durations = _load_durations_txt(sub)
        if durations is not None:
            durations = durations[: len(json_files)]
            if len(durations) < len(json_files):
                durations = durations + [durations[-1] if durations else 1] * (
                    len(json_files) - len(durations)
                )
        else:
            durations = _durations_from_json(metas)

        specs[sub.name] = {"captions": captions, "chunk_durations": durations}
    return specs


def clamp_chunk_durations(durations: list[int], max_chunks: int) -> list[int]:
    """Clamp per-shot chunk counts to a rendered ``num_blocks`` budget.

    Mirrors ``MultiTextConcatDataset._apply_shot_durations`` so the boundaries
    match what was actually generated: shots are filled in order until the budget
    is spent (trailing shots dropped); any leftover is folded into the last kept
    shot. With ``max_chunks <= 0`` the durations pass through unchanged.
    """
    if max_chunks is None or max_chunks <= 0:
        return list(durations)
    clamped: list[int] = []
    remaining = int(max_chunks)
    for d in durations:
        if remaining <= 0:
            break
        take = min(int(d), remaining)
        clamped.append(take)
        remaining -= take
    if remaining > 0 and clamped:
        clamped[-1] += remaining
    return clamped


def match_theme(stem: str, known_themes) -> str | None:
    """Resolve a generated video stem to one of ``known_themes``.

    Tries, in order: exact stem match; the ``rank<R>-<name>-seed<S>`` capture;
    a ``safe_filename_token`` map; and finally the longest theme contained in
    the stem (handles overlapping names like ``indoor_*`` deterministically).
    """
    themes = list(known_themes)
    theme_set = set(themes)
    if stem in theme_set:
        return stem

    token_map: dict[str, str] = {}
    for t in themes:
        token_map.setdefault(_filename_token(t), t)

    m = _GENERATED_STEM.match(stem)
    if m:
        name = m.group("name")
        if name in theme_set:
            return name
        if name in token_map:
            return token_map[name]

    if stem in token_map:
        return token_map[stem]

    contained = [t for t in themes if t in stem or _filename_token(t) in stem]
    if contained:
        return max(contained, key=len)
    return None


def build_spec_resolver(
    prompts_dir: str | Path,
    *,
    with_captions: bool = True,
    max_chunks: int | None = None,
    caption_field: str = "caption",
):
    """Return ``resolve(stem) -> spec | None`` over a multi-view prompts dir.

    ``spec`` carries ``chunk_durations`` (for exact boundaries) and, when
    ``with_captions`` is set, ``captions`` (for the prompt-adherence guard).
    ``max_chunks`` clamps the chunk budget to the rendered ``num_blocks`` so the
    boundaries line up with what was actually generated (and trailing shots that
    were never rendered are dropped, along with their captions). ``None`` means
    the stem matched no theme and the pair should be skipped.
    """
    specs = load_shot_specs(prompts_dir, caption_field=caption_field)
    themes = list(specs.keys())

    def resolve(stem: str):
        theme = match_theme(stem, themes)
        if theme is None:
            return None
        spec = specs[theme]
        durations = clamp_chunk_durations(spec["chunk_durations"], max_chunks)
        out = {"chunk_durations": durations, "theme": theme}
        if with_captions:
            out["captions"] = list(spec["captions"])[: len(durations)]
        return out

    return resolve
