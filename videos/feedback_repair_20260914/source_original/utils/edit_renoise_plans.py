"""Normalisation of static masked-renoise edit plans.

Kept free of torch / model imports so the agent demo and its CPU tests can
validate cumulative repair plans without loading the diffusion pipeline.  The
pipeline imports these helpers; nothing here imports the pipeline.
"""

from __future__ import annotations

from typing import Any


def plain_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict / OmegaConf node / namespace uniformly."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        value = getattr(cfg, key, default)
    except Exception:
        return default
    return default if value is None else value


def normalize_edit_renoise_config(raw: Any) -> dict[str, Any] | None:
    """Return one enabled edit-renoise config in the pipeline's internal form."""
    if raw is None or not bool(plain_get(raw, "enabled", False)):
        return None
    return {
        "chunk_start": int(plain_get(raw, "chunk_start", -1)),
        "chunk_end": int(plain_get(raw, "chunk_end", -1)),
        "entry_index": int(plain_get(raw, "entry_index", 2)),
        "entry_sigma": plain_get(raw, "entry_sigma", None),
        "mask_path": str(plain_get(raw, "mask_path", "")),
        "mask_threshold": float(plain_get(raw, "mask_threshold", 0.35)),
        "mask_dilate": int(plain_get(raw, "mask_dilate", 1)),
        "prompt": str(plain_get(raw, "prompt", "")),
        "noise_seed": int(plain_get(raw, "noise_seed", 4242)),
        # cut-paste-SDEdit: initialize the masked region from a pasted
        # golden-subject latent instead of the drifted content.
        "collage": bool(plain_get(raw, "collage", False)),
        "donor_chunk": int(plain_get(raw, "donor_chunk", -1)),
        "donor_frame": int(plain_get(raw, "donor_frame", 0)),
        "donor_mask_path": str(plain_get(raw, "donor_mask_path", "")),
        "donor_flip": bool(plain_get(raw, "donor_flip", False)),
        "paste_pin": bool(plain_get(raw, "paste_pin", False)),
    }


def normalize_edit_renoise_plans(raw_plans: Any) -> list[dict[str, Any]]:
    """Normalize and validate a set of non-overlapping static edit plans.

    A list is deliberately required: accepting a mapping here would make a
    misspelled single-plan config iterate over its keys and fail much later.
    Disabled entries are ignored so generated experiment configs can retain
    explicit feature-off arms.
    """
    if raw_plans is None:
        return []
    if isinstance(raw_plans, (str, bytes, dict)):
        raise ValueError("inference.edit_renoise_plans must be a list of plans")
    try:
        items = list(raw_plans)
    except TypeError as exc:
        raise ValueError("inference.edit_renoise_plans must be a list of plans") from exc

    plans = []
    for index, raw in enumerate(items):
        plan = normalize_edit_renoise_config(raw)
        if plan is None:
            continue
        if plan["chunk_start"] < 0 or plan["chunk_end"] < plan["chunk_start"]:
            raise ValueError(
                "inference.edit_renoise_plans[%d] has invalid chunk range [%d, %d]"
                % (index, plan["chunk_start"], plan["chunk_end"])
            )
        plans.append(plan)

    plans.sort(key=lambda plan: (plan["chunk_start"], plan["chunk_end"]))
    for previous, current in zip(plans, plans[1:]):
        if current["chunk_start"] <= previous["chunk_end"]:
            raise ValueError(
                "inference.edit_renoise_plans overlap: [%d, %d] and [%d, %d]"
                % (
                    previous["chunk_start"],
                    previous["chunk_end"],
                    current["chunk_start"],
                    current["chunk_end"],
                )
            )
    return plans


__all__ = ["normalize_edit_renoise_config", "normalize_edit_renoise_plans", "plain_get"]
