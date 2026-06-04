#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Run baseline and KV-RAG inference with matched seeds, then evaluate outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.multiview_prompts import build_spec_resolver
from evaluation.video_consistency import (
    build_clip_adherence_scorer,
    compare_cross_perspective_dirs,
    compare_video_dirs,
    evaluate_cross_perspective_gate,
    save_metrics_json,
)

# The recommended single-scene multi-perspective settings for the modified
# variant (viewpoint-robust key + persistent scene anchors) are the argparse
# defaults of the --modified_* flags; build_kv_rag_from_args reads them so the
# AC-3.2 ranking can vary the key/value per run.


DEFAULT_KV_RAG = {
    "enabled": True,
    "top_k": 2,
    "max_entries": 32,
    "max_tokens_per_entry": 1024,
    "layers": [0, 7, 14, 21, 29],
    "min_frame_gap": 0,
    "retrieve_during_denoise": True,
    "retrieve_during_recache": False,
    "store_after_recache": True,
    "token_policy": "uniform",
    "similarity": "cosine",
    "verbose": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True, help="Base inference yaml.")
    parser.add_argument("--output_root", default="videos/kv_rag_ablation", help="Ablation output directory.")
    parser.add_argument("--metrics_json", default=None, help="Metric JSON path. Defaults inside output_root.")
    parser.add_argument("--generator_ckpt", default=None, help="Override checkpoints.generator_ckpt.")
    parser.add_argument("--lora_ckpt", default=None, help="Override checkpoints.lora_ckpt.")
    parser.add_argument("--no_lora_adapter", action="store_true", help="Remove adapter/lora_ckpt from generated configs.")
    parser.add_argument("--data_path", default=None, help="Override data.data_path.")
    parser.add_argument("--skip_generation", action="store_true", help="Only evaluate existing output directories.")
    parser.add_argument("--baseline_dir", default=None, help="Existing baseline output directory.")
    parser.add_argument("--kv_rag_dir", default=None, help="Existing KV-RAG output directory.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap for evaluation.")
    parser.add_argument("--stride", type=int, default=1, help="Evaluation frame stride.")
    parser.add_argument(
        "--mode",
        choices=("temporal", "cross_perspective"),
        default="temporal",
        help="temporal: adjacent/optical-flow consistency. cross_perspective: "
        "single-scene multi-viewpoint milestone gate over a vendored prompt set.",
    )
    parser.add_argument(
        "--prompts_dir",
        default=None,
        help="cross_perspective: vendored <theme>/{0..N}.json prompt set "
        "(e.g. example/multiview_prompts). Used to build the render subset and to "
        "resolve per-shot boundaries/captions during evaluation.",
    )
    parser.add_argument(
        "--prompt_subset",
        default=None,
        help="cross_perspective: comma-separated theme folder names to render "
        "(default: the first two sorted themes).",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=None,
        help="cross_perspective: equal-split shot count fallback when --prompts_dir is absent.",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="cross_perspective: rendered chunk budget for clamping boundaries "
        "(auto-derived from the config when generating).",
    )
    parser.add_argument(
        "--min_consistency_wins",
        type=int,
        default=2,
        help="cross_perspective: prompts where modified must beat baseline consistency.",
    )
    parser.add_argument(
        "--adherence_tolerance",
        type=float,
        default=0.0,
        help="cross_perspective: allowed per-prompt adherence drop (modified >= baseline - tol).",
    )
    parser.add_argument(
        "--dry_run_without_adherence",
        action="store_true",
        help="cross_perspective: consistency-only NON-GATE run that skips the CLIP "
        "prompt-adherence guard. This does NOT evaluate AC-7 and never reports a "
        "pass; for offline metric exploration only. The real gate enforces "
        "adherence by default.",
    )
    parser.add_argument("--clip_model", default="ViT-B-32", help="open_clip model for adherence.")
    parser.add_argument("--clip_pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--clip_device", default=None, help="Device for the CLIP adherence scorer.")
    # Modified-variant knobs so the AC-3.2 rendered ranking can vary the
    # key/value representation per run (defaults = recommended multi-view).
    parser.add_argument("--modified_retrieval_key_mode", default="salient_set")
    parser.add_argument("--modified_retrieval_value_mode", default="raw")
    parser.add_argument("--modified_scene_score_bonus", type=float, default=0.1)
    parser.add_argument("--modified_boundary_inject_anchors", type=int, default=2)
    parser.add_argument(
        "--modified_scene_memory_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle the persistent scene partition for the modified variant.",
    )
    return parser.parse_args()


def _modified_kv_rag_from_args(args) -> dict:
    """Build the modified-variant KV-RAG overrides from CLI (defaults=recommended)."""
    return {
        "scene_memory_enabled": bool(args.modified_scene_memory_enabled),
        "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
        "scene_score_bonus": float(args.modified_scene_score_bonus),
        "retrieval_key_mode": args.modified_retrieval_key_mode,
        "retrieval_value_mode": args.modified_retrieval_value_mode,
    }


def _num_blocks_from_cfg(cfg) -> int | None:
    """Replicate inference.py's num_blocks = num_output_frames // num_frame_per_block."""
    nfpb = _value(cfg, "model_kwargs", "num_frame_per_block")
    if nfpb is None:
        nfpb = getattr(cfg, "num_frame_per_block", 8)
    nof = getattr(cfg, "num_output_frames", None)
    if nof is None:
        shape = _value(cfg, "data", "image_or_video_shape")
        if shape is not None and len(shape) > 1:
            nof = shape[1]
    if not nof or not nfpb:
        return None
    return int(nof) // int(nfpb)


def build_prompt_subset(prompts_dir: str, subset: list[str] | None, dest: Path) -> list[str]:
    """Copy the chosen theme folders into a fresh subset dir for rendering."""
    src = Path(prompts_dir)
    caption_root = src / "caption" if (src / "caption").is_dir() else src
    available = sorted(
        p.name
        for p in caption_root.iterdir()
        if p.is_dir() and any(f.name != "global.json" for f in p.glob("*.json"))
    )
    chosen = subset or available[:2]
    missing = [t for t in chosen if t not in available]
    if missing:
        raise ValueError(f"--prompt_subset themes not found in {prompts_dir}: {missing}")
    if len(chosen) < 2:
        raise ValueError("cross_perspective gate needs at least two themes to render")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for theme in chosen:
        shutil.copytree(caption_root / theme, dest / theme)
    return chosen


def _set_nested(cfg, section: str, key: str, value) -> None:
    if section not in cfg or cfg[section] is None:
        cfg[section] = {}
    cfg[section][key] = value
    cfg[key] = value


def _delete_key(cfg, key: str) -> None:
    if key in cfg:
        del cfg[key]


def _apply_overrides(cfg, args: argparse.Namespace):
    if args.generator_ckpt:
        _set_nested(cfg, "checkpoints", "generator_ckpt", args.generator_ckpt)
    if args.lora_ckpt:
        _set_nested(cfg, "checkpoints", "lora_ckpt", args.lora_ckpt)
    if args.data_path:
        _set_nested(cfg, "data", "data_path", args.data_path)
    if args.no_lora_adapter:
        _delete_key(cfg, "adapter")
        _delete_key(cfg, "lora_ckpt")
        if "checkpoints" in cfg and cfg.checkpoints is not None and "lora_ckpt" in cfg.checkpoints:
            del cfg.checkpoints["lora_ckpt"]
    return cfg


def _value(cfg, section: str, key: str):
    if key in cfg:
        return cfg[key]
    if section in cfg and cfg[section] is not None and key in cfg[section]:
        return cfg[section][key]
    return None


def _looks_like_placeholder(path_value) -> bool:
    if path_value is None:
        return False
    text = str(path_value)
    return text.startswith("/path/to/") or "/path/to/" in text


def _path_exists(path_value) -> bool:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def _preflight_config(cfg, config_name: str) -> None:
    errors = []
    data_path = _value(cfg, "data", "data_path")
    generator_ckpt = _value(cfg, "checkpoints", "generator_ckpt")
    lora_ckpt = _value(cfg, "checkpoints", "lora_ckpt")
    adapter_enabled = "adapter" in cfg and cfg.adapter is not None

    for label, path_value, required in (
        ("data.data_path", data_path, True),
        ("checkpoints.generator_ckpt", generator_ckpt, generator_ckpt is not None),
        ("checkpoints.lora_ckpt", lora_ckpt, adapter_enabled and lora_ckpt is not None),
    ):
        if required and not path_value:
            errors.append(f"{label} is required but empty")
            continue
        if path_value and _looks_like_placeholder(path_value):
            errors.append(f"{label} still uses placeholder path: {path_value}")
            continue
        if path_value and not _path_exists(path_value):
            errors.append(f"{label} does not exist: {path_value}")

    if adapter_enabled and not lora_ckpt:
        errors.append("adapter is configured but checkpoints.lora_ckpt is missing")

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            f"{config_name} is not ready to run:\n  - {joined}\n"
            "Pass --generator_ckpt/--lora_ckpt/--data_path, edit the yaml, "
            "or use --no_lora_adapter for merged generator checkpoints."
        )


def write_variant_configs(
    cfg,
    output_root: Path,
    *,
    filename_from_sample_name: bool = False,
    modified_kv_rag_extra: dict | None = None,
) -> tuple[Path, Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_root / "baseline"
    rag_dir = output_root / "kv_rag"

    baseline = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    baseline.output_folder = str(baseline_dir)
    if "inference" not in baseline:
        baseline.inference = {}
    baseline.inference.output_folder = str(baseline_dir)
    baseline.inference.filename_prefix = "baseline"
    baseline.inference.kv_rag = {"enabled": False}
    if "kv_rag" in baseline:
        baseline.kv_rag.enabled = False

    rag = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    rag.output_folder = str(rag_dir)
    if "inference" not in rag:
        rag.inference = {}
    rag.inference.output_folder = str(rag_dir)
    rag.inference.filename_prefix = "kv_rag"
    existing = OmegaConf.to_container(rag.inference.get("kv_rag", {}), resolve=True) or {}
    merged = dict(DEFAULT_KV_RAG)
    merged.update(existing)
    merged["enabled"] = True
    if modified_kv_rag_extra:
        merged.update(modified_kv_rag_extra)
    rag.inference.kv_rag = merged

    # Theme-named output stems so the evaluator can match videos to prompt folders.
    if filename_from_sample_name:
        for variant in (baseline, rag):
            variant.inference.filename_from_sample_name = True
            variant.filename_from_sample_name = True

    baseline_cfg_path = config_dir / "baseline.yaml"
    rag_cfg_path = config_dir / "kv_rag.yaml"
    OmegaConf.save(baseline, baseline_cfg_path)
    OmegaConf.save(rag, rag_cfg_path)
    return baseline_cfg_path, rag_cfg_path, baseline_dir, rag_dir


def run_inference(config_path: Path) -> None:
    cmd = [sys.executable, str(ROOT / "inference.py"), "--config_path", str(config_path)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_temporal(args, output_root: Path) -> None:
    if args.skip_generation:
        if not args.baseline_dir or not args.kv_rag_dir:
            raise ValueError("--skip_generation requires --baseline_dir and --kv_rag_dir")
        baseline_dir = Path(args.baseline_dir)
        rag_dir = Path(args.kv_rag_dir)
    else:
        cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
        _preflight_config(cfg, args.config_path)
        baseline_cfg, rag_cfg, baseline_dir, rag_dir = write_variant_configs(cfg, output_root)
        run_inference(baseline_cfg)
        run_inference(rag_cfg)

    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "kv_rag_metrics.json"
    result = compare_video_dirs(
        baseline_dir,
        rag_dir,
        max_frames=args.max_frames,
        stride=max(1, args.stride),
    )
    save_metrics_json(result, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    print(f"Compared pairs: {result['num_pairs']}")
    print("Delta means (KV-RAG - baseline):")
    for name, stats in result["delta_summary"].items():
        print(f"  {name}: {stats['mean']:.6f}")


def _run_cross_perspective(args, output_root: Path) -> None:
    # AC-7 enforces prompt adherence; only an explicit dry run skips it (and that
    # run is NOT a gate -- it can never report a pass).
    require_adherence = not args.dry_run_without_adherence
    modified_settings = _modified_kv_rag_from_args(args)

    if require_adherence and not args.prompts_dir:
        raise ValueError(
            "The AC-7 gate requires --prompts_dir (for shot captions + the CLIP "
            "adherence guard). For a consistency-only non-gate run use "
            "--dry_run_without_adherence."
        )

    num_blocks = args.num_blocks
    if args.skip_generation:
        if not args.baseline_dir or not args.kv_rag_dir:
            raise ValueError("--skip_generation requires --baseline_dir and --kv_rag_dir")
        baseline_dir = Path(args.baseline_dir)
        rag_dir = Path(args.kv_rag_dir)
    else:
        cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
        if args.prompts_dir:
            subset = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else None
            chosen = build_prompt_subset(args.prompts_dir, subset, output_root / "prompt_subset")
            _set_nested(cfg, "data", "data_path", str(output_root / "prompt_subset"))
            print(f"[gate] rendering subset: {chosen}")
        if num_blocks is None:
            num_blocks = _num_blocks_from_cfg(cfg)
        print(f"[gate] modified variant: {modified_settings}")
        _preflight_config(cfg, args.config_path)
        baseline_cfg, rag_cfg, baseline_dir, rag_dir = write_variant_configs(
            cfg,
            output_root,
            filename_from_sample_name=True,
            modified_kv_rag_extra=modified_settings,
        )
        run_inference(baseline_cfg)
        run_inference(rag_cfg)

    if args.prompts_dir:
        print(f"[gate] resolving boundaries with num_blocks={num_blocks}")
        shots_for = build_spec_resolver(
            args.prompts_dir, with_captions=require_adherence, max_chunks=num_blocks
        )
    elif args.num_shots is not None:
        shots_for = args.num_shots
    else:
        raise ValueError("cross_perspective mode requires --prompts_dir or --num_shots")

    adherence_scorer = None
    if require_adherence:
        adherence_scorer = build_clip_adherence_scorer(
            model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
        )

    result = compare_cross_perspective_dirs(
        baseline_dir,
        rag_dir,
        shots_for=shots_for,
        adherence_scorer=adherence_scorer,
        max_frames=args.max_frames,
        stride=max(1, args.stride),
    )
    gate = evaluate_cross_perspective_gate(
        result,
        min_consistency_wins=args.min_consistency_wins,
        adherence_tolerance=args.adherence_tolerance,
        require_adherence=require_adherence,
    )
    gate["ac7_evaluated"] = require_adherence
    gate["modified_settings"] = modified_settings
    if not require_adherence:
        # Consistency-only dry run is not an AC-7 result; never claim a pass.
        gate["note"] = "AC-7 NOT evaluated: --dry_run_without_adherence skips the adherence guard."
        gate["passed"] = False

    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "kv_rag_cross_perspective.json"
    save_metrics_json({"gate": gate, "comparison": result}, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    print(f"Compared pairs: {result['num_pairs']} (skipped: {result.get('skipped_stems', [])})")
    print(
        f"[gate] consistency wins: {gate['consistency_wins']}/{gate['num_pairs']} "
        f"(need >= {gate['min_consistency_wins']})"
    )
    if require_adherence:
        print(
            f"[gate] adherence guard: {'OK' if gate['adherence_ok'] else 'FAIL'} "
            f"(tol={gate['adherence_tolerance']}, failures={gate['adherence_failures']})"
        )
    for p in gate["per_prompt"]:
        line = (
            f"  {p['stem']}: consistency {p['consistency_baseline']:.4f} -> "
            f"{p['consistency_modified']:.4f} ({'win' if p['consistency_win'] else 'no win'})"
        )
        if "adherence_modified_mean" in p:
            line += (
                f" | adherence {p['adherence_baseline_mean']:.4f} -> "
                f"{p['adherence_modified_mean']:.4f} ({'ok' if p['adherence_ok'] else 'REGRESS'})"
            )
        print(line)
    if not require_adherence:
        print("[gate] RESULT: DRY RUN -- AC-7 NOT evaluated (no adherence guard); not a pass.")
        return
    print(f"[gate] RESULT: {'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"]:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.mode == "cross_perspective":
        _run_cross_perspective(args, output_root)
    else:
        _run_temporal(args, output_root)


if __name__ == "__main__":
    main()
