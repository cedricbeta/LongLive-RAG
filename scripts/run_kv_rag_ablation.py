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

from evaluation.multiview_prompts import build_spec_resolver, load_shot_specs
from evaluation.video_consistency import (
    build_clip_text_encoder,
    build_clip_adherence_scorer,
    build_raft_dynamic_scorer,
    compare_cross_perspective_dirs,
    compare_video_dirs,
    evaluate_cross_perspective_gate,
    prompt_text_similarity_lint,
    save_metrics_json,
)

# The recommended long-video multi-shot settings for the modified variant
# (content-robust key + persistent scene anchors) are the argparse defaults of
# the --modified_* flags. The older cross_perspective name is kept as a
# backward-compatible alias for the same concatenated multi-shot path.


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
        choices=("temporal", "long_multishot", "cross_perspective", "multiview_vbench"),
        default="temporal",
        help="temporal: adjacent/optical-flow consistency. long_multishot: "
        "one concatenated long video per scene, scored for within-video "
        "multi-shot scene consistency. cross_perspective is a legacy alias for "
        "long_multishot. multiview_vbench: render ONE video per perspective "
        "(multiview_per_perspective) for baseline + modified and score the "
        "AC-2 cross-video VBench suite (identity + dims) with the AC-4 gate.",
    )
    parser.add_argument(
        "--prompts_dir",
        default=None,
        help="long_multishot/cross_perspective: vendored <theme>/{0..N}.json "
        "prompt set (e.g. example/long_multishot_prompts). Used to build the "
        "render subset and to resolve per-shot boundaries/captions during evaluation.",
    )
    parser.add_argument(
        "--prompt_subset",
        default=None,
        help="long_multishot/cross_perspective: comma-separated theme folder "
        "names to render (default: the first two sorted themes).",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=None,
        help="long_multishot/cross_perspective: equal-split shot count fallback when --prompts_dir is absent.",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="long_multishot/cross_perspective: rendered chunk budget for clamping boundaries "
        "(auto-derived from the config when generating).",
    )
    parser.add_argument(
        "--min_consistency_wins",
        type=int,
        default=None,
        help="long_multishot/cross_perspective: prompts where modified must beat "
        "baseline consistency. Default is 1 for long_multishot, 2 for the "
        "legacy cross_perspective alias.",
    )
    parser.add_argument(
        "--adherence_tolerance",
        type=float,
        default=0.0,
        help="long_multishot/cross_perspective: allowed per-prompt adherence drop (modified >= baseline - tol).",
    )
    parser.add_argument(
        "--dry_run_without_adherence",
        action="store_true",
        help="long_multishot/cross_perspective: consistency-only NON-GATE run that skips the CLIP "
        "prompt-adherence guard. This does NOT evaluate AC-7 and never reports a "
        "pass; for offline metric exploration only. The real gate enforces "
        "adherence by default.",
    )
    parser.add_argument("--clip_model", default="ViT-B-32", help="open_clip model for adherence.")
    parser.add_argument("--clip_pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--clip_device", default=None, help="Device for the CLIP adherence scorer.")
    parser.add_argument("--prompt_similarity_floor", type=float, default=0.12,
                        help="long_multishot: minimum pairwise CLIP text similarity among shot captions; "
                        "below-floor scenes are rejected as ill-posed prompt data.")
    parser.add_argument("--motion_backend", default="raft", choices=("raft", "farneback"),
                        help="long_multishot: dynamic_degree backend. Use raft for gate runs; "
                        "farneback is intended for CPU tests/dry checks.")
    # multiview_vbench: which milestone backbones to load (off => GPU-free dims
    # only: temporal_style/appearance_style/overall_consistency + companions).
    parser.add_argument("--vbench_subject", action="store_true",
                        help="multiview_vbench: enable DINO subject-consistency (needs DINO).")
    parser.add_argument("--vbench_background", action="store_true",
                        help="multiview_vbench: enable CLIP background-consistency (needs CLIP).")
    parser.add_argument("--vbench_identity", action="store_true",
                        help="multiview_vbench: enable subject-IDENTITY consistency (ArcFace/DINO-patch).")
    parser.add_argument("--subject_kind", default="auto", choices=("auto", "human", "object"),
                        help="multiview_vbench: identity subject kind when --vbench_identity is set.")
    parser.add_argument("--max_perspectives", type=int, default=None,
                        help="multiview_vbench: cap perspectives rendered per scene (a LOGGED "
                        "coverage bound, AC-7; omit to render all).")
    parser.add_argument("--diversity_tolerance", type=float, default=0.0,
                        help="multiview_vbench: allowed inter_video_diversity drop (anti-collapse).")
    parser.add_argument("--motion_tolerance", type=float, default=None,
                        help="long_multishot/multiview_vbench: relative allowed dynamic_degree "
                        "drop (modified >= baseline*(1-tol)). Required for guarded runs.")
    parser.add_argument("--min_scene_wins", type=int, default=None,
                        help="multiview_vbench: scenes the modified aggregate must win (default ceil(N/2)).")
    parser.add_argument("--finalists", default=None,
                        help="long_multishot/multiview_vbench: comma-separated key:value finalists (e.g. "
                        "'pooled:raw,semantic:raw,subject_identity:raw'). When set, the "
                        "baseline is rendered ONCE and each finalist scored against it, then "
                        "ranked on the rendered aggregate into one consolidated JSON.")
    # Modified-variant knobs so the AC-3.2 rendered ranking can vary the
    # key/value representation per run (defaults = recommended multi-view).
    parser.add_argument("--modified_retrieval_key_mode", default="subject_identity")
    parser.add_argument("--modified_retrieval_value_mode", default="raw")
    parser.add_argument("--modified_scene_score_bonus", type=float, default=0.15)
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
    if not chosen:
        raise ValueError("long_multishot mode needs at least one theme to render")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for theme in chosen:
        shutil.copytree(caption_root / theme, dest / theme)
    return chosen


def build_multiview_subset(prompts_dir: str, subset: list[str] | None, dest: Path,
                           *, max_perspectives: int | None = None) -> tuple[list[str], dict]:
    """Copy chosen theme folders for per-perspective rendering, optionally capping
    perspectives per scene. Returns ``(chosen_themes, coverage)`` where coverage
    records the per-scene perspective counts actually rendered (AC-7: no silent
    caps -- the cap is logged, not hidden)."""
    src = Path(prompts_dir)
    caption_root = src / "caption" if (src / "caption").is_dir() else src
    available = sorted(
        p.name for p in caption_root.iterdir()
        if p.is_dir() and any(f.name != "global.json" for f in p.glob("*.json"))
    )
    chosen = subset or available[:2]
    missing = [t for t in chosen if t not in available]
    if missing:
        raise ValueError(f"--prompt_subset themes not found in {prompts_dir}: {missing}")
    # Cross-video scoring is per-SCENE over a scene's perspectives, so the real
    # requirement is >= 2 PERSPECTIVES per selected scene (one scene is fine; the
    # gate handles n=1 via min_scene_wins=ceil(1/2)=1). Two SCENES are not required.
    if not chosen:
        raise ValueError("multiview_vbench gate needs at least one scene with >= 2 perspectives")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, int] = {}
    for theme in chosen:
        out_dir = dest / theme
        out_dir.mkdir(parents=True, exist_ok=True)
        persp_jsons = sorted(
            (f for f in (caption_root / theme).glob("*.json") if f.name != "global.json"),
            key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem),
        )
        if max_perspectives is not None and max_perspectives > 0:
            persp_jsons = persp_jsons[:max_perspectives]
        if len(persp_jsons) < 2:
            raise ValueError(
                f"multiview_vbench scene {theme!r} has {len(persp_jsons)} perspective(s) "
                "to render; cross-video scoring needs >= 2 per scene (check "
                "--max_perspectives and the scene's <i>.json files)."
            )
        for jf in persp_jsons:
            shutil.copy2(jf, out_dir / jf.name)
        for extra in ("global.json", "shot_durations.txt"):
            ep = caption_root / theme / extra
            if ep.exists():
                shutil.copy2(ep, out_dir / extra)
        coverage[theme] = len(persp_jsons)
    return chosen, coverage


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
    multiview_per_perspective: bool = False,
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

    # One video per perspective (scenes contiguous so scene memory persists across
    # a scene's perspectives); the sample_name -> "<scene>-p<P>" stem is what the
    # VBench evaluator groups on.
    if multiview_per_perspective:
        for variant in (baseline, rag):
            variant.inference.multiview_per_perspective = True
            variant.multiview_per_perspective = True

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


def _build_long_scorers(args, *, require_adherence: bool) -> tuple[dict, dict]:
    """Build long_multishot metric/guard scorers, recording reduced coverage."""
    from evaluation.vbench_consistency import build_clip_image_encoder, build_dino_encoder

    subject = _try_build_backbone("long subject(DINO)", lambda: build_dino_encoder(device=args.clip_device))
    background = _try_build_backbone(
        "long background(CLIP)", lambda: build_clip_image_encoder(device=args.clip_device)
    )
    adherence = _try_build_backbone(
        "long adherence(CLIP)", lambda: build_clip_adherence_scorer(
            model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
        )
    ) if require_adherence else None
    text = _try_build_backbone(
        "long prompt-lint text(CLIP)", lambda: build_clip_text_encoder(
            model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
        )
    )
    if args.motion_backend == "raft":
        dynamic = _try_build_backbone(
            "long motion(RAFT)", lambda: build_raft_dynamic_scorer(device=args.clip_device)
        )
    else:
        from evaluation.video_consistency import farneback_dynamic_degree
        dynamic = farneback_dynamic_degree
    scorers = {
        "subject_encoder": subject,
        "background_encoder": background,
        "adherence_scorer": adherence,
        "invariant_scorer": adherence,
        "text_encoder": text,
        "dynamic_scorer": dynamic,
    }
    coverage = {
        "subject_dino": {"requested": True, "loaded": bool(subject)},
        "background_clip": {"requested": True, "loaded": bool(background)},
        "adherence_clip": {"requested": bool(require_adherence), "loaded": bool(adherence) if require_adherence else False},
        "prompt_lint_clip_text": {"requested": True, "loaded": bool(text)},
        "dynamic_degree": {"requested": args.motion_backend, "loaded": bool(dynamic)},
    }
    return scorers, coverage


def _long_prompt_lint(args) -> tuple[dict, list[str]]:
    """Return prompt-lint JSON and scenes allowed for method evaluation."""
    specs = load_shot_specs(args.prompts_dir)
    text_encoder = None
    try:
        text_encoder = build_clip_text_encoder(
            model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
        )
    except Exception as exc:
        lint = {
            scene: {
                "passed": False,
                "blocked_reason": f"CLIP text encoder unavailable: {exc}",
                "floor": float(args.prompt_similarity_floor),
                "matrix": [],
                "negative_control": bool(spec.get("negative_control", False)),
            }
            for scene, spec in specs.items()
        }
        return lint, []
    lint: dict[str, dict] = {}
    allowed: list[str] = []
    for scene, spec in specs.items():
        entry = prompt_text_similarity_lint(
            list(spec.get("captions", [])), text_encoder,
            floor=float(args.prompt_similarity_floor),
        )
        entry["negative_control"] = bool(spec.get("negative_control", False))
        lint[scene] = entry
        if entry.get("passed") or entry["negative_control"]:
            allowed.append(scene)
    return lint, allowed


def _filter_resolver(base_resolver, allowed_scenes: set[str]):
    def resolve(stem: str):
        spec = base_resolver(stem)
        if spec is None:
            return None
        if allowed_scenes and spec.get("theme") not in allowed_scenes:
            return None
        return spec
    return resolve


def _run_cross_perspective(args, output_root: Path) -> None:
    mode_label = "long_multishot" if args.mode == "long_multishot" else "cross_perspective"
    if args.finalists and args.mode == "long_multishot":
        return _run_long_multishot_finalists(args, output_root)
    # AC-7 enforces prompt adherence; only an explicit dry run skips it (and that
    # run is NOT a gate -- it can never report a pass).
    require_adherence = not args.dry_run_without_adherence
    modified_settings = _modified_kv_rag_from_args(args)
    min_consistency_wins = args.min_consistency_wins
    if min_consistency_wins is None:
        min_consistency_wins = 1 if args.mode == "long_multishot" else 2

    if require_adherence and not args.prompts_dir:
        raise ValueError(
            f"The {mode_label} gate requires --prompts_dir (for shot captions + the CLIP "
            "adherence guard). For a consistency-only non-gate run use "
            "--dry_run_without_adherence."
        )
    if args.mode == "long_multishot" and require_adherence and args.motion_tolerance is None:
        raise ValueError(
            "The long_multishot gate requires --motion_tolerance as a RELATIVE "
            "dynamic_degree non-regression guard, e.g. --motion_tolerance 0.2."
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
            print(f"[{mode_label}] rendering subset: {chosen}")
        if num_blocks is None:
            num_blocks = _num_blocks_from_cfg(cfg)
        print(f"[{mode_label}] modified variant: {modified_settings}")
        _preflight_config(cfg, args.config_path)
        baseline_cfg, rag_cfg, baseline_dir, rag_dir = write_variant_configs(
            cfg,
            output_root,
            filename_from_sample_name=True,
            modified_kv_rag_extra=modified_settings,
        )
        run_inference(baseline_cfg)
        run_inference(rag_cfg)

    prompt_lint = {}
    scorer_coverage = {}
    scorers = {
        "subject_encoder": None,
        "background_encoder": None,
        "adherence_scorer": None,
        "invariant_scorer": None,
        "dynamic_scorer": None,
    }
    if args.prompts_dir and args.mode == "long_multishot":
        prompt_lint, _allowed = _long_prompt_lint(args)
        scorers, scorer_coverage = _build_long_scorers(args, require_adherence=require_adherence)

    if args.prompts_dir:
        print(f"[{mode_label}] resolving boundaries with num_blocks={num_blocks}")
        shots_for = build_spec_resolver(
            args.prompts_dir, with_captions=require_adherence, max_chunks=num_blocks
        )
    elif args.num_shots is not None:
        shots_for = args.num_shots
    else:
        raise ValueError(f"{mode_label} mode requires --prompts_dir or --num_shots")

    adherence_scorer = scorers.get("adherence_scorer")
    if require_adherence and args.mode != "long_multishot":
        adherence_scorer = build_clip_adherence_scorer(
            model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
        )

    result = compare_cross_perspective_dirs(
        baseline_dir,
        rag_dir,
        shots_for=shots_for,
        adherence_scorer=adherence_scorer,
        invariant_scorer=scorers.get("invariant_scorer"),
        subject_encoder=scorers.get("subject_encoder"),
        background_encoder=scorers.get("background_encoder"),
        dynamic_scorer=scorers.get("dynamic_scorer"),
        max_frames=args.max_frames,
        stride=max(1, args.stride),
    )
    gate = evaluate_cross_perspective_gate(
        result,
        min_consistency_wins=min_consistency_wins,
        adherence_tolerance=args.adherence_tolerance,
        diversity_tolerance=args.diversity_tolerance,
        motion_tolerance=args.motion_tolerance,
        require_adherence=require_adherence,
        require_invariant=(args.mode == "long_multishot"),
    )
    gate["ac7_evaluated"] = require_adherence
    gate["modified_settings"] = modified_settings
    gate["scorer_coverage"] = scorer_coverage
    gate["prompt_lint"] = prompt_lint
    if args.mode == "long_multishot":
        selected = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else list(prompt_lint)
        lint_failures = [
            s for s in selected
            if s in prompt_lint
            and not prompt_lint[s].get("negative_control", False)
            and not prompt_lint[s].get("passed", False)
        ]
        missing_scorers = sorted(
            name for name, info in scorer_coverage.items()
            if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
        )
        blocked = []
        if lint_failures:
            blocked.append(f"ill-posed prompt scene(s) below text-similarity floor: {lint_failures}")
        if missing_scorers:
            blocked.append(f"requested scorer(s) unavailable: {missing_scorers}")
        if blocked:
            prev = gate.get("blocked_reason")
            gate["blocked_reason"] = "; ".join(([prev] if prev else []) + blocked)
            gate["passed"] = False
            gate["is_null_result"] = True
    if not require_adherence:
        # Consistency-only dry run is not an AC-7 result; never claim a pass.
        gate["note"] = "AC-7 NOT evaluated: --dry_run_without_adherence skips the adherence guard."
        gate["passed"] = False

    default_metrics_name = (
        "kv_rag_long_multishot.json"
        if args.mode == "long_multishot"
        else "kv_rag_cross_perspective.json"
    )
    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / default_metrics_name
    save_metrics_json({"gate": gate, "comparison": result}, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    print(f"Compared pairs: {result['num_pairs']} (skipped: {result.get('skipped_stems', [])})")
    print(
        f"[{mode_label}] consistency wins: {gate['consistency_wins']}/{gate['num_pairs']} "
        f"(need >= {gate['min_consistency_wins']})"
    )
    if require_adherence:
        print(
            f"[{mode_label}] adherence guard: {'OK' if gate['adherence_ok'] else 'FAIL'} "
            f"(tol={gate['adherence_tolerance']}, failures={gate['adherence_failures']})"
        )
    for p in gate["per_prompt"]:
        line = (
            f"  {p['stem']}: {p['metric']} {p['consistency_baseline']:.4f} -> "
            f"{p['consistency_modified']:.4f} ({'win' if p['consistency_win'] else 'no win'})"
        )
        if "adherence_modified_mean" in p:
            line += (
                f" | adherence {p['adherence_baseline_mean']:.4f} -> "
                f"{p['adherence_modified_mean']:.4f} ({'ok' if p['adherence_ok'] else 'REGRESS'})"
            )
        print(line)
    if not require_adherence:
        print(f"[{mode_label}] RESULT: DRY RUN -- AC-7 NOT evaluated (no adherence guard); not a pass.")
        return
    print(f"[{mode_label}] RESULT: {'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"]:
        raise SystemExit(1)


def _write_one_variant(cfg, output_root: Path, name: str, *, kv_rag_settings: dict,
                       multiview_per_perspective: bool = False) -> tuple[Path, Path]:
    """Write a single inference config (baseline OR one modified finalist) for the
    per-perspective render and return ``(config_path, output_dir)``."""
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / name
    variant = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    variant.output_folder = str(out_dir)
    if "inference" not in variant or variant.inference is None:
        variant.inference = {}
    variant.inference.output_folder = str(out_dir)
    variant.inference.filename_prefix = name
    variant.inference.kv_rag = kv_rag_settings
    variant.inference.filename_from_sample_name = True
    variant.filename_from_sample_name = True
    if multiview_per_perspective:
        variant.inference.multiview_per_perspective = True
        variant.multiview_per_perspective = True
    cfg_path = config_dir / f"{name}.yaml"
    OmegaConf.save(variant, cfg_path)
    return cfg_path, out_dir


def _parse_finalists(spec: str) -> list[tuple[str, str]]:
    """Parse 'key:value,key:value' into [(key, value), ...] (value defaults to raw)."""
    out: list[tuple[str, str]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        key, _, value = tok.partition(":")
        out.append((key.strip(), value.strip() or "raw"))
    if len(out) < 2:
        raise ValueError("--finalists needs >= 2 key:value finalists (e.g. "
                         "'pooled:raw,subject_identity:raw')")
    return out


def _try_build_backbone(label: str, builder):
    """Build an optional AC-2 backbone, returning None (with a logged warning) if it
    cannot load -- so a missing model/package records reduced coverage instead of
    crashing the gate post-render."""
    try:
        return builder()
    except Exception as exc:
        print(f"[multiview_vbench][warn] {label} backbone unavailable -> dim omitted: {exc}")
        return None


def _missing_requested_backbones(backbones: dict) -> list[str]:
    """Names of AC-2 backbones that were REQUESTED (--vbench_*) but failed to load."""
    return sorted(
        name for name, info in backbones.items()
        if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
    )


def _finalize_finalist_ranking(finalist_records: list, missing_requested_backbones: list,
                               *, gate_evaluated: bool = True):
    """Rank finalists and decide winner/null -- FAIL-CLOSED.

    A winner must NOT be selected when the rendered selector is incomplete or the
    full gate did not run, because a reduced/unguarded "win" would violate AC-3/AC-6
    and the AC-4 honest-null contract. Every finalist is forced ``passed=False`` with
    a ``blocked_reason`` when EITHER:
      * a REQUESTED AC-2 backbone failed to load (reduced suite), OR
      * the adherence guard was skipped (``gate_evaluated=False``, i.e. a
        ``--dry_run_without_adherence`` run -- the gate's ``passed`` is not a real pass).
    Otherwise the best passing finalist (by mean rendered aggregate delta) wins.
    Returns ``(ranked, winner, blocked_reason)``.
    """
    reasons = []
    if missing_requested_backbones:
        reasons.append(f"blocking prerequisite(s) unavailable or rejected: {list(missing_requested_backbones)}")
    if not gate_evaluated:
        reasons.append("adherence guard skipped (--dry_run_without_adherence)")
    blocked_reason = None
    if reasons:
        blocked_reason = "; ".join(reasons) + " -- refusing to select a rendered winner (fail-closed)."
        for fr in finalist_records:
            fr["passed"] = False
            fr["blocked_reason"] = blocked_reason
    ranked = sorted(finalist_records, key=lambda r: (r["passed"], r["mean_aggregate_delta"]),
                    reverse=True)
    winner = next((r for r in ranked if r["passed"]), None)
    return ranked, winner, blocked_reason


def _finalist_kv_rag(base_kv_rag: dict | None, settings: dict) -> dict:
    """Merge the modified-variant KV-RAG block for one finalist.

    Mirrors ``write_variant_configs``: start from ``DEFAULT_KV_RAG``, overlay the
    base config's own ``inference.kv_rag`` block (so layers / token limits / key
    hyperparameters from the requested YAML are honored), force ``enabled``, then
    apply the finalist key/value/scene-memory ``settings`` LAST so they win.
    """
    merged = dict(DEFAULT_KV_RAG)
    merged.update(base_kv_rag or {})
    merged["enabled"] = True
    merged.update(settings)
    return merged


def _base_kv_rag_block(cfg) -> dict:
    """The base config's ``inference.kv_rag`` block as a plain dict (or {}).

    Handles the supported shorthands explicitly (mirroring
    ``KVRAGConfig._to_plain_dict``): a boolean ``kv_rag: false/true`` becomes
    ``{"enabled": bool}`` and a plain dict passes through, so a non-``DictConfig``
    block never reaches ``OmegaConf.to_container`` (which raises on a bool).
    """
    inf = cfg.get("inference") if "inference" in cfg else None
    block = inf.get("kv_rag") if (inf is not None and "kv_rag" in inf) else None
    if block is None and "kv_rag" in cfg:
        block = cfg.get("kv_rag")
    if block is None:
        return {}
    if isinstance(block, bool):
        return {"enabled": block}
    if OmegaConf.is_config(block):
        return OmegaConf.to_container(block, resolve=True) or {}
    if isinstance(block, dict):
        return dict(block)
    return {}


def _run_long_multishot_finalists(args, output_root: Path) -> None:
    """Consolidated long_multishot gate: baseline vs multiple KV finalists."""
    import numpy as np

    require_adherence = not args.dry_run_without_adherence
    if require_adherence and not args.prompts_dir:
        raise ValueError("long_multishot finalist gate requires --prompts_dir.")
    if require_adherence and args.motion_tolerance is None:
        raise ValueError(
            "long_multishot finalist gate requires --motion_tolerance as a relative "
            "dynamic_degree non-regression guard."
        )
    finalists = _parse_finalists(args.finalists)

    cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
    subset = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else None
    chosen = build_prompt_subset(args.prompts_dir, subset, output_root / "prompt_subset")
    _set_nested(cfg, "data", "data_path", str(output_root / "prompt_subset"))
    num_blocks = args.num_blocks if args.num_blocks is not None else _num_blocks_from_cfg(cfg)
    base_kv_rag = _base_kv_rag_block(cfg)
    print(f"[long-finalist-gate] subset: {chosen}")
    print(f"[long-finalist-gate] finalists: {finalists}")
    _preflight_config(cfg, args.config_path)

    prompt_lint, _allowed = _long_prompt_lint(args)
    chosen_lint = {scene: prompt_lint.get(scene) for scene in chosen if scene in prompt_lint}
    lint_failures = [
        scene for scene, entry in chosen_lint.items()
        if entry and not entry.get("negative_control", False) and not entry.get("passed", False)
    ]
    lint_passing_main = [
        scene for scene, entry in chosen_lint.items()
        if entry and not entry.get("negative_control", False) and entry.get("passed", False)
    ]
    negative_controls = [
        scene for scene, entry in chosen_lint.items()
        if entry and entry.get("negative_control", False)
    ]

    scorers, scorer_coverage = _build_long_scorers(args, require_adherence=require_adherence)
    missing_scorers = sorted(
        name for name, info in scorer_coverage.items()
        if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
    )
    prereq_blocks = []
    if lint_failures:
        prereq_blocks.append(f"ill-posed prompt scene(s) below text-similarity floor: {lint_failures}")
    if len(lint_passing_main) < 3:
        prereq_blocks.append(f"need >=3 lint-passing non-control scenes, got {len(lint_passing_main)}")
    if not negative_controls:
        prereq_blocks.append("negative control scene missing")
    if missing_scorers:
        prereq_blocks.append(f"requested scorer(s) unavailable: {missing_scorers}")
    if prereq_blocks:
        blocked_reason = "; ".join(prereq_blocks)
        finalist_records = [
            {
                "key": key,
                "value": value,
                "settings": {
                    "scene_memory_enabled": bool(args.modified_scene_memory_enabled),
                    "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
                    "scene_score_bonus": float(args.modified_scene_score_bonus),
                    "retrieval_key_mode": key,
                    "retrieval_value_mode": value,
                },
                "passed": False,
                "blocked_reason": blocked_reason,
                "mean_aggregate_delta": None,
                "scene_wins": 0,
                "num_scenes": len(lint_passing_main),
            }
            for key, value in finalists
        ]
        consolidated = {
            "is_prefilter": False,
            "selector": "long_multishot shot-anchor centroid gate",
            "winner": None,
            "is_null_result": True,
            "blocked_reason": blocked_reason,
            "render_attempted": False,
            "prompt_lint": chosen_lint,
            "lint_passing_main_scenes": lint_passing_main,
            "negative_controls": negative_controls,
            "scorer_coverage": scorer_coverage,
            "guards": {
                "adherence_tolerance": args.adherence_tolerance,
                "diversity_tolerance_relative": args.diversity_tolerance,
                "motion_tolerance_relative": args.motion_tolerance,
                "invariant_tolerance": 0.0,
                "require_adherence": require_adherence,
            },
            "ranking": [
                {
                    "key": r["key"],
                    "value": r["value"],
                    "passed": False,
                    "mean_aggregate_delta": r["mean_aggregate_delta"],
                    "scene_wins": f"0/{len(lint_passing_main)}",
                    "blocked_reason": blocked_reason,
                }
                for r in finalist_records
            ],
            "finalists": finalist_records,
        }
        metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "long_multishot_finalist_gate.json"
        save_metrics_json(consolidated, metrics_json)
        print(f"Wrote metrics: {metrics_json.resolve()}")
        print(f"[long-finalist-gate] BLOCKED before render (fail-closed): {blocked_reason}")
        return

    baseline_cfg, baseline_dir = _write_one_variant(
        cfg, output_root, "baseline", kv_rag_settings={"enabled": False},
        multiview_per_perspective=False,
    )
    run_inference(baseline_cfg)

    shots_for = build_spec_resolver(
        args.prompts_dir, with_captions=require_adherence, max_chunks=num_blocks
    )

    finalist_records = []
    for key, value in finalists:
        settings = {
            "scene_memory_enabled": bool(args.modified_scene_memory_enabled),
            "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
            "scene_score_bonus": float(args.modified_scene_score_bonus),
            "retrieval_key_mode": key,
            "retrieval_value_mode": value,
        }
        merged = _finalist_kv_rag(base_kv_rag, settings)
        name = f"mod_{key}_{value}"
        mod_cfg, mod_dir = _write_one_variant(
            cfg, output_root, name, kv_rag_settings=merged,
            multiview_per_perspective=False,
        )
        print(f"[long-finalist-gate] rendering {name}: {settings}")
        run_inference(mod_cfg)
        result = compare_cross_perspective_dirs(
            baseline_dir, mod_dir, shots_for=shots_for,
            adherence_scorer=scorers.get("adherence_scorer"),
            invariant_scorer=scorers.get("invariant_scorer"),
            subject_encoder=scorers.get("subject_encoder"),
            background_encoder=scorers.get("background_encoder"),
            dynamic_scorer=scorers.get("dynamic_scorer"),
            max_frames=args.max_frames, stride=max(1, args.stride),
        )
        gate = evaluate_cross_perspective_gate(
            result, min_consistency_wins=args.min_consistency_wins or -(-len(lint_passing_main) // 2),
            adherence_tolerance=args.adherence_tolerance,
            diversity_tolerance=args.diversity_tolerance,
            motion_tolerance=args.motion_tolerance,
            require_adherence=require_adherence,
            require_invariant=True,
        )
        gate["prompt_lint"] = chosen_lint
        gate["scorer_coverage"] = scorer_coverage
        gate["modified_settings"] = settings
        blocked = []
        if lint_failures:
            blocked.append(f"ill-posed prompt scene(s) below text-similarity floor: {lint_failures}")
        if len(lint_passing_main) < 3:
            blocked.append(f"need >=3 lint-passing non-control scenes, got {len(lint_passing_main)}")
        if not negative_controls:
            blocked.append("negative control scene missing")
        if missing_scorers:
            blocked.append(f"requested scorer(s) unavailable: {missing_scorers}")
        if blocked:
            prev = gate.get("blocked_reason")
            gate["blocked_reason"] = "; ".join(([prev] if prev else []) + blocked)
            gate["passed"] = False
            gate["is_null_result"] = True
        deltas = [
            r["modified_metrics"].get("anchor_centroid_consistency", float("nan"))
            - r["baseline_metrics"].get("anchor_centroid_consistency", float("nan"))
            for r in result["records"]
            if not r.get("negative_control", False)
        ]
        mean_delta = float(np.nanmean(deltas)) if deltas else float("nan")
        finalist_records.append({
            "key": key,
            "value": value,
            "settings": settings,
            "passed": bool(gate["passed"]),
            "mean_aggregate_delta": mean_delta,
            "scene_wins": gate["consistency_wins"],
            "num_scenes": len(lint_passing_main),
            "gate": gate,
            "comparison": result,
        })

    ranked, winner, blocked_reason = _finalize_finalist_ranking(
        finalist_records,
        missing_scorers + lint_failures + ([] if len(lint_passing_main) >= 3 else ["insufficient_lint_passing_scenes"])
        + ([] if negative_controls else ["missing_negative_control"]),
        gate_evaluated=require_adherence,
    )
    consolidated = {
        "is_prefilter": False,
        "selector": "long_multishot shot-anchor centroid gate",
        "winner": ({"key": winner["key"], "value": winner["value"]} if winner else None),
        "is_null_result": winner is None,
        "blocked_reason": blocked_reason,
        "render_attempted": True,
        "prompt_lint": chosen_lint,
        "lint_passing_main_scenes": lint_passing_main,
        "negative_controls": negative_controls,
        "scorer_coverage": scorer_coverage,
        "guards": {
            "adherence_tolerance": args.adherence_tolerance,
            "diversity_tolerance_relative": args.diversity_tolerance,
            "motion_tolerance_relative": args.motion_tolerance,
            "invariant_tolerance": 0.0,
            "require_adherence": require_adherence,
        },
        "ranking": [
            {
                "key": r["key"],
                "value": r["value"],
                "passed": r["passed"],
                "mean_aggregate_delta": r["mean_aggregate_delta"],
                "scene_wins": f"{r['scene_wins']}/{r['num_scenes']}",
            }
            for r in ranked
        ],
        "finalists": finalist_records,
    }
    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "long_multishot_finalist_gate.json"
    save_metrics_json(consolidated, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    if blocked_reason:
        print(f"[long-finalist-gate] BLOCKED (fail-closed): {blocked_reason}")
    for r in ranked:
        print(f"  {r['key']}+{r['value']}: mean_delta={r['mean_aggregate_delta']:+.4f} "
              f"wins={r['scene_wins']}/{r['num_scenes']} passed={r['passed']}")
    if winner:
        print(f"[long-finalist-gate] WINNER: {winner['key']}+{winner['value']}")
    else:
        print("[long-finalist-gate] RESULT: honest NULL/BLOCKED -- no finalist passed all guards.")


def _run_multiview_finalists(args, output_root: Path) -> None:
    """AC-3/AC-4/AC-6 consolidated finalist gate: render the baseline ONCE and each
    finalist (key,value) against it, score the FULL AC-2 suite, rank on the rendered
    aggregate, and write one ranked JSON (winner or honest null). FAIL-CLOSED: a
    requested AC-2 backbone that cannot load forces a null, never a reduced-suite win."""
    from evaluation.vbench_consistency import (
        build_clip_image_encoder, build_dino_encoder, build_identity_encoder,
        compare_multiview_vbench_dirs, evaluate_multiview_vbench_gate,
    )
    from evaluation.multiview_prompts import load_shot_specs
    from evaluation.retrieval_screen import run_screen
    import numpy as np

    require_adherence = not args.dry_run_without_adherence
    if require_adherence and not args.prompts_dir:
        raise ValueError("multiview_vbench finalist gate requires --prompts_dir.")
    if require_adherence and args.motion_tolerance is None:
        raise ValueError("multiview_vbench finalist gate requires --motion_tolerance "
                         "(AC-4 motion guard); pass e.g. --motion_tolerance 0.3.")
    finalists = _parse_finalists(args.finalists)

    cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
    subset = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else None
    chosen, coverage = build_multiview_subset(
        args.prompts_dir, subset, output_root / "prompt_subset",
        max_perspectives=args.max_perspectives,
    )
    _set_nested(cfg, "data", "data_path", str(output_root / "prompt_subset"))
    base_kv_rag = _base_kv_rag_block(cfg)  # honor the requested config's KV-RAG block
    print(f"[finalist-gate] subset: {chosen}")
    print(f"[finalist-gate] perspective coverage (AC-7 -- logged): {coverage}")
    print(f"[finalist-gate] finalists: {finalists}")
    _preflight_config(cfg, args.config_path)

    # Render the baseline (no scene memory) ONCE, shared across finalists.
    baseline_cfg, baseline_dir = _write_one_variant(
        cfg, output_root, "baseline", kv_rag_settings={"enabled": False},
        multiview_per_perspective=True,
    )
    run_inference(baseline_cfg)

    captions_for = {s: v["captions"] for s, v in load_shot_specs(args.prompts_dir).items()}
    adherence_scorer = build_clip_adherence_scorer(
        model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
    ) if require_adherence else None

    dino = _try_build_backbone("subject(DINO)", lambda: build_dino_encoder(device=args.clip_device)) \
        if args.vbench_subject else None
    clip = _try_build_backbone("background(CLIP)", lambda: build_clip_image_encoder(device=args.clip_device)) \
        if args.vbench_background else None
    # Pre-check identity availability once (to record {requested, loaded} + fail
    # closed); the actual encoder is built FRESH PER SCENE via the factory below,
    # because build_identity_encoder('auto') is stateful and must not be shared.
    identity_probe = _try_build_backbone("identity", lambda: build_identity_encoder(
        subject_kind=args.subject_kind, device=args.clip_device, eager_fallback=True)) \
        if args.vbench_identity else None
    identity_factory = (lambda kind: build_identity_encoder(
        subject_kind=kind, device=args.clip_device)) if identity_probe is not None else None
    backbones = {
        "subject_dino": {"requested": bool(args.vbench_subject), "loaded": bool(dino)},
        "background_clip": {"requested": bool(args.vbench_background), "loaded": bool(clip)},
        "identity": {"requested": bool(args.vbench_identity), "loaded": bool(identity_probe)},
        "subject_kind": args.subject_kind,
    }

    finalist_records = []
    for key, value in finalists:
        settings = {
            "scene_memory_enabled": bool(args.modified_scene_memory_enabled),
            "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
            "scene_score_bonus": float(args.modified_scene_score_bonus),
            "retrieval_key_mode": key, "retrieval_value_mode": value,
        }
        merged = _finalist_kv_rag(base_kv_rag, settings)
        name = f"mod_{key}_{value}"
        mod_cfg, mod_dir = _write_one_variant(
            cfg, output_root, name, kv_rag_settings=merged, multiview_per_perspective=True
        )
        print(f"[finalist-gate] rendering {name}: {settings}")
        run_inference(mod_cfg)
        result = compare_multiview_vbench_dirs(
            baseline_dir, mod_dir, dino_encoder=dino, clip_encoder=clip,
            identity_encoder_for=identity_factory, subject_kind_default=args.subject_kind,
            adherence_scorer=adherence_scorer,
            captions_for=captions_for, max_frames=args.max_frames, stride=max(1, args.stride),
        )
        gate = evaluate_multiview_vbench_gate(
            result, min_scene_wins=args.min_scene_wins,
            adherence_tolerance=args.adherence_tolerance,
            diversity_tolerance=args.diversity_tolerance,
            motion_tolerance=args.motion_tolerance, require_adherence=require_adherence,
        )
        deltas = [r["modified_metrics"].get("aggregate_consistency", float("nan"))
                  - r["baseline_metrics"].get("aggregate_consistency", float("nan"))
                  for r in result["records"]]
        mean_delta = float(np.nanmean(deltas)) if deltas else float("nan")
        finalist_records.append({
            "key": key, "value": value, "settings": settings,
            "passed": gate["passed"], "scene_wins": gate["scene_wins"],
            "num_scenes": gate["num_scenes"], "mean_aggregate_delta": mean_delta,
            "gate": gate, "comparison": result,
        })

    # Rank + decide winner/null, FAIL-CLOSED if a requested backbone is missing OR
    # the adherence guard was skipped (dry run is not a real pass).
    missing_requested = _missing_requested_backbones(backbones)
    ranked, winner, blocked_reason = _finalize_finalist_ranking(
        finalist_records, missing_requested, gate_evaluated=require_adherence)
    consolidated = {
        "is_prefilter": False,
        "selector": "rendered AC-2 suite",
        "winner": ({"key": winner["key"], "value": winner["value"]} if winner else None),
        "is_null_result": winner is None,
        "missing_requested_backbones": missing_requested,
        "blocked_reason": blocked_reason,
        "ranking": [{"key": r["key"], "value": r["value"], "passed": r["passed"],
                     "mean_aggregate_delta": r["mean_aggregate_delta"],
                     "scene_wins": f"{r['scene_wins']}/{r['num_scenes']}"} for r in ranked],
        "perspective_coverage": coverage,
        "backbones": backbones,
        "guards": {"adherence_tolerance": args.adherence_tolerance,
                   "diversity_tolerance": args.diversity_tolerance,
                   "motion_tolerance": args.motion_tolerance,
                   "require_adherence": require_adherence},
        "finalists": finalist_records,
        "offline_screen": run_screen(),
    }
    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "multiview_finalist_gate.json"
    save_metrics_json(consolidated, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    if blocked_reason:
        print(f"[finalist-gate] BLOCKED (fail-closed): {blocked_reason}")
    print("[finalist-gate] RANKING (rendered AC-2 aggregate):")
    for r in ranked:
        print(f"  {r['key']}+{r['value']}: mean_agg_delta={r['mean_aggregate_delta']:+.4f} "
              f"wins={r['scene_wins']}/{r['num_scenes']} passed={r['passed']}")
    if winner:
        print(f"[finalist-gate] WINNER: {winner['key']}+{winner['value']}")
    elif blocked_reason:
        print("[finalist-gate] RESULT: NULL (blocked) -- requested AC-2 backbone(s) "
              "unavailable; no winner on a reduced suite.")
    else:
        print("[finalist-gate] RESULT: honest NULL -- no finalist passed all guards.")


def _run_multiview_vbench(args, output_root: Path) -> None:
    """AC-3/AC-4 driver: render ONE video per perspective for baseline + modified,
    score the AC-2 cross-video VBench suite, and decide the gate. The offline
    GPU-free screen ranking is recorded alongside as a labeled pre-filter only."""
    from evaluation.vbench_consistency import (
        build_clip_image_encoder,
        build_dino_encoder,
        build_identity_encoder,
        compare_multiview_vbench_dirs,
        evaluate_multiview_vbench_gate,
    )
    from evaluation.multiview_prompts import load_shot_specs
    from evaluation.retrieval_screen import run_screen

    if args.finalists:
        return _run_multiview_finalists(args, output_root)

    require_adherence = not args.dry_run_without_adherence
    modified_settings = _modified_kv_rag_from_args(args)
    coverage: dict | None = None

    if require_adherence and not args.prompts_dir:
        raise ValueError(
            "The multiview_vbench gate requires --prompts_dir (per-perspective "
            "captions + the CLIP adherence guard). Use --dry_run_without_adherence "
            "for a consistency-only non-gate run."
        )
    if require_adherence and args.motion_tolerance is None:
        # AC-4 forbids a consistency gain bought by a motion collapse, so a real
        # gate MUST enforce the dynamic_degree non-regression guard with an
        # explicit, recorded tolerance (no silent off-by-default for a pass).
        raise ValueError(
            "The multiview_vbench gate requires --motion_tolerance (AC-4 motion "
            "non-regression guard); pass e.g. --motion_tolerance 0.3. Use "
            "--dry_run_without_adherence for a non-gate exploratory run."
        )

    if args.skip_generation:
        if not args.baseline_dir or not args.kv_rag_dir:
            raise ValueError("--skip_generation requires --baseline_dir and --kv_rag_dir")
        baseline_dir = Path(args.baseline_dir)
        rag_dir = Path(args.kv_rag_dir)
    else:
        if not args.prompts_dir:
            raise ValueError("multiview_vbench requires --prompts_dir for the perspective set")
        cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
        subset = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else None
        chosen, coverage = build_multiview_subset(
            args.prompts_dir, subset, output_root / "prompt_subset",
            max_perspectives=args.max_perspectives,
        )
        _set_nested(cfg, "data", "data_path", str(output_root / "prompt_subset"))
        print(f"[gate] rendering per-perspective subset: {chosen}")
        print(f"[gate] perspective coverage (AC-7 -- logged, not capped silently): {coverage}")
        print(f"[gate] modified variant: {modified_settings}")
        _preflight_config(cfg, args.config_path)
        baseline_cfg, rag_cfg, baseline_dir, rag_dir = write_variant_configs(
            cfg, output_root,
            filename_from_sample_name=True,
            multiview_per_perspective=True,
            modified_kv_rag_extra=modified_settings,
        )
        run_inference(baseline_cfg)
        run_inference(rag_cfg)

    captions_for = None
    adherence_scorer = None
    if args.prompts_dir:
        captions_for = {s: v["captions"] for s, v in load_shot_specs(args.prompts_dir).items()}
    if require_adherence:
        adherence_scorer = build_clip_adherence_scorer(
            model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.clip_device
        )

    # Guard optional backbone loading so a missing model/package records reduced
    # coverage (and fails the gate closed) instead of crashing AFTER the renders.
    dino = _try_build_backbone("subject(DINO)", lambda: build_dino_encoder(device=args.clip_device)) \
        if args.vbench_subject else None
    clip = _try_build_backbone("background(CLIP)", lambda: build_clip_image_encoder(device=args.clip_device)) \
        if args.vbench_background else None
    # Pre-check identity availability once; the encoder is built FRESH PER SCENE via
    # the factory (stateful 'auto' encoder must not be shared across scenes).
    identity_probe = _try_build_backbone("identity", lambda: build_identity_encoder(
        subject_kind=args.subject_kind, device=args.clip_device, eager_fallback=True)) \
        if args.vbench_identity else None
    identity_factory = (lambda kind: build_identity_encoder(
        subject_kind=kind, device=args.clip_device)) if identity_probe is not None else None
    backbones = {
        "subject_dino": {"requested": bool(args.vbench_subject), "loaded": bool(dino)},
        "background_clip": {"requested": bool(args.vbench_background), "loaded": bool(clip)},
        "identity": {"requested": bool(args.vbench_identity), "loaded": bool(identity_probe)},
        "subject_kind": args.subject_kind,
    }
    missing_backbones = _missing_requested_backbones(backbones)
    if not (args.vbench_subject or args.vbench_background or args.vbench_identity):
        print("[multiview_vbench] note: no semantic backbones (--vbench_subject/"
              "_background/_identity) -> GPU-free dims only (temporal_style, "
              "appearance_style, overall_consistency) + companions feed the aggregate.")

    result = compare_multiview_vbench_dirs(
        baseline_dir, rag_dir,
        dino_encoder=dino, clip_encoder=clip,
        identity_encoder_for=identity_factory, subject_kind_default=args.subject_kind,
        adherence_scorer=adherence_scorer, captions_for=captions_for,
        max_frames=args.max_frames, stride=max(1, args.stride),
    )
    gate = evaluate_multiview_vbench_gate(
        result,
        min_scene_wins=args.min_scene_wins,
        adherence_tolerance=args.adherence_tolerance,
        diversity_tolerance=args.diversity_tolerance,
        motion_tolerance=args.motion_tolerance,
        require_adherence=require_adherence,
    )
    gate["modified_settings"] = modified_settings
    gate["perspective_coverage"] = coverage
    gate["backbones"] = backbones
    gate["missing_requested_backbones"] = missing_backbones
    if missing_backbones:
        # A requested AC-2 backbone could not load -> fail closed with a recorded
        # result (reduced-suite pass would be misleading), never a post-render crash.
        gate["blocked_reason"] = (
            f"requested AC-2 backbone(s) unavailable: {missing_backbones}; "
            "gate reports a reduced/null result (fail-closed)."
        )
        gate["passed"] = False
        gate["is_null_result"] = True
    if not require_adherence:
        gate["note"] = "adherence guard NOT evaluated (--dry_run_without_adherence); not a pass."
        gate["passed"] = False

    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "multiview_vbench_gate.json"
    save_metrics_json(
        {"gate": gate, "comparison": result,
         "offline_screen": run_screen()},
        metrics_json,
    )
    print(f"Wrote metrics: {metrics_json.resolve()}")
    print(f"Scenes compared: {result['num_scenes']}")
    print(f"[gate] aggregate wins: {gate['scene_wins']}/{gate['num_scenes']} "
          f"(need >= {gate['min_scene_wins']})")
    if require_adherence:
        print(f"[gate] adherence guard: {'OK' if gate['adherence_ok'] else 'FAIL'} "
              f"(failures={gate['adherence_failures']})")
    print(f"[gate] diversity guard: {'OK' if gate['diversity_ok'] else 'FAIL'} "
          f"(failures={gate['diversity_failures']})")
    for p in gate["per_scene"]:
        print(f"  {p['scene']}: {p['metric']} {p['baseline']:.4f} -> {p['modified']:.4f} "
              f"({'win' if p['win'] else 'no win'})")
    if not require_adherence:
        print("[gate] RESULT: DRY RUN -- adherence not evaluated; not a pass.")
        return
    print(f"[gate] RESULT: {'PASS' if gate['passed'] else 'NULL/FAIL (honest result recorded)'}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.mode in {"long_multishot", "cross_perspective"}:
        _run_cross_perspective(args, output_root)
    elif args.mode == "multiview_vbench":
        _run_multiview_vbench(args, output_root)
    else:
        _run_temporal(args, output_root)


if __name__ == "__main__":
    main()
