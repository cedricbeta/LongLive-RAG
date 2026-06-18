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
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.multiview_prompts import build_spec_resolver, clamp_chunk_durations, load_shot_specs
from evaluation.video_consistency import (
    build_clip_text_encoder,
    build_clip_adherence_scorer,
    build_raft_dynamic_scorer,
    compare_cross_perspective_dirs,
    compare_video_dirs,
    evaluate_cross_perspective_gate,
    prompt_text_similarity_lint,
    save_metrics_json,
    summarize_records,
)
from utils.kv_rag import KEY_MODES, VALUE_MODES

# The recommended long-video multi-shot settings for the modified variant
# (content-robust key + persistent scene anchors) are the argparse defaults of
# the --modified_* flags. The older cross_perspective name is kept as a
# backward-compatible alias for the same concatenated multi-shot path.


DEFAULT_KV_RAG = {
    "enabled": True,
    "top_k": 2,
    "max_entries": 32,
    "max_frames_per_entry": 1,
    # Legacy alias kept for configs/tests that still merge token caps.
    "max_tokens_per_entry": 1024,
    "layers": [0, 7, 14, 21, 29],
    "min_frame_gap": 0,
    "retrieve_during_denoise": True,
    "retrieve_during_recache": False,
    "store_after_recache": True,
    "token_policy": "uniform",
    "similarity": "cosine",
    "verbose": True,
    "frame_aligned_store": True,
    "require_frame_aligned": True,
    "reinject_rope": True,
    "persistent_logit_bias_lambda": 0.0,
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
    parser.add_argument("--mechanism_sweep", action="store_true",
                        help="long_multishot: run the fixed scene-memory structure/dose sweep "
                        "with baseline seed replicates and noise-floored wins.")
    parser.add_argument("--strategy_stage", choices=("A", "B1", "B2", "verdict", "lever"), default=None,
                        help="long_multishot: run the staged frame-level key/value protocol. "
                        "A renders/evaluates baseline seeds and writes drift-audit JSON; "
                        "B1 screens keys with raw values; B2 screens top keys x values; "
                        "verdict evaluates top combos across baseline seeds; "
                        "lever reruns the best verdict combo with persistent logit bias.")
    parser.add_argument("--previous_stage_json", default=None,
                        help="strategy_stage B1/B2/verdict: prior stage JSON carrying "
                        "admitted scenes, noise floor, and baseline seed dirs.")
    parser.add_argument("--top_keys", default=None,
                        help="strategy_stage B2: comma-separated key modes. If omitted, "
                        "read top keys from --previous_stage_json ranking.")
    parser.add_argument("--top_combos", default=None,
                        help="strategy_stage verdict: comma-separated key:value combos. "
                        "If omitted, read top combos from --previous_stage_json ranking.")
    parser.add_argument("--logit_bias_lambdas", default="1,2",
                        help="strategy_stage lever: comma-separated persistent frame-column "
                        "logit-bias lambdas for the best prior combo.")
    parser.add_argument("--max_render_stems_per_invocation", type=int, default=0,
                        help="strategy_stage resume helper: render at most this many missing "
                        "scene stems before returning a fail-closed incomplete JSON. "
                        "Default 0 renders all missing stems.")
    parser.add_argument("--blocks_per_shot", type=int, default=6,
                        help="long_multishot mechanism_sweep: generated sparse prompt shot duration "
                        "in latent blocks when a balanced total-block split is impossible. "
                        "Gate admission requires final durations in [6, 12].")
    parser.add_argument("--baseline_seeds", default="0,1,2",
                        help="long_multishot mechanism_sweep: comma-separated baseline seeds "
                        "used to estimate per-scene centroid sigma.")
    parser.add_argument("--baseline_drift_floor", type=float, default=0.02,
                        help="long_multishot mechanism_sweep: reject main scenes whose "
                        "3-seed baseline drift (1 - baseline metric mean) is below this floor.")
    parser.add_argument("--admission_diversity_floor", type=float, default=0.03,
                        help="long_multishot mechanism_sweep: rendered baseline "
                        "inter-shot composition-diversity floor for scene admission.")
    parser.add_argument("--noise_sigma_multiplier", type=float, default=2.0,
                        help="long_multishot mechanism_sweep: consistency win threshold is "
                        "delta > multiplier * baseline-seed sigma.")
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
    if args.strategy_stage and args.mode == "long_multishot":
        return _run_long_multishot_strategy_stage(args, output_root)
    if args.mechanism_sweep and args.mode == "long_multishot":
        return _run_long_multishot_mechanism_sweep(args, output_root)
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
                       seed: int | None = None,
                       multiview_per_perspective: bool = False,
                       config_name: str | None = None) -> tuple[Path, Path]:
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
    if seed is not None:
        _set_nested(variant, "logging", "seed", int(seed))
    if multiview_per_perspective:
        variant.inference.multiview_per_perspective = True
        variant.multiview_per_perspective = True
    cfg_path = config_dir / f"{config_name or name}.yaml"
    OmegaConf.save(variant, cfg_path)
    return cfg_path, out_dir


def _single_stem_prompt_subset(prompt_subset_dir: Path, stem: str, dest_root: Path) -> Path:
    """Create a prompt subset containing one scene, preserving the scene folder."""
    src = prompt_subset_dir / stem
    if not src.is_dir():
        raise FileNotFoundError(f"prompt subset scene missing: {src}")
    dest = dest_root / stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest / stem)
    return dest


def _render_missing_variant_stems(
    cfg,
    output_root: Path,
    *,
    prompt_subset_dir: Path,
    variant_name: str,
    variant_dir: Path,
    kv_rag_settings: dict,
    seed: int,
    stems: list[str],
    status: dict | None = None,
    max_stems: int | None = None,
) -> tuple[dict, bool]:
    """Render only missing scene videos for a variant and return the final status."""
    current = status or _variant_output_status(variant_dir, variant_name, stems)
    rendered = False
    rendered_count = 0
    single_root = output_root / "prompt_subset_single" / variant_name
    while not current["complete"]:
        missing_stems = list(current["missing_stems"])
        progress = False
        for stem in missing_stems:
            if max_stems is not None and rendered_count >= max_stems:
                return current, rendered
            stem_subset = _single_stem_prompt_subset(prompt_subset_dir, stem, single_root)
            stem_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            _set_nested(stem_cfg, "data", "data_path", str(stem_subset))
            cfg_path, _ = _write_one_variant(
                stem_cfg,
                output_root,
                variant_name,
                kv_rag_settings=kv_rag_settings,
                seed=seed,
                multiview_per_perspective=False,
                config_name=f"{variant_name}__{stem}",
            )
            print(f"[strategy] rendering {variant_name} stem={stem}")
            run_inference(cfg_path)
            rendered = True
            rendered_count += 1
            next_status = _variant_output_status(variant_dir, variant_name, stems)
            if next_status["per_stem"].get(stem, {}).get("complete"):
                progress = True
            current = next_status
        if not progress:
            break
    return current, rendered


def _variant_output_status(output_dir: Path, variant_name: str, stems: list[str]) -> dict:
    """Report whether a rendered variant already has one non-empty video per stem."""
    per_stem: dict[str, dict] = {}
    missing: list[str] = []
    for stem in stems:
        matches = sorted(output_dir.glob(f"{variant_name}-rank*-{stem}-*_regular.mp4"))
        usable = [path for path in matches if path.exists() and path.stat().st_size > 0]
        per_stem[stem] = {
            "complete": bool(usable),
            "paths": [str(path) for path in usable],
        }
        if not usable:
            missing.append(stem)
    return {
        "complete": not missing,
        "missing_stems": missing,
        "per_stem": per_stem,
    }


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


def _parse_seed_list(spec: str) -> list[int]:
    seeds = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if tok:
            seeds.append(int(tok))
    if len(seeds) < 2:
        raise ValueError("--baseline_seeds needs at least two seeds to estimate sigma")
    return seeds


def _mechanism_sweep_arms(args) -> list[dict]:
    """Fixed AC mechanism grid, all subject_identity+raw."""
    base = {
        "scene_memory_enabled": True,
        "scene_score_bonus": float(args.modified_scene_score_bonus),
        "retrieval_key_mode": "subject_identity",
        "retrieval_value_mode": "raw",
        "attention_diagnostic": True,
    }
    return [
        {
            "name": "A_incumbent_shot0_boundary",
            "description": "shot0 seed + boundary pulse",
            "settings": {
                **base,
                "scene_memory_rolling": False,
                "scene_memory_injection_schedule": "boundary",
                "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
            },
        },
        {
            "name": "B_rolling_boundary",
            "description": "rolling completed-shot memory + boundary pulse",
            "settings": {
                **base,
                "scene_memory_rolling": True,
                "scene_memory_injection_schedule": "boundary",
                "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
            },
        },
        {
            "name": "C_rolling_every_chunk",
            "description": "rolling completed-shot memory + every-chunk pulse",
            "settings": {
                **base,
                "scene_memory_rolling": True,
                "scene_memory_injection_schedule": "every_chunk",
                "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
            },
        },
        {
            "name": "D_rolling_every_chunk_anchors8",
            "description": "rolling completed-shot memory + every-chunk pulse + 8 anchors",
            "settings": {
                **base,
                "scene_memory_rolling": True,
                "scene_memory_injection_schedule": "every_chunk",
                "boundary_inject_anchors": 8,
            },
        },
        {
            "name": "E_shot0_every_chunk",
            "description": "shot0 seed + every-chunk pulse",
            "settings": {
                **base,
                "scene_memory_rolling": False,
                "scene_memory_injection_schedule": "every_chunk",
                "boundary_inject_anchors": int(args.modified_boundary_inject_anchors),
            },
        },
    ]


def _mechanism_prompt_roots(prompts_dir: str) -> list[Path]:
    roots = []
    for part in str(prompts_dir).split(":"):
        part = part.strip()
        if not part:
            continue
        root = Path(part)
        roots.append(root / "caption" if (root / "caption").is_dir() else root)
    if not roots:
        raise ValueError("mechanism_sweep requires at least one prompt source directory")
    return roots


def _balanced_shot_durations(
    num_shots: int,
    total_blocks: int | None,
    *,
    fallback_blocks_per_shot: int,
    min_blocks: int = 6,
    max_blocks: int = 12,
) -> list[int]:
    """Deterministically split a render budget into 6-12 block shots."""
    n = int(num_shots)
    if n <= 0:
        return []
    if total_blocks is None:
        return [int(fallback_blocks_per_shot)] * n
    total = int(total_blocks)
    if n * min_blocks <= total <= n * max_blocks:
        durations = [min_blocks] * n
        remaining = total - n * min_blocks
        idx = 0
        while remaining > 0:
            add = min(max_blocks - durations[idx], remaining)
            durations[idx] += add
            remaining -= add
            idx = (idx + 1) % n
        return durations
    return [int(fallback_blocks_per_shot)] * n


def _local_attn_size_from_cfg(cfg) -> int | None:
    value = _value(cfg, "inference", "local_attn_size")
    if value is None:
        value = _value(cfg, "model_kwargs", "local_attn_size")
    return None if value is None else int(value)


def _num_frame_per_block_from_cfg(cfg) -> int:
    value = _value(cfg, "model_kwargs", "num_frame_per_block")
    if value is None:
        value = getattr(cfg, "num_frame_per_block", 8)
    return int(value)


def _num_output_frames_from_cfg(cfg) -> int | None:
    nof = getattr(cfg, "num_output_frames", None)
    if nof is None:
        shape = _value(cfg, "data", "image_or_video_shape")
        if shape is not None and len(shape) > 1:
            nof = shape[1]
    return None if nof is None else int(nof)


def _shot_pair_separation_report(
    durations: list[int],
    *,
    frames_per_block: int,
    local_attn_size: int,
) -> dict:
    starts = []
    cursor = 0
    for d in durations:
        starts.append(cursor)
        cursor += int(d) * int(frames_per_block)
    total_pairs = 0
    separated_pairs = 0
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            total_pairs += 1
            if abs(starts[j] - starts[i]) > int(local_attn_size):
                separated_pairs += 1
    required = -(-total_pairs // 2)
    return {
        "total_pairs": total_pairs,
        "separated_pairs": separated_pairs,
        "required_pairs": required,
        "passed": bool(total_pairs > 0 and separated_pairs >= required),
    }


def _long_regime_report(cfg, prompt_subset_dir: Path, *, num_blocks: int | None) -> dict:
    specs = load_shot_specs(prompt_subset_dir)
    local_attn_size = _local_attn_size_from_cfg(cfg)
    num_output_frames = _num_output_frames_from_cfg(cfg)
    frames_per_block = _num_frame_per_block_from_cfg(cfg)
    planned_blocks = int(num_blocks) if num_blocks is not None else None
    blocked: list[str] = []
    if local_attn_size is None or local_attn_size <= 0:
        blocked.append("local_attn_size unavailable or non-positive")
    if num_output_frames is None:
        blocked.append("num_output_frames unavailable")
    else:
        if num_output_frames < 480:
            blocked.append(f"planned frames {num_output_frames} < 480")
        if local_attn_size and num_output_frames <= 3 * local_attn_size:
            blocked.append(
                f"planned frames {num_output_frames} <= 3*local_attn_size ({3 * local_attn_size})"
            )
    scene_reports = {}
    if planned_blocks is not None:
        for scene, spec in specs.items():
            durations = clamp_chunk_durations(spec.get("chunk_durations", []), planned_blocks)
            in_range = all(6 <= int(d) <= 12 for d in durations)
            sep = _shot_pair_separation_report(
                durations,
                frames_per_block=frames_per_block,
                local_attn_size=int(local_attn_size or 0),
            )
            total_duration = sum(int(d) for d in durations)
            scene_blocked = []
            if total_duration != planned_blocks:
                scene_blocked.append(
                    f"shot_durations sum {total_duration} != planned render blocks {planned_blocks}"
                )
            if not in_range:
                scene_blocked.append(f"shot_durations outside 6-12 blocks/shot: {durations}")
            if not sep["passed"]:
                scene_blocked.append(
                    "fewer than half of shot pairs are separated beyond local_attn_size "
                    f"({sep['separated_pairs']}/{sep['total_pairs']}, need {sep['required_pairs']})"
                )
            scene_reports[scene] = {
                "shot_durations": durations,
                "num_shots": len(durations),
                "duration_blocks": total_duration,
                "duration_frames": total_duration * frames_per_block,
                "durations_in_6_12_blocks": bool(in_range),
                "shot_pair_separation": sep,
                "blocked_reason": "; ".join(scene_blocked) if scene_blocked else None,
                "passed": not scene_blocked,
            }
        bad = [scene for scene, rec in scene_reports.items() if not rec["passed"]]
        if bad:
            blocked.append(f"scene regime proof failed: {bad}")
    return {
        "planned_frames": num_output_frames,
        "planned_blocks": planned_blocks,
        "frames_per_block": frames_per_block,
        "local_attn_size": local_attn_size,
        "scene_regime": scene_reports,
        "passed": not blocked,
        "blocked_reason": "; ".join(blocked) if blocked else None,
    }


def _frame_contract_stats(diag: dict) -> dict:
    stats = {
        "frame_alignment_store_drops": 0,
        "frame_alignment_inject_drops": 0,
        "outside_window_injection_calls": 0,
        "outside_window_injected_frames": 0,
        "injected_frames": 0,
        "direct_injection_calls": 0,
        "direct_injected_frames": 0,
        "reinject_rope_injection_calls": 0,
        "reinject_rope_injected_frames": 0,
        "persistent_logit_bias_calls": 0,
        "persistent_logit_bias_tokens": 0,
        "persistent_logit_bias_frames": 0,
        "persistent_logit_bias_skipped_unaligned": 0,
    }
    for bank in ("pos", "neg"):
        bank_stats = (diag.get(bank) or {}).get("stats", {})
        for key in stats:
            stats[key] += int(bank_stats.get(key, 0) or 0)
    return stats


def _frame_contract_blocked_reason(attention_by_scene: dict[str, dict]) -> str | None:
    drops: list[str] = []
    zero_injection: list[str] = []
    for scene, rec in attention_by_scene.items():
        diag = rec.get("diagnostic", {}) if isinstance(rec, dict) else {}
        stats = _frame_contract_stats(diag)
        if stats["frame_alignment_store_drops"] or stats["frame_alignment_inject_drops"]:
            drops.append(
                f"{scene}: store_drops={stats['frame_alignment_store_drops']} "
                f"inject_drops={stats['frame_alignment_inject_drops']}"
            )
        if stats["outside_window_injected_frames"] <= 0:
            zero_injection.append(scene)
    reasons = []
    if drops:
        reasons.append(f"frame-contract drop(s): {drops}")
    if zero_injection:
        reasons.append(f"zero outside-window frame injections: {zero_injection}")
    return "; ".join(reasons) if reasons else None


def _strategy_blocked_json(
    *,
    stage: str,
    blocked_reason: str,
    render_attempted: bool,
    chosen: list[str] | None = None,
    prompt_subset_dir: Path | None = None,
    regime_proof: dict | None = None,
    prompt_lint: dict | None = None,
    lint_passing_main: list[str] | None = None,
    admitted_main: list[str] | None = None,
    negative_controls: list[str] | None = None,
    scorer_coverage: dict | None = None,
    missing_invariant: list[str] | None = None,
    reauthor_scenes: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    out = {
        "selector": "long_multishot frame-level KV-RAG strategy",
        "strategy_stage": stage,
        "winner": None,
        "is_null_result": True,
        "blocked_reason": blocked_reason,
        "render_attempted": bool(render_attempted),
        "chosen_scenes": chosen or [],
        "prompt_subset_dir": str(prompt_subset_dir) if prompt_subset_dir else None,
        "metric": "anchor_drift_aggregate_consistency",
        "regime_proof": regime_proof or {},
        "prompt_lint": prompt_lint or {},
        "lint_passing_main_scenes": lint_passing_main or [],
        "admitted_main_scenes": admitted_main or [],
        "negative_controls": negative_controls or [],
        "missing_invariant_contrast_scenes": missing_invariant or [],
        "reauthor_scenes": reauthor_scenes or [],
        "scorer_coverage": scorer_coverage or {},
    }
    if extra:
        out.update(extra)
    return out


def _strategy_candidates_from_previous(
    previous: dict,
    *,
    limit: int,
    keys_only: bool = False,
) -> list[str] | list[tuple[str, str]]:
    ranking = previous.get("ranking", [])
    if keys_only:
        keys: list[str] = []
        for row in ranking:
            key = row.get("key")
            if key and key not in keys:
                keys.append(str(key))
            if len(keys) >= limit:
                break
        return keys
    combos: list[tuple[str, str]] = []
    for row in ranking:
        key = row.get("key")
        value = row.get("value")
        if key and value:
            combo = (str(key), str(value))
            if combo not in combos:
                combos.append(combo)
        if len(combos) >= limit:
            break
    return combos


def _strategy_stage_candidates(args, previous: dict | None = None) -> list[tuple[str, str]]:
    stage = args.strategy_stage
    if stage == "A":
        return []
    if stage == "B1":
        # The semantic key is included: the pipeline wires it to the current
        # conditional caption embedding via _set_kv_rag_context_key.
        return [(key, "raw") for key in KEY_MODES]
    if stage == "B2":
        if args.top_keys:
            keys = [k.strip() for k in args.top_keys.split(",") if k.strip()]
        else:
            if previous is None:
                raise ValueError("strategy_stage B2 requires --top_keys or --previous_stage_json")
            keys = list(_strategy_candidates_from_previous(previous, limit=2, keys_only=True))
        if len(keys) < 2:
            raise ValueError(f"strategy_stage B2 needs two key modes, got {keys}")
        return [(key, value) for key in keys[:2] for value in VALUE_MODES]
    if stage == "verdict":
        if args.top_combos:
            return _parse_finalists(args.top_combos)
        if previous is None:
            raise ValueError("strategy_stage verdict requires --top_combos or --previous_stage_json")
        combos = list(_strategy_candidates_from_previous(previous, limit=2, keys_only=False))
        if len(combos) < 2:
            raise ValueError(f"strategy_stage verdict needs two key:value combos, got {combos}")
        return combos[:2]
    if stage == "lever":
        return []
    raise ValueError(f"Unsupported strategy_stage={stage!r}")


def _strategy_prepare_subset(args, output_root: Path, cfg) -> tuple[list[str], Path, int | None, dict]:
    subset = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else None
    prompt_subset_dir = output_root / "prompt_subset"
    num_blocks = args.num_blocks if args.num_blocks is not None else _num_blocks_from_cfg(cfg)
    chosen = build_mechanism_prompt_subset(
        args.prompts_dir,
        subset,
        prompt_subset_dir,
        blocks_per_shot=int(args.blocks_per_shot),
        total_blocks=num_blocks,
    )
    regime_proof = _long_regime_report(cfg, prompt_subset_dir, num_blocks=num_blocks)
    return chosen, prompt_subset_dir, num_blocks, regime_proof


def _strategy_stage_a(args, output_root: Path) -> dict:
    import numpy as np

    started = time.monotonic()
    metric = "anchor_drift_aggregate_consistency"
    seeds = _parse_seed_list(args.baseline_seeds)
    cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
    chosen, prompt_subset_dir, num_blocks, regime_proof = _strategy_prepare_subset(args, output_root, cfg)
    _set_nested(cfg, "data", "data_path", str(prompt_subset_dir))
    _set_nested(cfg, "inference", "sparse_long_multishot", True)

    if not regime_proof["passed"]:
        bad = [
            scene for scene, rec in regime_proof.get("scene_regime", {}).items()
            if rec.get("blocked_reason")
        ]
        return _strategy_blocked_json(
            stage="A",
            blocked_reason=regime_proof["blocked_reason"],
            render_attempted=False,
            chosen=chosen,
            prompt_subset_dir=prompt_subset_dir,
            regime_proof=regime_proof,
            reauthor_scenes=bad,
            extra={"wall_clock_sec": round(time.monotonic() - started, 3)},
        )

    prompt_lint, _allowed = _long_prompt_lint(
        argparse.Namespace(**{**vars(args), "prompts_dir": str(prompt_subset_dir)})
    )
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
    prompt_specs = load_shot_specs(prompt_subset_dir)
    missing_invariant = [
        scene for scene in lint_passing_main
        if not prompt_specs.get(scene, {}).get("invariant_caption")
        or not prompt_specs.get(scene, {}).get("contrast_caption")
    ]
    scorers, scorer_coverage = _build_long_scorers(args, require_adherence=True)
    missing_scorers = sorted(
        name for name, info in scorer_coverage.items()
        if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
    )
    prereq_blocks: list[str] = []
    if seeds != [0, 1, 2]:
        prereq_blocks.append(f"baseline_seeds must be exactly 0,1,2, got {seeds}")
    if lint_failures:
        prereq_blocks.append(f"ill-posed prompt scene(s) below text-similarity floor: {lint_failures}")
    if len(lint_passing_main) < 3:
        prereq_blocks.append(f"need >=3 lint-passing non-control scenes, got {len(lint_passing_main)}")
    if not negative_controls:
        prereq_blocks.append("negative control scene missing")
    if missing_invariant:
        prereq_blocks.append(f"missing invariant/contrast captions: {missing_invariant}")
    if missing_scorers:
        prereq_blocks.append(f"requested scorer(s) unavailable: {missing_scorers}")
    if prereq_blocks:
        reauthor = sorted(set(lint_failures + missing_invariant))
        return _strategy_blocked_json(
            stage="A",
            blocked_reason="; ".join(prereq_blocks),
            render_attempted=False,
            chosen=chosen,
            prompt_subset_dir=prompt_subset_dir,
            regime_proof=regime_proof,
            prompt_lint=chosen_lint,
            lint_passing_main=lint_passing_main,
            negative_controls=negative_controls,
            scorer_coverage=scorer_coverage,
            missing_invariant=missing_invariant,
            reauthor_scenes=reauthor,
            extra={"wall_clock_sec": round(time.monotonic() - started, 3)},
        )

    _preflight_config(cfg, args.config_path)
    shots_for = build_spec_resolver(prompt_subset_dir, with_captions=True, max_chunks=num_blocks)
    baseline_dirs: dict[int, Path] = {}
    baseline_seed_results: dict[int, dict] = {}
    baseline_render_status: dict[int, dict] = {}
    for seed in seeds:
        variant_name = f"baseline_seed{seed}"
        baseline_cfg, baseline_dir = _write_one_variant(
            cfg,
            output_root,
            variant_name,
            kv_rag_settings={"enabled": False},
            seed=seed,
            multiview_per_perspective=False,
        )
        baseline_dirs[seed] = baseline_dir
        status = _variant_output_status(baseline_dir, variant_name, chosen)
        if status["complete"]:
            print(f"[strategy-A] reusing complete baseline seed {seed}: {baseline_dir}")
            baseline_render_status[seed] = {**status, "rendered": False, "reused": True}
        else:
            print(
                f"[strategy-A] rendering baseline seed {seed}; "
                f"missing stems={status['missing_stems']}"
            )
            status, rendered = _render_missing_variant_stems(
                cfg,
                output_root,
                prompt_subset_dir=prompt_subset_dir,
                variant_name=variant_name,
                variant_dir=baseline_dir,
                kv_rag_settings={"enabled": False},
                seed=seed,
                stems=chosen,
                status=status,
                max_stems=(
                    int(args.max_render_stems_per_invocation)
                    if int(args.max_render_stems_per_invocation) > 0 else None
                ),
            )
            baseline_render_status[seed] = {
                **status,
                "rendered": bool(rendered),
                "reused": False,
                "resume_mode": "per_stem",
                "seed_config": str(baseline_cfg),
            }
        if not status["complete"]:
            return _strategy_blocked_json(
                stage="A",
                blocked_reason=(
                    f"render incomplete for baseline seed {seed}: "
                    f"missing stems={status['missing_stems']}"
                ),
                render_attempted=True,
                chosen=chosen,
                prompt_subset_dir=prompt_subset_dir,
                regime_proof=regime_proof,
                prompt_lint=chosen_lint,
                lint_passing_main=lint_passing_main,
                negative_controls=negative_controls,
                scorer_coverage=scorer_coverage,
                reauthor_scenes=[],
                extra={
                    "baseline_seed_dirs": {str(k): str(v) for k, v in baseline_dirs.items()},
                    "baseline_render_status": {
                        str(k): v for k, v in baseline_render_status.items()
                    },
                    "wall_clock_sec": round(time.monotonic() - started, 3),
                },
            )

    for seed in seeds:
        print(f"[strategy-A] evaluating baseline seed {seed}: {baseline_dirs[seed]}")
        baseline_seed_results[seed] = _evaluate_single_long_dir(
            baseline_dirs[seed], shots_for=shots_for, scorers=scorers, args=args
        )

    baseline_by_seed = {
        seed: _scene_metric_records_by_theme(result, metric)
        for seed, result in baseline_seed_results.items()
    }
    seed0 = seeds[0]
    seed0_records = baseline_by_seed[seed0]
    noise_floor_by_scene: dict[str, dict] = {}
    admission: dict[str, dict] = {}
    admitted_main: list[str] = []
    for scene in chosen:
        vals = [
            baseline_by_seed.get(seed, {}).get(scene, {}).get("metric", float("nan"))
            for seed in seeds
        ]
        sigma = _metric_sigma(vals)
        threshold = float(args.noise_sigma_multiplier) * sigma if not np.isnan(sigma) else float("nan")
        finite_vals = [float(v) for v in vals if not np.isnan(v)]
        baseline_drift = float(np.mean([1.0 - v for v in finite_vals])) if finite_vals else float("nan")
        seed0_metrics = seed0_records.get(scene, {}).get("metrics", {})
        diversity = seed0_metrics.get("inter_shot_composition_diversity", float("nan"))
        is_negative = scene in negative_controls
        admitted = (
            not is_negative
            and scene in lint_passing_main
            and diversity == diversity
            and diversity >= float(args.admission_diversity_floor)
            and baseline_drift == baseline_drift
            and baseline_drift >= float(args.baseline_drift_floor)
        )
        reason = None
        if is_negative:
            reason = "negative control is sanity-checked, not admitted as a main scene"
        elif not (diversity == diversity):
            reason = "baseline inter-shot composition diversity unscorable"
        elif diversity < float(args.admission_diversity_floor):
            reason = (
                "baseline inter-shot composition diversity below admission floor "
                f"({diversity:.6f} < {float(args.admission_diversity_floor):.6f})"
            )
        elif not (baseline_drift == baseline_drift):
            reason = f"baseline {metric} drift unscorable"
        elif baseline_drift < float(args.baseline_drift_floor):
            reason = (
                f"baseline drift below headroom floor ({baseline_drift:.6f} "
                f"< {float(args.baseline_drift_floor):.6f})"
            )
        admission[scene] = {
            "admitted": bool(admitted),
            "negative_control": bool(is_negative),
            "baseline_seed": seed0,
            "baseline_inter_shot_composition_diversity": diversity,
            "diversity_floor": float(args.admission_diversity_floor),
            "baseline_drift": baseline_drift,
            "baseline_drift_floor": float(args.baseline_drift_floor),
            "blocked_reason": reason,
        }
        noise_floor_by_scene[scene] = {
            "baseline_values": [float(v) for v in vals],
            "sigma": sigma,
            "threshold": threshold,
            "sigma_multiplier": float(args.noise_sigma_multiplier),
            "baseline_drift": baseline_drift,
        }
        if admitted:
            admitted_main.append(scene)

    blocked_reason = None
    if len(admitted_main) < 3:
        blocked_reason = f"need >=3 admitted main scenes after drift audit, got {len(admitted_main)}"
    if not negative_controls:
        blocked_reason = "; ".join([r for r in (blocked_reason, "negative control scene missing") if r])
    reauthor_scenes = [
        scene for scene, rec in admission.items()
        if not rec.get("admitted") and not rec.get("negative_control")
    ]
    return {
        "selector": "long_multishot frame-level KV-RAG strategy",
        "strategy_stage": "A",
        "winner": None,
        "is_null_result": bool(blocked_reason),
        "blocked_reason": blocked_reason,
        "render_attempted": True,
        "chosen_scenes": chosen,
        "prompt_subset_dir": str(prompt_subset_dir),
        "metric": metric,
        "regime_proof": regime_proof,
        "baseline_seeds": seeds,
        "baseline_seed_dirs": {str(k): str(v) for k, v in baseline_dirs.items()},
        "baseline_render_status": {str(k): v for k, v in baseline_render_status.items()},
        "baseline_seed_results": baseline_seed_results,
        "prompt_lint": chosen_lint,
        "lint_passing_main_scenes": lint_passing_main,
        "admitted_main_scenes": admitted_main,
        "admission": admission,
        "negative_controls": negative_controls,
        "noise_floor": noise_floor_by_scene,
        "scorer_coverage": scorer_coverage,
        "guards": {
            "adherence_tolerance": args.adherence_tolerance,
            "diversity_tolerance_relative": args.diversity_tolerance,
            "motion_tolerance_relative": args.motion_tolerance,
            "invariant_tolerance": 0.0,
            "require_adherence": True,
            "admission_diversity_floor": float(args.admission_diversity_floor),
            "baseline_drift_floor": float(args.baseline_drift_floor),
            "noise_sigma_multiplier": float(args.noise_sigma_multiplier),
            "frame_contract_required": True,
        },
        "reauthor_scenes": reauthor_scenes if blocked_reason else [],
        "wall_clock_sec": round(time.monotonic() - started, 3),
    }


def _finite_mean(values) -> float:
    vals = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if f == f:
            vals.append(f)
    return sum(vals) / len(vals) if vals else float("nan")


def _rankable_float(value) -> float:
    try:
        f = float(value)
    except Exception:
        return float("-inf")
    return f if f == f else float("-inf")


def _strategy_settings_for_combo(
    args,
    key: str,
    value: str,
    *,
    persistent_logit_bias_lambda: float = 0.0,
) -> dict:
    settings = _modified_kv_rag_from_args(args)
    settings.update({
        "retrieval_key_mode": key,
        "retrieval_value_mode": value,
        "attention_diagnostic": True,
        "persistent_logit_bias_lambda": float(persistent_logit_bias_lambda),
    })
    return settings


def _kv_rag_contract_summary(kv_rag_settings: dict | None) -> dict:
    kv = dict(kv_rag_settings or {})
    max_frames = kv.get("max_frames_per_entry", None)
    out = {
        "enabled": bool(kv.get("enabled", False)),
        "max_frames_per_entry": max_frames,
        "legacy_max_tokens_per_entry": kv.get("max_tokens_per_entry", None),
        "frame_aligned_store": bool(kv.get("frame_aligned_store", False)),
        "require_frame_aligned": bool(kv.get("require_frame_aligned", False)),
        "reinject_rope": bool(kv.get("reinject_rope", False)),
        "retrieval_key_mode": kv.get("retrieval_key_mode"),
        "retrieval_value_mode": kv.get("retrieval_value_mode"),
        "persistent_logit_bias_lambda": float(kv.get("persistent_logit_bias_lambda", 0.0) or 0.0),
    }
    blocked: list[str] = []
    if not out["enabled"]:
        blocked.append("kv_rag disabled")
    if max_frames is None:
        blocked.append("max_frames_per_entry missing (would use legacy token cap)")
    elif int(max_frames) < 0:
        blocked.append(f"max_frames_per_entry negative: {max_frames}")
    if not out["frame_aligned_store"]:
        blocked.append("frame_aligned_store=false")
    if not out["require_frame_aligned"]:
        blocked.append("require_frame_aligned=false")
    if not out["reinject_rope"]:
        blocked.append("reinject_rope=false")
    out["frame_level_contract_ok"] = not blocked
    out["blocked_reason"] = "; ".join(blocked) if blocked else None
    return out


def _strategy_context_from_previous(
    previous: dict | None,
    *,
    stage: str,
) -> tuple[dict | None, str | None]:
    if previous is None:
        return None, f"strategy_stage {stage} requires --previous_stage_json"
    if previous.get("blocked_reason"):
        return None, f"previous stage blocked: {previous['blocked_reason']}"

    reasons: list[str] = []
    prompt_subset_value = previous.get("prompt_subset_dir")
    prompt_subset_dir = Path(prompt_subset_value) if prompt_subset_value else None
    if prompt_subset_dir is None or not prompt_subset_dir.exists():
        reasons.append(f"previous prompt_subset_dir missing or unavailable: {prompt_subset_value}")

    baseline_seed_dirs: dict[int, Path] = {}
    for seed, path in (previous.get("baseline_seed_dirs") or {}).items():
        try:
            baseline_seed_dirs[int(seed)] = Path(path)
        except Exception:
            reasons.append(f"invalid baseline seed dir entry: {seed}={path}")
    required_seeds = [0, 1, 2] if stage in {"verdict", "lever"} else [0]
    for seed in required_seeds:
        path = baseline_seed_dirs.get(seed)
        if path is None or not path.exists():
            reasons.append(f"baseline seed {seed} dir missing or unavailable: {path}")

    admitted_main = list(previous.get("admitted_main_scenes") or [])
    negative_controls = list(previous.get("negative_controls") or [])
    min_admitted = 3 if stage in {"verdict", "lever"} else 2
    if len(admitted_main) < min_admitted:
        reasons.append(
            f"need >={min_admitted} admitted main scenes for strategy_stage {stage}, "
            f"got {len(admitted_main)}"
        )
    if not negative_controls:
        reasons.append("negative control scene missing")

    noise_floor = previous.get("noise_floor") or {}
    allowed_eval_scenes = set(admitted_main) | set(negative_controls)
    missing_noise = [
        scene for scene in sorted(allowed_eval_scenes)
        if not isinstance(noise_floor.get(scene), dict)
        or "threshold" not in noise_floor.get(scene, {})
    ]
    if missing_noise:
        reasons.append(f"noise floor missing for scene(s): {missing_noise}")

    regime_proof = previous.get("regime_proof") or {}
    if regime_proof and not regime_proof.get("passed", True):
        reasons.append(f"previous regime proof failed: {regime_proof.get('blocked_reason')}")

    if reasons:
        return None, "; ".join(reasons)
    return {
        "prompt_subset_dir": prompt_subset_dir,
        "baseline_seed_dirs": baseline_seed_dirs,
        "baseline_seeds": list(previous.get("baseline_seeds") or [0, 1, 2]),
        "admitted_main_scenes": admitted_main,
        "negative_controls": negative_controls,
        "noise_floor": noise_floor,
        "regime_proof": regime_proof,
        "prompt_lint": previous.get("prompt_lint") or {},
        "lint_passing_main_scenes": previous.get("lint_passing_main_scenes") or admitted_main,
        "admission": previous.get("admission") or {},
        "guards": previous.get("guards") or {},
    }, None


def _strategy_stage_name(stage: str, key: str, value: str, seed: int) -> str:
    token = f"stage_{stage}_{key}_{value}_seed{seed}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in token)


def _strategy_arm_name(stage: str, key: str, value: str, seed: int, bias_lambda: float) -> str:
    base = _strategy_stage_name(stage, key, value, seed)
    if float(bias_lambda) <= 0.0:
        return base
    return f"{base}_bias{str(float(bias_lambda)).replace('.', 'p')}"


def _parse_float_list(spec: str) -> list[float]:
    out = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise ValueError("expected at least one numeric value")
    return out


def _parse_combo_list_allow_one(spec: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        key, _, value = tok.partition(":")
        out.append((key.strip(), value.strip() or "raw"))
    return out


def _strategy_lever_arms(args, previous: dict | None) -> list[dict]:
    if previous is None:
        raise ValueError("strategy_stage lever requires --previous_stage_json")
    combos = _parse_combo_list_allow_one(args.top_combos)[:1] if args.top_combos else []
    if not combos:
        ranking = previous.get("ranking", [])
        for row in ranking:
            key = row.get("key")
            value = row.get("value")
            if key and value:
                combos = [(str(key), str(value))]
                break
    if not combos:
        raise ValueError("strategy_stage lever needs a prior ranking or --top_combos")
    key, value = combos[0]
    return [
        {
            "key": key,
            "value": value,
            "persistent_logit_bias_lambda": lam,
        }
        for lam in _parse_float_list(args.logit_bias_lambdas)
    ]


def _strategy_summarize_candidate(
    *,
    stage: str,
    key: str,
    value: str,
    settings: dict,
    seed_records: list[dict],
    admitted_main: list[str],
    negative_controls: list[str],
    noise_floor: dict[str, dict],
    min_wins: int,
    kv_rag_contract: dict | None = None,
) -> dict:
    kv_rag_contract = kv_rag_contract or {"frame_level_contract_ok": True}
    frontier: list[dict] = []
    for rec in seed_records:
        frontier.extend(rec.get("frontier", []))

    scene_win_details = []
    scene_wins = 0
    for scene in admitted_main:
        vals = [
            row.get("consistency_delta", float("nan"))
            for row in frontier
            if row.get("scene") == scene and not row.get("negative_control", False)
        ]
        mean_delta = _finite_mean(vals)
        threshold = float((noise_floor.get(scene) or {}).get("threshold", float("nan")))
        win = mean_delta == mean_delta and threshold == threshold and mean_delta > threshold
        scene_wins += int(win)
        scene_win_details.append({
            "scene": scene,
            "paired_delta_mean": mean_delta,
            "paired_noise_threshold": threshold,
            "num_seed_pairs": sum(1 for v in vals if _rankable_float(v) != float("-inf")),
            "win_above_noise": bool(win),
        })

    negative_control_details = []
    negative_control_failures = []
    for scene in negative_controls:
        vals = [
            row.get("consistency_delta", float("nan"))
            for row in frontier
            if row.get("scene") == scene and row.get("negative_control", False)
        ]
        mean_delta = _finite_mean(vals)
        threshold = float((noise_floor.get(scene) or {}).get("threshold", float("nan")))
        failed = mean_delta == mean_delta and threshold == threshold and mean_delta > threshold
        if failed:
            negative_control_failures.append(scene)
        negative_control_details.append({
            "scene": scene,
            "paired_delta_mean": mean_delta,
            "paired_noise_threshold": threshold,
            "num_seed_pairs": sum(1 for v in vals if _rankable_float(v) != float("-inf")),
            "sane": not failed,
        })

    guard_failures: list[str] = []
    reinject_rope_frames = 0
    logit_bias_frames = 0
    logit_bias_calls = 0
    for rec in seed_records:
        seed = rec.get("seed")
        gate = rec.get("gate", {})
        if gate.get("blocked_reason"):
            guard_failures.append(f"seed {seed}: {gate['blocked_reason']}")
        for key_name in (
            "adherence_ok",
            "diversity_ok",
            "motion_ok",
            "invariant_ok",
            "scorable_ok",
            "negative_control_ok",
            "frame_contract_ok",
        ):
            if gate.get(key_name) is False:
                guard_failures.append(f"seed {seed}: {key_name}=false")
        for stats in (rec.get("frame_contract") or {}).values():
            reinject_rope_frames += int(stats.get("reinject_rope_injected_frames", 0) or 0)
            logit_bias_frames += int(stats.get("persistent_logit_bias_frames", 0) or 0)
            logit_bias_calls += int(stats.get("persistent_logit_bias_calls", 0) or 0)
    if negative_control_failures:
        guard_failures.append(
            f"negative control above paired noise floor: {negative_control_failures}"
        )
    bias_lambda = float(settings.get("persistent_logit_bias_lambda", 0.0) or 0.0)
    if stage == "lever":
        if reinject_rope_frames <= 0:
            guard_failures.append("lever verification failed: re-RoPE frame-aligned injection path did not engage")
        if bias_lambda > 0.0 and logit_bias_frames <= 0:
            guard_failures.append("lever verification failed: persistent logit bias touched zero frame columns")
    if not kv_rag_contract.get("frame_level_contract_ok", False):
        guard_failures.append(
            "frame-level contract config invalid: "
            f"{kv_rag_contract.get('blocked_reason')}"
        )
    guard_failures = sorted(set(guard_failures))
    guards_ok = not guard_failures

    mean_delta = _finite_mean(
        row.get("consistency_delta", float("nan"))
        for row in frontier
        if not row.get("negative_control", False)
    )
    mean_attention_mass = _finite_mean(row.get("attention_mass_mean", 0.0) for row in frontier)
    mean_frame_mass = _finite_mean(row.get("per_frame_attention_mass_mean", 0.0) for row in frontier)
    return {
        "key": key,
        "value": value,
        "persistent_logit_bias_lambda": bias_lambda,
        "settings": settings,
        "kv_rag_contract": kv_rag_contract,
        "passed": bool(guards_ok and scene_wins >= int(min_wins)),
        "guard_status": "ok" if guards_ok else "failed",
        "guard_failures": guard_failures,
        "mean_aggregate_delta": mean_delta,
        "paired_delta_mean": mean_delta,
        "mean_attention_mass": mean_attention_mass,
        "per_frame_attention_mass_mean": mean_frame_mass,
        "reinject_rope_injected_frames": reinject_rope_frames,
        "persistent_logit_bias_frames": logit_bias_frames,
        "persistent_logit_bias_calls": logit_bias_calls,
        "scene_wins": scene_wins,
        "num_scenes": len(admitted_main),
        "min_scene_wins": int(min_wins),
        "scene_win_details": scene_win_details,
        "negative_control_details": negative_control_details,
        "frontier": frontier,
        "seed_records": seed_records,
    }


def _strategy_ranking_rows(ranked: list[dict]) -> list[dict]:
    return [
        {
            "key": rec["key"],
            "value": rec["value"],
            "persistent_logit_bias_lambda": rec.get("persistent_logit_bias_lambda", 0.0),
            "passed": bool(rec["passed"]),
            "guard_status": rec["guard_status"],
            "paired_delta_mean": rec["paired_delta_mean"],
            "mean_aggregate_delta": rec["mean_aggregate_delta"],
            "scene_wins": f"{rec['scene_wins']}/{rec['num_scenes']}",
            "min_scene_wins": rec["min_scene_wins"],
            "per_frame_attention_mass_mean": rec["per_frame_attention_mass_mean"],
            "mean_attention_mass": rec["mean_attention_mass"],
            "kv_rag_contract": rec.get("kv_rag_contract", {}),
            "reinject_rope_injected_frames": rec.get("reinject_rope_injected_frames", 0),
            "persistent_logit_bias_frames": rec.get("persistent_logit_bias_frames", 0),
            "persistent_logit_bias_calls": rec.get("persistent_logit_bias_calls", 0),
            "guard_failures": rec["guard_failures"],
        }
        for rec in ranked
    ]


def _strategy_screen_selection(stage: str, ranked: list[dict]) -> tuple[dict, str | None]:
    guard_clean = [r for r in ranked if not r.get("guard_failures")]
    if stage == "B1":
        selected_keys: list[str] = []
        for rec in guard_clean:
            if rec["key"] not in selected_keys:
                selected_keys.append(rec["key"])
            if len(selected_keys) >= 2:
                break
        if len(selected_keys) < 2:
            return {"selected_keys": selected_keys, "selected_combos": []}, (
                f"strategy_stage B1 needs two guard-clean keys for B2, got {selected_keys}"
            )
        return {"selected_keys": selected_keys, "selected_combos": []}, None
    if stage == "B2":
        selected_combos = [(rec["key"], rec["value"]) for rec in guard_clean[:2]]
        if len(selected_combos) < 2:
            return {"selected_keys": [], "selected_combos": selected_combos}, (
                f"strategy_stage B2 needs two guard-clean combos for verdict, got {selected_combos}"
            )
        return {"selected_keys": [], "selected_combos": selected_combos}, None
    return {"selected_keys": [], "selected_combos": []}, None


def _strategy_stage_render_eval(
    args,
    output_root: Path,
    *,
    previous: dict,
    candidates: list[tuple[str, str]] | None = None,
    candidate_arms: list[dict] | None = None,
) -> dict:
    started = time.monotonic()
    stage = args.strategy_stage
    metric = "anchor_drift_aggregate_consistency"
    context, context_block = _strategy_context_from_previous(previous, stage=stage)
    if context_block:
        return _strategy_blocked_json(
            stage=stage,
            blocked_reason=context_block,
            render_attempted=False,
            extra={
                "candidate_combos": candidate_arms or [
                    {"key": k, "value": v} for k, v in (candidates or [])
                ],
                "previous_stage_json": args.previous_stage_json,
                "wall_clock_sec": round(time.monotonic() - started, 3),
            },
        )

    cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
    prompt_subset_dir = context["prompt_subset_dir"]
    _set_nested(cfg, "data", "data_path", str(prompt_subset_dir))
    _set_nested(cfg, "inference", "sparse_long_multishot", True)
    num_blocks = args.num_blocks
    if num_blocks is None:
        num_blocks = context.get("regime_proof", {}).get("planned_blocks")
    if num_blocks is None:
        num_blocks = _num_blocks_from_cfg(cfg)

    scorers, scorer_coverage = _build_long_scorers(args, require_adherence=True)
    missing_scorers = sorted(
        name for name, info in scorer_coverage.items()
        if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
    )
    if missing_scorers:
        return _strategy_blocked_json(
            stage=stage,
            blocked_reason=f"requested scorer(s) unavailable: {missing_scorers}",
            render_attempted=False,
            prompt_subset_dir=prompt_subset_dir,
            regime_proof=context.get("regime_proof"),
            prompt_lint=context.get("prompt_lint"),
            lint_passing_main=context.get("lint_passing_main_scenes"),
            negative_controls=context.get("negative_controls"),
            scorer_coverage=scorer_coverage,
            extra={
                "candidate_combos": candidate_arms or [
                    {"key": k, "value": v} for k, v in (candidates or [])
                ],
                "previous_stage_json": args.previous_stage_json,
                "wall_clock_sec": round(time.monotonic() - started, 3),
            },
        )

    seeds = [0] if stage in {"B1", "B2"} else [0, 1, 2]
    admitted_main = context["admitted_main_scenes"]
    negative_controls = context["negative_controls"]
    allowed_eval_scenes = set(admitted_main) | set(negative_controls)
    render_stems = sorted(allowed_eval_scenes)
    min_wins = args.min_scene_wins or -(-len(admitted_main) // 2)
    baseline_dirs = context["baseline_seed_dirs"]
    base_kv_rag = _base_kv_rag_block(cfg)
    shots_for = build_spec_resolver(
        prompt_subset_dir,
        with_captions=True,
        max_chunks=num_blocks,
    )
    _preflight_config(cfg, args.config_path)

    if candidate_arms is None:
        candidate_arms = [
            {"key": k, "value": v, "persistent_logit_bias_lambda": 0.0}
            for k, v in (candidates or [])
        ]
    candidate_records: list[dict] = []
    all_frontier: list[dict] = []
    render_attempted = False
    render_cap = (
        int(args.max_render_stems_per_invocation)
        if int(args.max_render_stems_per_invocation) > 0 else None
    )
    if render_cap is not None:
        for arm in candidate_arms:
            key = str(arm["key"])
            value = str(arm["value"])
            bias_lambda = float(arm.get("persistent_logit_bias_lambda", 0.0) or 0.0)
            settings = _strategy_settings_for_combo(
                args,
                key,
                value,
                persistent_logit_bias_lambda=bias_lambda,
            )
            merged = _finalist_kv_rag(base_kv_rag, settings)
            kv_rag_contract = _kv_rag_contract_summary(merged)
            if not kv_rag_contract["frame_level_contract_ok"]:
                continue
            for seed in seeds:
                name = _strategy_arm_name(stage, key, value, seed, bias_lambda)
                _, mod_dir = _write_one_variant(
                    cfg,
                    output_root,
                    name,
                    kv_rag_settings=merged,
                    seed=seed,
                    multiview_per_perspective=False,
                )
                status = _variant_output_status(mod_dir, name, render_stems)
                if status["complete"]:
                    print(f"[strategy-{stage}] preflight complete {name}: {mod_dir}")
                    continue
                print(
                    f"[strategy-{stage}] preflight rendering {name}: {settings}; "
                    f"missing stems={status['missing_stems']}"
                )
                status, rendered = _render_missing_variant_stems(
                    cfg,
                    output_root,
                    prompt_subset_dir=prompt_subset_dir,
                    variant_name=name,
                    variant_dir=mod_dir,
                    kv_rag_settings=merged,
                    seed=seed,
                    stems=render_stems,
                    status=status,
                    max_stems=render_cap,
                )
                render_attempted = render_attempted or rendered
                if not status["complete"]:
                    return _strategy_blocked_json(
                        stage=stage,
                        blocked_reason=(
                            f"render incomplete for {name}: "
                            f"missing stems={status['missing_stems']}"
                        ),
                        render_attempted=render_attempted,
                        prompt_subset_dir=prompt_subset_dir,
                        regime_proof=context.get("regime_proof"),
                        prompt_lint=context.get("prompt_lint"),
                        lint_passing_main=context.get("lint_passing_main_scenes"),
                        admitted_main=admitted_main,
                        negative_controls=negative_controls,
                        scorer_coverage=scorer_coverage,
                        extra={
                            "candidate_combos": candidate_arms,
                            "previous_stage_json": args.previous_stage_json,
                            "variant_render_status": status,
                            "wall_clock_sec": round(time.monotonic() - started, 3),
                        },
                    )
    for arm in candidate_arms:
        key = str(arm["key"])
        value = str(arm["value"])
        bias_lambda = float(arm.get("persistent_logit_bias_lambda", 0.0) or 0.0)
        settings = _strategy_settings_for_combo(
            args,
            key,
            value,
            persistent_logit_bias_lambda=bias_lambda,
        )
        merged = _finalist_kv_rag(base_kv_rag, settings)
        kv_rag_contract = _kv_rag_contract_summary(merged)
        if not kv_rag_contract["frame_level_contract_ok"]:
            candidate_records.append(_strategy_summarize_candidate(
                stage=stage,
                key=key,
                value=value,
                settings=settings,
                kv_rag_contract=kv_rag_contract,
                seed_records=[],
                admitted_main=admitted_main,
                negative_controls=negative_controls,
                noise_floor=context["noise_floor"],
                min_wins=min_wins,
            ))
            continue
        seed_records: list[dict] = []
        for seed in seeds:
            name = _strategy_arm_name(stage, key, value, seed, bias_lambda)
            mod_cfg, mod_dir = _write_one_variant(
                cfg,
                output_root,
                name,
                kv_rag_settings=merged,
                seed=seed,
                multiview_per_perspective=False,
            )
            status = _variant_output_status(mod_dir, name, render_stems)
            if status["complete"]:
                print(f"[strategy-{stage}] reusing complete {name}: {mod_dir}")
            else:
                print(
                    f"[strategy-{stage}] rendering {name}: {settings}; "
                    f"missing stems={status['missing_stems']}"
                )
                status, rendered = _render_missing_variant_stems(
                    cfg,
                    output_root,
                    prompt_subset_dir=prompt_subset_dir,
                    variant_name=name,
                    variant_dir=mod_dir,
                    kv_rag_settings=merged,
                    seed=seed,
                    stems=render_stems,
                    status=status,
                    max_stems=render_cap,
                )
                render_attempted = render_attempted or rendered
            if not status["complete"]:
                return _strategy_blocked_json(
                    stage=stage,
                    blocked_reason=(
                        f"render incomplete for {name}: "
                        f"missing stems={status['missing_stems']}"
                    ),
                    render_attempted=render_attempted,
                    prompt_subset_dir=prompt_subset_dir,
                    regime_proof=context.get("regime_proof"),
                    prompt_lint=context.get("prompt_lint"),
                    lint_passing_main=context.get("lint_passing_main_scenes"),
                    admitted_main=admitted_main,
                    negative_controls=negative_controls,
                    scorer_coverage=scorer_coverage,
                    extra={
                        "candidate_combos": candidate_arms,
                        "previous_stage_json": args.previous_stage_json,
                        "variant_render_status": status,
                        "wall_clock_sec": round(time.monotonic() - started, 3),
                    },
                )
            result_all = compare_cross_perspective_dirs(
                baseline_dirs[seed],
                mod_dir,
                shots_for=shots_for,
                adherence_scorer=scorers.get("adherence_scorer"),
                invariant_scorer=scorers.get("invariant_scorer"),
                subject_encoder=scorers.get("subject_encoder"),
                background_encoder=scorers.get("background_encoder"),
                dynamic_scorer=scorers.get("dynamic_scorer"),
                max_frames=args.max_frames,
                stride=max(1, args.stride),
            )
            filtered_records = [
                r for r in result_all["records"]
                if r.get("theme") in allowed_eval_scenes
            ]
            result = _copy_result_with_records(result_all, filtered_records)
            gate = evaluate_cross_perspective_gate(
                result,
                metric=metric,
                min_consistency_wins=min_wins,
                adherence_tolerance=args.adherence_tolerance,
                diversity_tolerance=args.diversity_tolerance,
                motion_tolerance=args.motion_tolerance,
                require_adherence=True,
                require_invariant=True,
            )
            gate = _apply_noise_floor_gate(
                gate,
                metric=metric,
                noise_floor_by_scene=context["noise_floor"],
                sigma_multiplier=float(args.noise_sigma_multiplier),
            )
            attention_by_scene = _load_attention_diagnostics(mod_dir, shots_for)
            frame_contract_reason = _frame_contract_blocked_reason({
                scene: attention_by_scene.get(scene, {"diagnostic": {}})
                for scene in allowed_eval_scenes
            })
            if frame_contract_reason:
                prev = gate.get("blocked_reason")
                gate["blocked_reason"] = "; ".join(([prev] if prev else []) + [frame_contract_reason])
                gate["passed"] = False
                gate["is_null_result"] = True
                gate["frame_contract_ok"] = False
            else:
                gate["frame_contract_ok"] = True
            gate["modified_settings"] = settings
            gate["scorer_coverage"] = scorer_coverage
            rows = _frontier_rows(gate, attention_by_scene)
            for row in rows:
                row.update({
                    "stage": stage,
                    "seed": seed,
                    "key": key,
                    "value": value,
                    "persistent_logit_bias_lambda": bias_lambda,
                })
            all_frontier.extend(rows)
            seed_records.append({
                "seed": seed,
                "render_dir": str(mod_dir),
                "config": str(mod_cfg),
                "render_status": {**status, "resume_mode": "per_stem"},
                "gate": gate,
                "comparison": result,
                "attention_diagnostics": attention_by_scene,
                "frame_contract": {
                    scene: _frame_contract_stats(rec.get("diagnostic", {}))
                    for scene, rec in attention_by_scene.items()
                },
                "frontier": rows,
            })
        candidate_records.append(_strategy_summarize_candidate(
            stage=stage,
            key=key,
            value=value,
            settings=settings,
            kv_rag_contract=kv_rag_contract,
            seed_records=seed_records,
            admitted_main=admitted_main,
            negative_controls=negative_controls,
            noise_floor=context["noise_floor"],
            min_wins=min_wins,
        ))

    ranked = sorted(
        candidate_records,
        key=lambda rec: (
            not bool(rec.get("guard_failures")),
            int(rec.get("scene_wins", 0)),
            _rankable_float(rec.get("mean_aggregate_delta")),
        ),
        reverse=True,
    )
    selection = {"selected_keys": [], "selected_combos": []}
    winner = None
    blocked_reason = None
    null_reason = None
    if stage in {"B1", "B2"}:
        selection, blocked_reason = _strategy_screen_selection(stage, ranked)
    else:
        winner = next((rec for rec in ranked if rec.get("passed")), None)
        if winner is None:
            if ranked and all(rec.get("guard_failures") for rec in ranked):
                blocked_reason = f"all strategy_stage {stage} arm(s) failed guard(s)"
            else:
                if stage == "lever":
                    null_reason = (
                        "training-free frame-level KV injection does not move "
                        "cross-shot consistency at 5B; pivot to latent "
                        "re-anchoring or memory LoRA"
                    )
                else:
                    null_reason = (
                        "best combo did not beat paired noise on enough admitted scenes"
                    )

    return {
        "selector": "long_multishot frame-level KV-RAG strategy",
        "strategy_stage": stage,
        "screen_only": stage in {"B1", "B2"},
        "winner": (
            {
                "key": winner["key"],
                "value": winner["value"],
                "persistent_logit_bias_lambda": winner.get("persistent_logit_bias_lambda", 0.0),
            }
            if winner and stage in {"verdict", "lever"} else None
        ),
        "is_null_result": bool((stage in {"verdict", "lever"} and winner is None) or blocked_reason),
        "blocked_reason": blocked_reason,
        "null_reason": null_reason,
        "render_attempted": render_attempted,
        "candidate_combos": candidate_arms,
        "selected_keys": selection["selected_keys"],
        "selected_combos": [
            {"key": key, "value": value}
            for key, value in selection["selected_combos"]
        ],
        "previous_stage_json": args.previous_stage_json,
        "chosen_scenes": list(context.get("regime_proof", {}).get("scene_regime", {}).keys()),
        "prompt_subset_dir": str(prompt_subset_dir),
        "sparse_long_multishot": True,
        "metric": metric,
        "regime_proof": context.get("regime_proof"),
        "baseline_seeds": seeds,
        "baseline_seed_dirs": {
            str(seed): str(path)
            for seed, path in baseline_dirs.items()
            if seed in {0, 1, 2}
        },
        "prompt_lint": context.get("prompt_lint"),
        "lint_passing_main_scenes": context.get("lint_passing_main_scenes"),
        "admitted_main_scenes": admitted_main,
        "admission": context.get("admission"),
        "negative_controls": negative_controls,
        "noise_floor": context.get("noise_floor"),
        "scorer_coverage": scorer_coverage,
        "guards": {
            "adherence_tolerance": args.adherence_tolerance,
            "diversity_tolerance_relative": args.diversity_tolerance,
            "motion_tolerance_relative": args.motion_tolerance,
            "invariant_tolerance": 0.0,
            "require_adherence": True,
            "admission_diversity_floor": float(args.admission_diversity_floor),
            "baseline_drift_floor": float(args.baseline_drift_floor),
            "noise_sigma_multiplier": float(args.noise_sigma_multiplier),
            "frame_contract_required": True,
            "min_scene_wins": int(min_wins),
            "lever_reinject_rope_verification_required": stage == "lever",
        },
        "ranking": _strategy_ranking_rows(ranked),
        "frontier": all_frontier,
        "candidates": ranked,
        "wall_clock_sec": round(time.monotonic() - started, 3),
    }


def _run_long_multishot_strategy_stage(args, output_root: Path) -> None:
    if not args.prompts_dir:
        raise ValueError("long_multishot strategy_stage requires --prompts_dir")
    if args.motion_tolerance is None:
        raise ValueError("long_multishot strategy_stage requires --motion_tolerance")
    if args.strategy_stage == "A":
        result = _strategy_stage_a(args, output_root)
        metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "strategy_stage_A_drift_audit.json"
        save_metrics_json(result, metrics_json)
        print(f"Wrote metrics: {metrics_json.resolve()}")
        if result.get("blocked_reason"):
            print(f"[strategy-A] BLOCKED: {result['blocked_reason']}")
            if result.get("reauthor_scenes"):
                print(f"[strategy-A] re-author/check scenes: {result['reauthor_scenes']}")
        else:
            print(f"[strategy-A] admitted main scenes: {result.get('admitted_main_scenes', [])}")
        return

    previous = None
    if args.previous_stage_json:
        previous = json.loads(Path(args.previous_stage_json).read_text(encoding="utf-8"))
    if args.strategy_stage == "lever":
        lever_arms = _strategy_lever_arms(args, previous)
        result = _strategy_stage_render_eval(
            args,
            output_root,
            previous=previous,
            candidate_arms=lever_arms,
        )
    else:
        candidates = _strategy_stage_candidates(args, previous)
        result = _strategy_stage_render_eval(
            args,
            output_root,
            previous=previous,
            candidates=candidates,
        )
    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / f"strategy_stage_{args.strategy_stage}.json"
    save_metrics_json(result, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    if result.get("blocked_reason"):
        print(f"[strategy-{args.strategy_stage}] BLOCKED: {result['blocked_reason']}")
    elif args.strategy_stage in {"verdict", "lever"}:
        if result.get("winner"):
            print(f"[strategy-{args.strategy_stage}] WINNER: {result['winner']}")
        else:
            print(f"[strategy-{args.strategy_stage}] RESULT: honest NULL -- {result.get('null_reason')}")
    else:
        print(
            f"[strategy-{args.strategy_stage}] selected keys={result.get('selected_keys', [])} "
            f"combos={result.get('selected_combos', [])}"
        )
    for row in result.get("ranking", []):
        print(
            f"  {row['key']}+{row['value']} bias={row.get('persistent_logit_bias_lambda', 0.0)}: "
            f"delta={row['paired_delta_mean']:+.4f} "
            f"wins={row['scene_wins']} guard={row['guard_status']} "
            f"frame_mass={row['per_frame_attention_mass_mean']:.6f}"
        )


def build_mechanism_prompt_subset(
    prompts_dir: str,
    subset: list[str] | None,
    dest: Path,
    *,
    blocks_per_shot: int = 2,
    total_blocks: int | None = None,
    negative_control: str = "shimmering_puzzle_surface",
) -> list[str]:
    """Copy existing prompt folders from one or more roots into a sparse subset.

    The only generated prompt metadata is ``shot_durations.txt``. When
    ``total_blocks`` can be split into 6-12 block shots, durations exactly fill
    the render budget; otherwise the legacy fixed ``blocks_per_shot`` fallback
    is written so the regime proof can fail closed before render.
    """
    roots = _mechanism_prompt_roots(prompts_dir)
    available: dict[str, Path] = {}
    for root in roots:
        for p in sorted(root.iterdir()):
            if p.is_dir() and any(f.name != "global.json" for f in p.glob("*.json")):
                available.setdefault(p.name, p)
    chosen = subset or sorted(available)
    missing = [t for t in chosen if t not in available]
    if missing:
        raise ValueError(f"--prompt_subset themes not found in {prompts_dir}: {missing}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for theme in chosen:
        src = available[theme]
        out = dest / theme
        shutil.copytree(src, out)
        json_files = sorted(
            [f for f in out.glob("*.json") if f.name != "global.json"],
            key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem),
        )
        durations = _balanced_shot_durations(
            len(json_files),
            total_blocks,
            fallback_blocks_per_shot=int(blocks_per_shot),
        )
        durations = [str(int(d)) for d in durations]
        (out / "shot_durations.txt").write_text("\n".join(durations) + "\n", encoding="utf-8")
        if theme == negative_control:
            global_path = out / "global.json"
            meta = {}
            if global_path.exists():
                try:
                    meta = json.loads(global_path.read_text(encoding="utf-8"))
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}
            meta["negative_control"] = True
            global_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chosen


def _copy_result_with_records(result: dict, records: list[dict]) -> dict:
    import copy

    out = copy.deepcopy(result)
    out["records"] = records
    out["num_pairs"] = len(records)
    if records:
        out["baseline_summary"] = summarize_records(records, "baseline_metrics")
        out["modified_summary"] = summarize_records(records, "modified_metrics")
        out["delta_summary"] = summarize_records(records, "delta")
    else:
        out["baseline_summary"] = {}
        out["modified_summary"] = {}
        out["delta_summary"] = {}
    return out


def _metric_sigma(values: list[float]) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _drift_reduction(baseline: float, modified: float) -> float:
    import numpy as np

    if np.isnan(baseline) or np.isnan(modified):
        return float("nan")
    denom = 1.0 - float(baseline)
    if denom <= 1e-8:
        return float("nan")
    return float((float(modified) - float(baseline)) / denom)


def _evaluate_single_long_dir(directory: Path, *, shots_for, scorers: dict, args) -> dict:
    """Evaluate one rendered long_multishot directory by comparing it to itself."""
    return compare_cross_perspective_dirs(
        directory,
        directory,
        shots_for=shots_for,
        adherence_scorer=scorers.get("adherence_scorer"),
        invariant_scorer=scorers.get("invariant_scorer"),
        subject_encoder=scorers.get("subject_encoder"),
        background_encoder=scorers.get("background_encoder"),
        dynamic_scorer=scorers.get("dynamic_scorer"),
        max_frames=args.max_frames,
        stride=max(1, args.stride),
    )


def _scene_metric_records_by_theme(result: dict, metric: str) -> dict[str, dict]:
    out = {}
    for rec in result.get("records", []):
        theme = rec.get("theme")
        if theme:
            out[theme] = {
                "metric": rec["baseline_metrics"].get(metric, float("nan")),
                "metrics": rec["baseline_metrics"],
                "path": rec.get("baseline"),
                "negative_control": bool(rec.get("negative_control", False)),
            }
    return out


def _attention_diag_mean(diag: dict) -> float:
    import numpy as np

    vals = []
    for bank in ("pos", "neg"):
        by_shot = (diag.get(bank) or {}).get("attention_mass_by_shot_layer", {})
        for layers in by_shot.values():
            for rec in layers.values():
                vals.append(float(rec.get("mean_mass", 0.0)))
    return float(np.mean(vals)) if vals else 0.0


def _attention_diag_frame_mean(diag: dict) -> float:
    import numpy as np

    vals = []
    for bank in ("pos", "neg"):
        by_frame = (diag.get(bank) or {}).get("attention_mass_by_shot_layer_frame", {})
        for layers in by_frame.values():
            for frames in layers.values():
                for rec in frames.values():
                    vals.append(float(rec.get("mean_mass", 0.0)))
    return float(np.mean(vals)) if vals else 0.0


def _load_attention_diagnostics(render_dir: Path, shots_for) -> dict[str, dict]:
    diagnostics: dict[str, dict] = {}
    for path in sorted(render_dir.glob("*_kv_rag_diag.json")):
        stem = path.stem
        if stem.endswith("_kv_rag_diag"):
            stem = stem[: -len("_kv_rag_diag")]
        spec = shots_for(stem) if callable(shots_for) else None
        theme = spec.get("theme") if isinstance(spec, dict) else None
        if not theme:
            continue
        try:
            diag = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diag = {"blocked_reason": f"could not read diagnostic sidecar: {exc}"}
        diagnostics[theme] = {
            "path": str(path),
            "mean_persistent_attention_mass": _attention_diag_mean(diag),
            "mean_persistent_frame_attention_mass": _attention_diag_frame_mean(diag),
            "diagnostic": diag,
        }
    return diagnostics


def _apply_noise_floor_gate(
    gate: dict,
    *,
    metric: str,
    noise_floor_by_scene: dict[str, dict],
    sigma_multiplier: float,
) -> dict:
    """Replace raw m>b consistency wins/negative-control triggers with delta > k*sigma."""
    consistency_wins = 0
    negative_control_failures: list[str] = []
    for entry in gate.get("per_prompt", []):
        theme = entry.get("theme")
        floor = noise_floor_by_scene.get(theme, {})
        sigma = float(floor.get("sigma", float("nan")))
        threshold = float(floor.get("threshold", float("nan")))
        baseline = float(entry.get("consistency_baseline", float("nan")))
        modified = float(entry.get("consistency_modified", float("nan")))
        delta = modified - baseline
        raw_win = bool(entry.get("consistency_win", False))
        noise_win = (
            not any(v != v for v in (delta, threshold))
            and delta > threshold
        )
        entry["consistency_win_raw"] = raw_win
        entry["consistency_win"] = bool(noise_win)
        entry["consistency_delta"] = float(delta)
        entry["noise_sigma"] = sigma
        entry["noise_threshold"] = threshold
        entry["noise_sigma_multiplier"] = float(sigma_multiplier)
        entry["relative_drift_reduction"] = _drift_reduction(baseline, modified)
        b0 = float(entry.get("anchor_to_shot0_baseline", entry.get("anchor_to_shot0_consistency_baseline", float("nan"))))
        m0 = float(entry.get("anchor_to_shot0_modified", entry.get("anchor_to_shot0_consistency_modified", float("nan"))))
        if b0 != b0:
            b0 = float("nan")
        if m0 != m0:
            m0 = float("nan")
        entry["to_shot0_delta"] = float(m0 - b0) if not (b0 != b0 or m0 != m0) else float("nan")
        if entry.get("negative_control", False):
            adherence_ok = bool(entry.get("adherence_ok", True))
            if noise_win and adherence_ok:
                negative_control_failures.append(entry.get("stem", theme or "?"))
                entry["negative_control_ok"] = False
                entry["negative_control_reason"] = (
                    f"{metric} delta exceeded {sigma_multiplier}*baseline sigma "
                    "without adherence loss; text-override evidence, not a pass"
                )
            else:
                entry["negative_control_ok"] = True
        elif noise_win:
            consistency_wins += 1

    gate["consistency_wins"] = int(consistency_wins)
    gate["consistency_ok"] = consistency_wins >= int(gate.get("min_consistency_wins", 0))
    gate["negative_control_failures"] = negative_control_failures
    gate["negative_control_ok"] = not negative_control_failures
    blocked_reasons = []
    if gate.get("unscorable"):
        blocked_reasons.append(f"unscorable input(s): {sorted(set(gate['unscorable']))}")
    if negative_control_failures:
        blocked_reasons.append(
            f"negative control indicates text override above noise floor: {negative_control_failures}"
        )
    gate["blocked_reason"] = "; ".join(blocked_reasons) if blocked_reasons else None
    gate["passed"] = bool(
        gate.get("consistency_ok")
        and gate.get("adherence_ok")
        and gate.get("diversity_ok")
        and gate.get("motion_ok")
        and gate.get("invariant_ok")
        and gate.get("scorable_ok")
        and gate.get("negative_control_ok")
    )
    gate["is_null_result"] = not gate["passed"]
    return gate


def _frontier_rows(gate: dict, attention_by_scene: dict[str, dict]) -> list[dict]:
    rows = []
    for entry in gate.get("per_prompt", []):
        theme = entry.get("theme")
        rows.append({
            "scene": theme,
            "stem": entry.get("stem"),
            "negative_control": bool(entry.get("negative_control", False)),
            "metric": entry.get("metric"),
            "consistency_baseline": entry.get("consistency_baseline"),
            "consistency_modified": entry.get("consistency_modified"),
            "consistency_delta": entry.get("consistency_delta"),
            "anchor_centroid_baseline": entry.get("anchor_centroid_baseline"),
            "anchor_centroid_modified": entry.get("anchor_centroid_modified"),
            "anchor_to_shot0_baseline": entry.get("anchor_to_shot0_baseline"),
            "anchor_to_shot0_modified": entry.get("anchor_to_shot0_modified"),
            "worst_anchor_to_shot0_baseline": entry.get("worst_anchor_to_shot0_baseline"),
            "worst_anchor_to_shot0_modified": entry.get("worst_anchor_to_shot0_modified"),
            "noise_sigma": entry.get("noise_sigma"),
            "noise_threshold": entry.get("noise_threshold"),
            "noise_floor_win": entry.get("consistency_win"),
            "relative_drift_reduction": entry.get("relative_drift_reduction"),
            "to_shot0_delta": entry.get("to_shot0_delta"),
            "dynamic_degree_delta": (
                entry.get("dynamic_degree_modified", float("nan"))
                - entry.get("dynamic_degree_baseline", float("nan"))
            ),
            "adherence_mean_delta": (
                entry.get("adherence_modified_mean", float("nan"))
                - entry.get("adherence_baseline_mean", float("nan"))
            ),
            "adherence_min_delta": (
                entry.get("adherence_modified_min", float("nan"))
                - entry.get("adherence_baseline_min", float("nan"))
            ),
            "invariant_margin_mean_delta": (
                entry.get("invariant_margin_modified_mean", float("nan"))
                - entry.get("invariant_margin_baseline_mean", float("nan"))
            ),
            "invariant_margin_min_delta": (
                entry.get("invariant_margin_modified_min", float("nan"))
                - entry.get("invariant_margin_baseline_min", float("nan"))
            ),
            "attention_mass_mean": (
                attention_by_scene.get(theme, {}).get("mean_persistent_attention_mass", 0.0)
                if theme else 0.0
            ),
            "per_frame_attention_mass_mean": (
                attention_by_scene.get(theme, {}).get("mean_persistent_frame_attention_mass", 0.0)
                if theme else 0.0
            ),
        })
    return rows


def _run_long_multishot_mechanism_sweep(args, output_root: Path) -> None:
    """Scene-memory structure/dose sweep with admission and noise-floor readout."""
    import numpy as np

    require_adherence = not args.dry_run_without_adherence
    if not require_adherence:
        raise ValueError("mechanism_sweep is a guarded gate and cannot use --dry_run_without_adherence")
    if not args.prompts_dir:
        raise ValueError("long_multishot mechanism_sweep requires --prompts_dir")
    if args.motion_tolerance is None:
        raise ValueError("long_multishot mechanism_sweep requires --motion_tolerance")

    seeds = _parse_seed_list(args.baseline_seeds)
    metric = "anchor_drift_aggregate_consistency"
    cfg = _apply_overrides(OmegaConf.load(args.config_path), args)
    subset = [s.strip() for s in args.prompt_subset.split(",")] if args.prompt_subset else None
    prompt_subset_dir = output_root / "prompt_subset"
    num_blocks = args.num_blocks if args.num_blocks is not None else _num_blocks_from_cfg(cfg)
    chosen = build_mechanism_prompt_subset(
        args.prompts_dir,
        subset,
        prompt_subset_dir,
        blocks_per_shot=int(args.blocks_per_shot),
        total_blocks=num_blocks,
    )
    _set_nested(cfg, "data", "data_path", str(prompt_subset_dir))
    _set_nested(cfg, "inference", "sparse_long_multishot", True)
    base_kv_rag = _base_kv_rag_block(cfg)
    regime_proof = _long_regime_report(cfg, prompt_subset_dir, num_blocks=num_blocks)
    print(f"[long-mechanism-sweep] subset: {chosen}")
    print(f"[long-mechanism-sweep] baseline seeds: {seeds}")
    print(f"[long-mechanism-sweep] metric: {metric} (centroid still reported)")
    print("[long-mechanism-sweep] sparse_long_multishot=true (global.json ignored by dataset)")
    if not regime_proof["passed"]:
        blocked_reason = regime_proof["blocked_reason"]
        consolidated = {
            "selector": "long_multishot scene-memory mechanism sweep",
            "winner": None,
            "is_null_result": True,
            "blocked_reason": blocked_reason,
            "render_attempted": False,
            "sparse_long_multishot": True,
            "metric": metric,
            "regime_proof": regime_proof,
            "reauthor_scenes": [
                scene for scene, rec in regime_proof.get("scene_regime", {}).items()
                if rec.get("blocked_reason")
            ],
            "arms": _mechanism_sweep_arms(args),
        }
        metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "long_multishot_mechanism_sweep.json"
        save_metrics_json(consolidated, metrics_json)
        print(f"Wrote metrics: {metrics_json.resolve()}")
        print(f"[long-mechanism-sweep] BLOCKED before render: {blocked_reason}")
        return

    prompt_lint, _allowed = _long_prompt_lint(argparse.Namespace(**{**vars(args), "prompts_dir": str(prompt_subset_dir)}))
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
    prompt_specs = load_shot_specs(prompt_subset_dir)
    missing_invariant = [
        scene for scene in lint_passing_main
        if not prompt_specs.get(scene, {}).get("invariant_caption")
        or not prompt_specs.get(scene, {}).get("contrast_caption")
    ]

    scorers, scorer_coverage = _build_long_scorers(args, require_adherence=True)
    missing_scorers = sorted(
        name for name, info in scorer_coverage.items()
        if isinstance(info, dict) and info.get("requested") and not info.get("loaded")
    )
    prereq_blocks = []
    if seeds != [0, 1, 2]:
        prereq_blocks.append(f"baseline_seeds must be exactly 0,1,2 for paired gate decisions, got {seeds}")
    if lint_failures:
        prereq_blocks.append(f"ill-posed prompt scene(s) below text-similarity floor: {lint_failures}")
    if len(lint_passing_main) < 3:
        prereq_blocks.append(f"need >=3 lint-passing non-control scenes, got {len(lint_passing_main)}")
    if not negative_controls:
        prereq_blocks.append("negative control scene missing")
    if missing_invariant:
        prereq_blocks.append(f"missing invariant/contrast captions: {missing_invariant}")
    if missing_scorers:
        prereq_blocks.append(f"requested scorer(s) unavailable: {missing_scorers}")
    if prereq_blocks:
        blocked_reason = "; ".join(prereq_blocks)
        consolidated = {
            "selector": "long_multishot scene-memory mechanism sweep",
            "winner": None,
            "is_null_result": True,
            "blocked_reason": blocked_reason,
            "render_attempted": False,
            "sparse_long_multishot": True,
            "metric": metric,
            "regime_proof": regime_proof,
            "prompt_lint": chosen_lint,
            "lint_passing_main_scenes": lint_passing_main,
            "negative_controls": negative_controls,
            "missing_invariant_contrast_scenes": missing_invariant,
            "scorer_coverage": scorer_coverage,
            "arms": _mechanism_sweep_arms(args),
        }
        metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "long_multishot_mechanism_sweep.json"
        save_metrics_json(consolidated, metrics_json)
        print(f"Wrote metrics: {metrics_json.resolve()}")
        print(f"[long-mechanism-sweep] BLOCKED before render: {blocked_reason}")
        return

    _preflight_config(cfg, args.config_path)

    shots_for = build_spec_resolver(
        prompt_subset_dir, with_captions=True, max_chunks=num_blocks
    )

    baseline_dirs: dict[int, Path] = {}
    baseline_seed_results: dict[int, dict] = {}
    for seed in seeds:
        baseline_cfg, baseline_dir = _write_one_variant(
            cfg,
            output_root,
            f"baseline_seed{seed}",
            kv_rag_settings={"enabled": False},
            seed=seed,
            multiview_per_perspective=False,
        )
        print(f"[long-mechanism-sweep] rendering baseline seed {seed}")
        run_inference(baseline_cfg)
        baseline_dirs[seed] = baseline_dir
        baseline_seed_results[seed] = _evaluate_single_long_dir(
            baseline_dir, shots_for=shots_for, scorers=scorers, args=args
        )

    baseline_by_seed = {
        seed: _scene_metric_records_by_theme(result, metric)
        for seed, result in baseline_seed_results.items()
    }
    seed0 = seeds[0]
    seed0_records = baseline_by_seed[seed0]
    noise_floor_by_scene: dict[str, dict] = {}
    admission: dict[str, dict] = {}
    admitted_main: list[str] = []
    for scene in chosen:
        vals = [
            baseline_by_seed.get(seed, {}).get(scene, {}).get("metric", float("nan"))
            for seed in seeds
        ]
        sigma = _metric_sigma(vals)
        threshold = float(args.noise_sigma_multiplier) * sigma if not np.isnan(sigma) else float("nan")
        noise_floor_by_scene[scene] = {
            "baseline_values": [float(v) for v in vals],
            "sigma": sigma,
            "threshold": threshold,
            "sigma_multiplier": float(args.noise_sigma_multiplier),
        }
        finite_vals = [float(v) for v in vals if not np.isnan(v)]
        baseline_drift = float(np.mean([1.0 - v for v in finite_vals])) if finite_vals else float("nan")
        seed0_metrics = seed0_records.get(scene, {}).get("metrics", {})
        diversity = seed0_metrics.get("inter_shot_composition_diversity", float("nan"))
        is_negative = scene in negative_controls
        admitted = (
            not is_negative
            and scene in lint_passing_main
            and scene not in missing_invariant
            and diversity == diversity
            and diversity >= float(args.admission_diversity_floor)
            and baseline_drift == baseline_drift
            and baseline_drift >= float(args.baseline_drift_floor)
        )
        reason = None
        if is_negative:
            reason = "negative control is sanity-checked, not admitted as a main scene"
        elif scene not in lint_passing_main:
            reason = "prompt lint failed"
        elif scene in missing_invariant:
            reason = "missing invariant/contrast captions"
        elif not (diversity == diversity):
            reason = "baseline inter-shot composition diversity unscorable"
        elif diversity < float(args.admission_diversity_floor):
            reason = (
                "baseline inter-shot composition diversity below admission floor "
                f"({diversity:.6f} < {float(args.admission_diversity_floor):.6f})"
            )
        elif not (baseline_drift == baseline_drift):
            reason = f"baseline {metric} drift unscorable"
        elif baseline_drift < float(args.baseline_drift_floor):
            reason = (
                f"baseline drift below headroom floor ({baseline_drift:.6f} "
                f"< {float(args.baseline_drift_floor):.6f})"
            )
        admission[scene] = {
            "admitted": bool(admitted),
            "negative_control": bool(is_negative),
            "baseline_seed": seed0,
            "baseline_inter_shot_composition_diversity": diversity,
            "diversity_floor": float(args.admission_diversity_floor),
            "baseline_drift": baseline_drift,
            "baseline_drift_floor": float(args.baseline_drift_floor),
            "blocked_reason": reason,
        }
        if admitted:
            admitted_main.append(scene)

    min_wins = args.min_scene_wins or -(-len(admitted_main) // 2)
    if not admitted_main:
        print("[long-mechanism-sweep][warn] no main scenes admitted; renders will still be scored fail-closed")

    arm_records = []
    frontier = []
    baseline_dir0 = baseline_dirs[seed0]
    allowed_eval_scenes = set(admitted_main) | set(negative_controls)
    for arm in _mechanism_sweep_arms(args):
        settings = arm["settings"]
        merged = _finalist_kv_rag(base_kv_rag, settings)
        name = arm["name"]
        mod_cfg, mod_dir = _write_one_variant(
            cfg,
            output_root,
            name,
            kv_rag_settings=merged,
            seed=seed0,
            multiview_per_perspective=False,
        )
        print(f"[long-mechanism-sweep] rendering {name}: {settings}")
        run_inference(mod_cfg)
        result_all = compare_cross_perspective_dirs(
            baseline_dir0,
            mod_dir,
            shots_for=shots_for,
            adherence_scorer=scorers.get("adherence_scorer"),
            invariant_scorer=scorers.get("invariant_scorer"),
            subject_encoder=scorers.get("subject_encoder"),
            background_encoder=scorers.get("background_encoder"),
            dynamic_scorer=scorers.get("dynamic_scorer"),
            max_frames=args.max_frames,
            stride=max(1, args.stride),
        )
        filtered_records = [
            r for r in result_all["records"]
            if r.get("theme") in allowed_eval_scenes
        ]
        result = _copy_result_with_records(result_all, filtered_records)
        gate = evaluate_cross_perspective_gate(
            result,
            metric=metric,
            min_consistency_wins=min_wins,
            adherence_tolerance=args.adherence_tolerance,
            diversity_tolerance=args.diversity_tolerance,
            motion_tolerance=args.motion_tolerance,
            require_adherence=True,
            require_invariant=True,
        )
        gate = _apply_noise_floor_gate(
            gate,
            metric=metric,
            noise_floor_by_scene=noise_floor_by_scene,
            sigma_multiplier=float(args.noise_sigma_multiplier),
        )
        attention_by_scene = _load_attention_diagnostics(mod_dir, shots_for)
        frame_contract_reason = _frame_contract_blocked_reason({
            scene: attention_by_scene.get(scene, {"diagnostic": {}})
            for scene in allowed_eval_scenes
        })
        if frame_contract_reason:
            prev = gate.get("blocked_reason")
            gate["blocked_reason"] = "; ".join(([prev] if prev else []) + [frame_contract_reason])
            gate["passed"] = False
            gate["is_null_result"] = True
            gate["frame_contract_ok"] = False
        else:
            gate["frame_contract_ok"] = True
        rows = _frontier_rows(gate, attention_by_scene)
        frontier.extend([{**row, "arm": name} for row in rows])
        deltas = [
            row["consistency_delta"]
            for row in rows
            if not row.get("negative_control") and row.get("consistency_delta") == row.get("consistency_delta")
        ]
        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        arm_records.append({
            "name": name,
            "description": arm["description"],
            "key": "subject_identity",
            "value": "raw",
            "settings": settings,
            "passed": bool(gate["passed"] and len(admitted_main) > 0),
            "mean_aggregate_delta": mean_delta,
            "scene_wins": gate["consistency_wins"],
            "num_scenes": len(admitted_main),
            "gate": gate,
            "comparison": result,
            "attention_diagnostics": attention_by_scene,
            "frame_contract": {
                scene: _frame_contract_stats(rec.get("diagnostic", {}))
                for scene, rec in attention_by_scene.items()
            },
            "frontier": rows,
        })

    ranked = sorted(
        arm_records,
        key=lambda r: (r["passed"], r["scene_wins"], r["mean_aggregate_delta"]),
        reverse=True,
    )
    winner = next((r for r in ranked if r["passed"]), None)
    blocked_reason = None
    if not admitted_main:
        blocked_reason = "no main scenes admitted by baseline diversity floor"
        for r in ranked:
            r["passed"] = False
    consolidated = {
        "selector": "long_multishot scene-memory mechanism sweep",
        "winner": ({"name": winner["name"], "key": winner["key"], "value": winner["value"]} if winner else None),
        "is_null_result": winner is None,
        "blocked_reason": blocked_reason,
        "render_attempted": True,
        "sparse_long_multishot": True,
        "metric": metric,
        "regime_proof": regime_proof,
        "baseline_seeds": seeds,
        "baseline_seed_dirs": {str(k): str(v) for k, v in baseline_dirs.items()},
        "prompt_lint": chosen_lint,
        "lint_passing_main_scenes": lint_passing_main,
        "admitted_main_scenes": admitted_main,
        "admission": admission,
        "negative_controls": negative_controls,
        "missing_invariant_contrast_scenes": missing_invariant,
        "noise_floor": noise_floor_by_scene,
        "scorer_coverage": scorer_coverage,
        "guards": {
            "adherence_tolerance": args.adherence_tolerance,
            "diversity_tolerance_relative": args.diversity_tolerance,
            "motion_tolerance_relative": args.motion_tolerance,
            "invariant_tolerance": 0.0,
            "require_adherence": True,
            "admission_diversity_floor": float(args.admission_diversity_floor),
            "noise_sigma_multiplier": float(args.noise_sigma_multiplier),
            "baseline_drift_floor": float(args.baseline_drift_floor),
            "frame_contract_required": True,
        },
        "ranking": [
            {
                "name": r["name"],
                "passed": r["passed"],
                "mean_aggregate_delta": r["mean_aggregate_delta"],
                "scene_wins": f"{r['scene_wins']}/{r['num_scenes']}",
            }
            for r in ranked
        ],
        "frontier": frontier,
        "arms": ranked,
    }
    metrics_json = Path(args.metrics_json) if args.metrics_json else output_root / "long_multishot_mechanism_sweep.json"
    save_metrics_json(consolidated, metrics_json)
    print(f"Wrote metrics: {metrics_json.resolve()}")
    for r in ranked:
        print(f"  {r['name']}: mean_delta={r['mean_aggregate_delta']:+.4f} "
              f"wins={r['scene_wins']}/{r['num_scenes']} passed={r['passed']}")
    if winner:
        print(f"[long-mechanism-sweep] WINNER: {winner['name']}")
    else:
        print("[long-mechanism-sweep] RESULT: honest NULL/BLOCKED -- no arm passed all guards.")


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
