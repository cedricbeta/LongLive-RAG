#!/usr/bin/env python3
"""Round 17 bounded VLM-in-the-loop ablation orchestrator."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(cmd: list[str]) -> None:
    print("[round17] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def patch_admission_prompt_subset(stage_a_json: Path, prompt_subset_dir: Path, out_path: Path) -> Path:
    data = load_json(stage_a_json)
    data["round17_parent_admission_json"] = str(stage_a_json)
    data["prompt_subset_dir"] = str(prompt_subset_dir)
    write_json(out_path, data)
    return out_path


def copy_optimizer_artifacts(src_docs: Path, src_videos: Path, dst_docs: Path, dst_videos: Path) -> None:
    dst_docs.mkdir(parents=True, exist_ok=True)
    dst_videos.mkdir(parents=True, exist_ok=True)
    for filename in ("optimizer_decisions.json", "manual_anchor_plan.json"):
        shutil.copy2(src_docs / filename, dst_docs / filename)
    for dirname in ("prompt_original", "prompt_refined"):
        src = src_videos / dirname
        dst = dst_videos / dirname
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def parse_seed_list(spec: str) -> list[int]:
    seeds = [int(tok.strip()) for tok in str(spec or "").split(",") if tok.strip()]
    if not seeds:
        raise ValueError("--baseline_seeds needs at least one seed")
    return seeds


def preseed_stage_a_baselines(stage_a_video_root: Path, seeds: list[int]) -> dict[str, Any]:
    """Copy prior Stage-A baseline videos so only new scenes render."""
    source_root = ROOT / "videos/frame_strategy_A"
    copied: list[str] = []
    missing: list[str] = []
    for seed in seeds:
        src_dir = source_root / f"baseline_seed{seed}"
        dst_dir = stage_a_video_root / f"baseline_seed{seed}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            missing.append(str(src_dir))
            continue
        for src in sorted(src_dir.glob(f"baseline_seed{seed}-rank*-*_regular.mp4")):
            dst = dst_dir / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                copied.append(str(dst))
    cfg_src = source_root / "configs"
    cfg_dst = stage_a_video_root / "configs"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        src = cfg_src / f"baseline_seed{seed}.yaml"
        if src.exists():
            dst = cfg_dst / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    return {
        "source_root": str(source_root),
        "requested_seeds": seeds,
        "copied": copied,
        "missing": missing,
        "policy": "reuse requested existing baseline seed videos; render only missing broader-scene stems",
    }


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_root", default=f"docs/round17_{stamp}")
    parser.add_argument("--video_root", default=f"videos/round17_{stamp}")
    parser.add_argument("--config_path", default="configs/inference_kv_rag_long_multishot.yaml")
    parser.add_argument("--admission_json_override", default=None)
    parser.add_argument(
        "--prompts_dir",
        default="example/long_multishot_prompts:example/multiview_prompts",
    )
    parser.add_argument("--prompt_subset", default=None, help="Debug cap; omitted means all prompt folders.")
    parser.add_argument("--rounds", type=int, default=3, help="Bounded refinement rounds, max 3.")
    parser.add_argument("--kv_selection_mode", default="per_boundary",
                        choices=("per_boundary", "legacy_single_call"))
    parser.add_argument("--kv_candidate_frames_per_shot", type=int, default=8)
    parser.add_argument("--kv_max_candidates_per_boundary", type=int, default=24)
    parser.add_argument("--prompt_review", default="on", choices=("on", "off"))
    parser.add_argument("--multi_pass_watch_arm", default="both",
                        choices=("both", "kv_only", "prompt_only", "baseline"),
                        help="From round 2 on, the optimizer watches THIS arm's previous-round "
                             "render (feedback loop). 'baseline' restores the old behaviour of "
                             "re-diagnosing the untouched baseline every round.")
    parser.add_argument("--primary_logit_bias_lambda", type=float, default=1.0)
    parser.add_argument("--extra_logit_bias_lambdas", default="2", help="Final static-prompt lambda sweep.")
    parser.add_argument("--kv_anchor_cap", type=int, default=4)
    parser.add_argument("--optimizer_frames_per_shot", type=int, default=4)
    parser.add_argument("--judge_frames_per_shot", type=int, default=1)
    parser.add_argument("--min_admitted_main_scenes", type=int, default=8)
    parser.add_argument("--baseline_seeds", default="0")
    parser.add_argument("--baseline_drift_floor", type=float, default=0.02)
    parser.add_argument("--admission_diversity_floor", type=float, default=0.03)
    parser.add_argument("--noise_sigma_multiplier", type=float, default=2.0)
    parser.add_argument("--single_seed_consistency_delta_threshold", type=float, default=0.02)
    parser.add_argument("--motion_tolerance", type=float, default=0.2)
    parser.add_argument("--diversity_tolerance", type=float, default=0.1)
    parser.add_argument("--adherence_tolerance", type=float, default=0.02)
    parser.add_argument("--prompt_similarity_floor", type=float, default=0.12)
    parser.add_argument("--clip_device", default="cuda:0")
    parser.add_argument("--optimizer_gpu", type=int, default=None)
    parser.add_argument("--generator_gpu", type=int, default=None)
    parser.add_argument("--codex_model", default="gpt-5.5")
    parser.add_argument("--claude_model", default="claude-opus-4-8")
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen_mem_fraction_static", type=float, default=0.45)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 1 or args.rounds > 3:
        raise ValueError("--rounds must be in [1, 3]")
    docs_root = Path(args.docs_root)
    video_root = Path(args.video_root)
    docs_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    stage_a_video_root = video_root / "stage_A"
    stage_a_json = Path(args.admission_json_override) if args.admission_json_override else docs_root / "stage_A_drift_audit.json"
    seeds = parse_seed_list(args.baseline_seeds)
    if args.admission_json_override:
        write_json(docs_root / "stage_A_preseed_baselines.json", {
            "policy": "stage-A skipped; using explicit admission_json_override",
            "admission_json_override": str(stage_a_json),
            "requested_seeds": seeds,
        })
    else:
        preseed = preseed_stage_a_baselines(stage_a_video_root, seeds)
        write_json(docs_root / "stage_A_preseed_baselines.json", preseed)
    stage_a_cmd = [
        sys.executable,
        "scripts/run_kv_rag_ablation.py",
        "--config_path",
        args.config_path,
        "--mode",
        "long_multishot",
        "--prompts_dir",
        args.prompts_dir,
        "--blocks_per_shot",
        "6",
        "--baseline_seeds",
        args.baseline_seeds,
        "--baseline_drift_floor",
        str(args.baseline_drift_floor),
        "--admission_diversity_floor",
        str(args.admission_diversity_floor),
        "--noise_sigma_multiplier",
        str(args.noise_sigma_multiplier),
        "--single_seed_consistency_delta_threshold",
        str(args.single_seed_consistency_delta_threshold),
        "--motion_tolerance",
        str(args.motion_tolerance),
        "--diversity_tolerance",
        str(args.diversity_tolerance),
        "--adherence_tolerance",
        str(args.adherence_tolerance),
        "--prompt_similarity_floor",
        str(args.prompt_similarity_floor),
        "--clip_device",
        args.clip_device,
        "--strategy_stage",
        "A",
        "--output_root",
        str(stage_a_video_root),
        "--metrics_json",
        str(stage_a_json),
        "--min_admitted_main_scenes",
        str(args.min_admitted_main_scenes),
    ]
    if args.prompt_subset:
        stage_a_cmd += ["--prompt_subset", args.prompt_subset]
    if args.admission_json_override:
        print(f"[round17] using admission override: {stage_a_json}", flush=True)
    elif args.resume and stage_a_json.exists():
        prior_stage_a = load_json(stage_a_json)
        if (
            not prior_stage_a.get("blocked_reason")
            and [int(seed) for seed in (prior_stage_a.get("baseline_seeds") or [])] == seeds
            and len(prior_stage_a.get("admitted_main_scenes") or []) >= args.min_admitted_main_scenes
        ):
            print(f"[round17] reusing completed Stage-A admission: {stage_a_json}", flush=True)
        else:
            run(stage_a_cmd)
    else:
        run(stage_a_cmd)

    stage_a = load_json(stage_a_json)
    if stage_a.get("blocked_reason"):
        raise RuntimeError(f"Stage-A admission blocked: {stage_a['blocked_reason']}")
    if len(stage_a.get("admitted_main_scenes") or []) < args.min_admitted_main_scenes:
        raise RuntimeError("Stage-A did not admit the required number of main scenes")

    iteration_records: list[dict[str, Any]] = []
    admission_path = stage_a_json
    for idx in range(1, args.rounds + 1):
        iter_docs = docs_root / f"iter{idx:02d}"
        iter_videos = video_root / f"iter{idx:02d}"
        optimizer_watch: tuple[str, str] | None = None
        if idx > 1:
            prev_videos = video_root / f"iter{idx - 1:02d}"
            admission_path = patch_admission_prompt_subset(
                stage_a_json,
                prev_videos / "prompt_refined",
                docs_root / f"iter{idx:02d}_admission.json",
            )
            if args.multi_pass_watch_arm != "baseline":
                # Feedback loop: diagnose the previous round's INTERVENED render,
                # not the untouched baseline, so later rounds target residual breaks.
                optimizer_watch = (
                    str(prev_videos / args.multi_pass_watch_arm),
                    args.multi_pass_watch_arm,
                )
        cmd = [
            sys.executable,
            "scripts/run_vlm_closed_judge_ablation.py",
            "--output_root",
            str(iter_docs),
            "--video_root",
            str(iter_videos),
            "--config_path",
            args.config_path,
            "--admission_json",
            str(admission_path),
            "--numeric_reference_json",
            str(stage_a_json),
            "--codex_model",
            args.codex_model,
            "--claude_model",
            args.claude_model,
            "--qwen_model",
            args.qwen_model,
            "--qwen_mem_fraction_static",
            str(args.qwen_mem_fraction_static),
            "--clip_device",
            args.clip_device,
            "--motion_tolerance",
            str(args.motion_tolerance),
            "--diversity_tolerance",
            str(args.diversity_tolerance),
            "--adherence_tolerance",
            str(args.adherence_tolerance),
            "--prompt_similarity_floor",
            str(args.prompt_similarity_floor),
            "--kv_anchor_cap",
            str(args.kv_anchor_cap),
            "--optimizer_frames_per_shot",
            str(args.optimizer_frames_per_shot),
            "--judge_frames_per_shot",
            str(args.judge_frames_per_shot),
            "--min_admitted_main_scenes",
            str(args.min_admitted_main_scenes),
            "--primary_logit_bias_lambda",
            str(args.primary_logit_bias_lambda),
            "--single_seed_consistency_delta_threshold",
            str(args.single_seed_consistency_delta_threshold),
            "--kv_selection_mode",
            args.kv_selection_mode,
            "--kv_candidate_frames_per_shot",
            str(args.kv_candidate_frames_per_shot),
            "--kv_max_candidates_per_boundary",
            str(args.kv_max_candidates_per_boundary),
            "--prompt_review",
            args.prompt_review,
        ]
        if optimizer_watch is not None:
            cmd += [
                "--optimizer_video_dir",
                optimizer_watch[0],
                "--optimizer_video_prefix",
                optimizer_watch[1],
            ]
        if args.optimizer_gpu is not None:
            cmd += ["--optimizer_gpu", str(args.optimizer_gpu)]
        if args.generator_gpu is not None:
            cmd += ["--generator_gpu", str(args.generator_gpu)]
        if args.resume:
            cmd.append("--resume")
        run(cmd)
        ledger = load_json(iter_docs / "ledger.json")
        iteration_records.append({
            "round": idx,
            "docs": str(iter_docs),
            "videos": str(iter_videos),
            "ledger": str(iter_docs / "ledger.json"),
            "pass": bool(ledger.get("pass")),
            "is_null_result": bool(ledger.get("is_null_result")),
            "winner": ledger.get("winner"),
            "blocked_reason": ledger.get("blocked_reason"),
        })
        if ledger.get("pass"):
            break

    final_idx = int(iteration_records[-1]["round"])
    final_docs = docs_root / f"iter{final_idx:02d}"
    final_videos = video_root / f"iter{final_idx:02d}"
    extra_records = []
    for token in [x.strip() for x in args.extra_logit_bias_lambdas.split(",") if x.strip()]:
        lam = float(token)
        if lam == float(args.primary_logit_bias_lambda):
            continue
        name = f"lambda{str(lam).replace('.', 'p')}_static"
        lam_docs = docs_root / name
        lam_videos = video_root / name
        copy_optimizer_artifacts(final_docs, final_videos, lam_docs, lam_videos)
        admission_path = patch_admission_prompt_subset(
            stage_a_json,
            final_videos / "prompt_original",
            docs_root / f"{name}_admission.json",
        )
        cmd = [
            sys.executable,
            "scripts/run_vlm_closed_judge_ablation.py",
            "--output_root",
            str(lam_docs),
            "--video_root",
            str(lam_videos),
            "--config_path",
            args.config_path,
            "--admission_json",
            str(admission_path),
            "--numeric_reference_json",
            str(stage_a_json),
            "--codex_model",
            args.codex_model,
            "--claude_model",
            args.claude_model,
            "--qwen_model",
            args.qwen_model,
            "--qwen_mem_fraction_static",
            str(args.qwen_mem_fraction_static),
            "--clip_device",
            args.clip_device,
            "--motion_tolerance",
            str(args.motion_tolerance),
            "--diversity_tolerance",
            str(args.diversity_tolerance),
            "--adherence_tolerance",
            str(args.adherence_tolerance),
            "--prompt_similarity_floor",
            str(args.prompt_similarity_floor),
            "--kv_anchor_cap",
            str(args.kv_anchor_cap),
            "--optimizer_frames_per_shot",
            str(args.optimizer_frames_per_shot),
            "--judge_frames_per_shot",
            str(args.judge_frames_per_shot),
            "--min_admitted_main_scenes",
            str(args.min_admitted_main_scenes),
            "--primary_logit_bias_lambda",
            str(lam),
            "--single_seed_consistency_delta_threshold",
            str(args.single_seed_consistency_delta_threshold),
            "--resume",
        ]
        if args.generator_gpu is not None:
            cmd += ["--generator_gpu", str(args.generator_gpu)]
        run(cmd)
        ledger = load_json(lam_docs / "ledger.json")
        extra_records.append({
            "lambda": lam,
            "docs": str(lam_docs),
            "videos": str(lam_videos),
            "ledger": str(lam_docs / "ledger.json"),
            "pass": bool(ledger.get("pass")),
            "is_null_result": bool(ledger.get("is_null_result")),
            "winner": ledger.get("winner"),
            "blocked_reason": ledger.get("blocked_reason"),
        })

    manifest = {
        "stage_A": str(stage_a_json),
        "stage_A_video_root": str(stage_a_video_root),
        "stage_A_preseed": str(docs_root / "stage_A_preseed_baselines.json"),
        "iterations": iteration_records,
        "extra_logit_bias_runs": extra_records,
        "final_iteration": iteration_records[-1],
        "caps": {
            "max_refinement_rounds": int(args.rounds),
            "primary_logit_bias_lambda": float(args.primary_logit_bias_lambda),
            "extra_logit_bias_lambdas": [r["lambda"] for r in extra_records],
            "kv_anchor_cap": int(args.kv_anchor_cap),
            "optimizer_frames_per_shot": int(args.optimizer_frames_per_shot),
            "kv_selection_mode": args.kv_selection_mode,
            "kv_candidate_frames_per_shot": int(args.kv_candidate_frames_per_shot),
            "kv_max_candidates_per_boundary": int(args.kv_max_candidates_per_boundary),
            "prompt_review": args.prompt_review,
            "multi_pass_watch_arm": args.multi_pass_watch_arm,
        },
        "command_args": vars(args),
    }
    write_json(docs_root / "round17_manifest.json", manifest)
    print(f"[round17] wrote {docs_root / 'round17_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
