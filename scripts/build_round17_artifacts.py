#!/usr/bin/env python3
"""Build Round 17 grids and REPORT.md from completed ledgers."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.multiview_prompts import build_spec_resolver  # noqa: E402
from evaluation.video_consistency import chunk_durations_to_boundaries, read_video  # noqa: E402
from run_kv_rag_ablation import _num_blocks_from_cfg  # type: ignore  # noqa: E402
from run_vlm_closed_judge_ablation import scene_video  # type: ignore  # noqa: E402


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: str | Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()
    except Exception:
        return str(path)


def safe_token(value: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:120] or "sample"


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}"


def overlay_label(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.7) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def resize_cell(frame: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(cell_w / max(1, w), cell_h / max(1, h))
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    x = (cell_w - nw) // 2
    y = (cell_h - nh) // 2
    out[y : y + nh, x : x + nw] = resized
    return out


def probe_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else 24.0


def render_4up_grid(
    *,
    scene: str,
    videos: dict[str, Path],
    labels: dict[str, str],
    prompt_dir: Path,
    config_path: Path,
    output: Path,
    cell_height: int = 240,
) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    num_blocks = _num_blocks_from_cfg(cfg)
    resolver = build_spec_resolver(prompt_dir, with_captions=True, max_chunks=num_blocks)
    spec = resolver(next(iter(videos.values())).stem)
    frames_by_arm = {arm: read_video(path) for arm, path in videos.items()}
    min_frames = min(frames.shape[0] for frames in frames_by_arm.values())
    first = next(iter(frames_by_arm.values()))
    cell_h = int(cell_height)
    cell_w = int(round(first.shape[2] * cell_h / max(1, first.shape[1])))
    if cell_w % 2:
        cell_w += 1
    if cell_h % 2:
        cell_h += 1
    fps = probe_fps(next(iter(videos.values())))
    boundaries: list[int] = []
    if spec and spec.get("chunk_durations"):
        derived = chunk_durations_to_boundaries(min_frames, list(spec["chunk_durations"]))
        boundaries = [int(x) for x in derived] if derived else []
    cut_to_index = {b: i + 1 for i, b in enumerate(boundaries)}
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cell_w * 2, cell_h * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {output}")
    # 2x2: top row baseline | prompt_only, bottom row kv_only | both
    grid_rows = [["baseline", "prompt_only"], ["kv_only", "both"]]
    for idx in range(min_frames):
        row_imgs = []
        for row in grid_rows:
            cells = []
            for arm in row:
                rgb = frames_by_arm[arm][idx]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cell = resize_cell(bgr, cell_w, cell_h)
                overlay_label(cell, labels[arm], 10, 28, 0.65)
                cells.append(cell)
            row_imgs.append(np.hstack(cells))
        canvas = np.vstack(row_imgs)
        overlay_label(canvas, f"%{{pts:hms}} {hms(idx / fps)}", cell_w - 60, cell_h * 2 - 12, 0.55)
        active_cuts = [b for b in boundaries if b <= idx < b + max(2, int(round(fps * 0.4)))]
        for boundary in active_cuts:
            cut_label = f"CUT {cut_to_index.get(boundary, '?')}"
            for row_top in (0, cell_h):
                cv2.line(canvas, (0, row_top + 46), (cell_w * 2, row_top + 46), (0, 0, 255), 3)
            overlay_label(canvas, cut_label, cell_w - 42, 72, 0.65)
        writer.write(canvas)
    writer.release()
    return {
        "scene": scene,
        "path": str(output),
        "frames": int(min_frames),
        "fps": float(fps),
        "shot_cut_boundaries": boundaries,
        "labels": labels,
    }


def summarize_ablation_table(ledger: dict[str, Any]) -> list[str]:
    lines = [
        "| arm | pass | judge wins | mean judge delta | numeric pass | attention mass | logit calls | guard reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger.get("ablation_table", []):
        reason = str(row.get("numeric_blocked_reason") or "").replace("\n", " ")
        if len(reason) > 120:
            reason = reason[:117] + "..."
        lines.append(
            "| {arm} | {passed} | {wins} | {delta} | {np} | {mass:.6f} | {calls} | {reason} |".format(
                arm=row.get("arm"),
                passed=row.get("passed"),
                wins=f"{row.get('judge_wins')}/{row.get('min_wins')}",
                delta=(
                    f"{float(row.get('mean_judge_delta')):+.4f}"
                    if row.get("mean_judge_delta") is not None else "n/a"
                ),
                np=row.get("numeric_pass"),
                mass=float(row.get("persistent_rag_attention_mass_mean", 0.0) or 0.0),
                calls=int(row.get("persistent_logit_bias_calls", 0) or 0),
                reason=reason or "-",
            )
        )
    return lines


def diff_summary(diff: dict[str, Any]) -> str:
    before = diff.get("before") or {}
    after = diff.get("after") or {}
    before_text = before.get("caption") or before.get("invariant_caption") or ""
    after_text = after.get("caption") or after.get("invariant_caption") or ""
    if len(before_text) > 260:
        before_text = before_text[:257] + "..."
    if len(after_text) > 360:
        after_text = after_text[:357] + "..."
    return (
        f"- `{diff.get('path')}`\n"
        f"  - before: {before_text or '<empty>'}\n"
        f"  - after: {after_text or '<empty>'}"
    )


def anchor_plan_mismatch(shared: dict[str, Any], plan: list[dict[str, Any]]) -> bool:
    shared_boundaries = shared.get("boundaries") or {}
    shared_map = {
        int(k): [int(item.get("frame_index")) for item in v]
        for k, v in shared_boundaries.items()
    }
    scene_map = {
        int(item.get("boundary_shot_index")): [int(x) for x in item.get("decoded_frame_indices", [])]
        for item in plan
    }
    return shared_map != scene_map


def build_report(
    *,
    docs_root: Path,
    video_root: Path,
    manifest: dict[str, Any],
    final_ledger: dict[str, Any],
    extra_ledgers: list[dict[str, Any]],
    grid_records: list[dict[str, Any]],
) -> str:
    stage_a = load_json(manifest["stage_A"])
    try:
        pre_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        pre_commit = "unavailable"
    lines: list[str] = []
    lines.append("# Round 17 VLM Closed-Judge Ablation")
    lines.append("")
    lines.append(f"- Generated UTC: {final_ledger.get('timestamp_utc')}")
    lines.append(f"- Repo hash before artifact commit: `{pre_commit}`")
    lines.append(f"- Result: `{'PASS' if final_ledger.get('pass') else 'HONEST NULL'}`")
    lines.append(f"- Winner: `{final_ledger.get('winner')}`")
    gate = final_ledger.get("consistency_delta_gate") or {}
    if gate.get("reduced_power_single_seed"):
        lines.append("- Gate power: `REDUCED-POWER single-seed exploratory gate; no paired noise floor`")
    lines.append(f"- Baseline path: `{final_ledger['admission']['baseline_dir']}`")
    baseline_cfg = Path(final_ledger["admission"]["baseline_dir"]).parent / "configs" / "baseline_seed0.yaml"
    lines.append(f"- Baseline config: `{baseline_cfg}`")
    lines.append(f"- Full Round17 manifest: `{rel(docs_root / 'round17_manifest.json', docs_root)}`")
    lines.append("")
    lines.append("## Pipeline")
    cmd = " ".join(["python", "scripts/run_round17_vlm_loop.py"] + [
        f"--{k} {v}" for k, v in (manifest.get("command_args") or {}).items()
        if k in {
            "docs_root", "video_root", "config_path", "prompts_dir", "rounds",
            "admission_json_override",
            "primary_logit_bias_lambda", "extra_logit_bias_lambdas",
            "kv_anchor_cap", "baseline_seeds", "baseline_drift_floor",
            "admission_diversity_floor", "single_seed_consistency_delta_threshold",
            "motion_tolerance", "diversity_tolerance", "adherence_tolerance",
            "clip_device", "min_admitted_main_scenes",
        }
    ])
    lines.append("Orchestrator command:")
    lines.append("")
    lines.append(f"```bash\nCUDA_VISIBLE_DEVICES=<idle-gpu> {cmd}\n```")
    lines.append("")
    lines.append("Per-arm reproduction commands for the final primary round:")
    final_video_root = Path(manifest["final_iteration"]["videos"])
    for arm in ("prompt_only", "kv_only", "both"):
        cfg = final_video_root / "configs" / f"{arm}.yaml"
        lines.append(f"- `{arm}`: `CUDA_VISIBLE_DEVICES=<generator-gpu> python inference.py --config_path {cfg}`")
    lines.append("")
    lines.append("## Scene Admission")
    lines.append(f"- Candidate scenes: {len(stage_a.get('chosen_scenes') or [])}")
    lines.append(f"- Admitted main scenes: {len(stage_a.get('admitted_main_scenes') or [])}")
    lines.append(f"- Negative controls: {stage_a.get('negative_controls')}")
    synthetic_negative = stage_a.get("synthetic_negative_control") or final_ledger["admission"].get("synthetic_negative_control") or {}
    if synthetic_negative:
        lines.append(
            "- Synthetic negative control: `{kind}` from `{scene}`".format(
                kind=synthetic_negative.get("fixture_kind") or "degraded_copy_cheat",
                scene=synthetic_negative.get("source_scene"),
            )
        )
    negative_result = final_ledger.get("negative_control") or {}
    negative_verdict = ((negative_result.get("judge") or {}).get("verdict") or {})
    negative_flags = negative_verdict.get("cheat_flags") or {}
    if negative_result:
        lines.append(
            "- Synthetic negative result: passed_guard=`{passed}` fixture=`{fixture}` flags=`{flags}`".format(
                passed=negative_result.get("passed"),
                fixture=negative_result.get("fixture_kind") or "degraded_copy_cheat",
                flags=negative_flags,
            )
        )
    deferred = stage_a.get("deferred_main_scenes") or final_ledger["admission"].get("deferred_main_scenes") or []
    if deferred:
        lines.append(f"- Deferred from fast run, not rejected: {deferred}")
    exploratory = stage_a.get("exploratory_main_scenes") or final_ledger["admission"].get("exploratory_main_scenes") or []
    if exploratory:
        lines.append(f"- Exploratory scored main scenes: {exploratory}")
    stage_guards = stage_a.get("guards") or {}
    lines.append(
        "- Consistency-delta gate: `{policy}` threshold=`{threshold}` seeds=`{seeds}`".format(
            policy=stage_guards.get("consistency_delta_gate"),
            threshold=stage_guards.get("single_seed_consistency_delta_threshold")
            if stage_guards.get("reduced_power_single_seed") else stage_guards.get("noise_sigma_multiplier"),
            seeds=stage_a.get("baseline_seeds"),
        )
    )
    if stage_guards.get("reduced_power_single_seed"):
        lines.append(f"- Ledger note: {stage_guards.get('gate_power_note')}")
    lines.append("")
    lines.append("| scene | admitted | negative | reject reason | diversity | drift |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for scene in stage_a.get("chosen_scenes") or []:
        rec = (stage_a.get("admission") or {}).get(scene, {})
        lines.append(
            "| {scene} | {adm} | {neg} | {reason} | {div} | {drift} |".format(
                scene=scene,
                adm=rec.get("admitted"),
                neg=rec.get("negative_control"),
                reason=str(rec.get("blocked_reason") or "-").replace("|", "/"),
                div=(
                    f"{float(rec.get('baseline_inter_shot_composition_diversity')):.4f}"
                    if rec.get("baseline_inter_shot_composition_diversity") is not None else "n/a"
                ),
                drift=(
                    f"{float(rec.get('baseline_drift')):.4f}"
                    if rec.get("baseline_drift") is not None else "n/a"
                ),
            )
        )
    lines.append("")
    authored = (stage_a.get("prompt_global_authoring") or {}).get("authored") or []
    lines.append(f"Authored missing `global.json` invariant/contrast captions in the prompt subset for {len(authored)} scene(s).")
    lines.append("")
    lines.append("## Final Primary Round")
    lines.extend(summarize_ablation_table(final_ledger))
    lines.append("")
    if extra_ledgers:
        lines.append("## Static Logit-Bias Sweep")
        for ledger in extra_ledgers:
            label = Path(ledger["artifacts"]["output_root"]).name
            lines.append(f"### {label}")
            lines.extend(summarize_ablation_table(ledger))
            lines.append("")
    lines.append("## Grid Videos")
    for rec in grid_records:
        lines.append(f"- `{rec['scene']}`: [{Path(rec['path']).name}]({rel(rec['path'], docs_root)}) cuts={rec['shot_cut_boundaries']}")
    lines.append("")
    lines.append("## Per-Scene Details")
    decisions = final_ledger.get("optimizer_decisions") or {}
    diffs_by_scene: dict[str, list[dict[str, Any]]] = {}
    for diff in final_ledger.get("prompt_refine_diffs") or []:
        diffs_by_scene.setdefault(diff.get("scene"), []).append(diff)
    shared_plan = final_ledger.get("manual_anchor_plan") or {}
    scenes = list(final_ledger["admission"].get("admitted_main_scenes") or [])
    scenes += list(final_ledger["admission"].get("negative_controls") or [])
    for scene in scenes:
        lines.append(f"### {scene}")
        scene_meta = (final_ledger["admission"].get("admission") or {}).get(scene, {})
        if scene_meta.get("exploratory_ill_posed_main"):
            lines.append("- Exploratory note: ill-posed/self-contradicting prompt promoted to scored main scene for this fast run.")
        scene_dec = (decisions.get(scene) or {}).get("decision") or {}
        lines.append(f"- Shared manual plan differs from per-scene KV plan: `{anchor_plan_mismatch(shared_plan, scene_dec.get('kv_anchor_plan') or [])}`")
        lines.append("- Prompt diffs:")
        for diff in diffs_by_scene.get(scene, []):
            lines.append(diff_summary(diff))
        if not diffs_by_scene.get(scene):
            lines.append("  - none")
        lines.append("- KV-selected decoded frames:")
        for item in scene_dec.get("kv_anchor_plan") or []:
            lines.append(
                f"  - boundary `{item.get('boundary_shot_index')}` frames `{item.get('decoded_frame_indices')}`: {item.get('rationale') or '-'}"
            )
        if not scene_dec.get("kv_anchor_plan"):
            lines.append("  - none")
        lines.append("- Selected/anchor frame gallery:")
        frame_records = (decisions.get(scene) or {}).get("frames_shown") or []
        chosen_frames = {
            int(frame)
            for item in scene_dec.get("kv_anchor_plan") or []
            for frame in item.get("decoded_frame_indices", [])
        }
        gallery_count = 0
        for rec in frame_records:
            for path in rec.get("paths") or []:
                if int(Path(path).stem.rsplit("idx", 1)[-1]) in chosen_frames:
                    lines.append(f"  - ![{scene} {Path(path).stem}]({rel(path, docs_root)})")
                    gallery_count += 1
        if gallery_count == 0:
            lines.append("  - no selected PNGs found")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_root", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--config_path", default="configs/inference_kv_rag_long_multishot.yaml")
    parser.add_argument("--cell_height", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs_root = Path(args.docs_root)
    video_root = Path(args.video_root)
    manifest = load_json(docs_root / "round17_manifest.json")
    final_ledger = load_json(manifest["final_iteration"]["ledger"])
    extra_ledgers = [load_json(rec["ledger"]) for rec in manifest.get("extra_logit_bias_runs", [])]
    final_videos = Path(manifest["final_iteration"]["videos"])
    final_prompt_dir = final_videos / "prompt_refined"
    baseline_dir = Path(final_ledger["admission"]["baseline_dir"])
    baseline_prefix = final_ledger["admission"]["baseline_prefix"]
    seed = 0
    grid_records = []
    scenes = list(final_ledger["admission"].get("admitted_main_scenes") or [])
    scenes += list(final_ledger["admission"].get("negative_controls") or [])
    for scene in scenes:
        videos = {
            "baseline": scene_video(baseline_dir, baseline_prefix, scene, seed),
            "prompt_only": scene_video(final_videos / "prompt_only", "prompt_only", scene, seed),
            "kv_only": scene_video(final_videos / "kv_only", "kv_only", scene, seed),
            "both": scene_video(final_videos / "both", "both", scene, seed),
        }
        out = video_root / "grid_comparisons" / f"{safe_token(scene)}_seed0_4up.mp4"
        rec = render_4up_grid(
            scene=scene,
            videos=videos,
            labels={
                "baseline": "baseline=[REUSED]",
                "prompt_only": "prompt_only",
                "kv_only": "kv_only",
                "both": "both",
            },
            prompt_dir=final_prompt_dir,
            config_path=final_videos / "configs" / "both.yaml",
            output=out,
            cell_height=args.cell_height,
        )
        grid_records.append(rec)
    write_text(docs_root / "grid_records.json", json.dumps(grid_records, indent=2, sort_keys=True) + "\n")
    report = build_report(
        docs_root=docs_root,
        video_root=video_root,
        manifest=manifest,
        final_ledger=final_ledger,
        extra_ledgers=extra_ledgers,
        grid_records=grid_records,
    )
    write_text(docs_root / "REPORT.md", report)
    print(f"[round17-artifacts] wrote {docs_root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
