#!/usr/bin/env python3
"""Freeze, run and present an exploratory baseline/original/feedback comparison.

prepare never renders. run executes immutable source snapshots on explicitly
assigned GPUs. report retains every selected case, including failed runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("utils", "pipeline", "model", "wan", "wan_5b", "evaluation", "configs", "scripts", "tests", "fouroversix")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def snapshot_source(target):
    target.mkdir(exist_ok=False)
    paths = list(ROOT.glob("*.py"))
    for folder in FOLDERS:
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and p.suffix in (".py", ".yaml", ".yml", ".json", ".txt", ".toml")
                     and "__pycache__" not in p.parts)
    hashes = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(relative)] = digest(dest)
    for folder in ("checkpoints", "longlive_models", "wan_models", "assets"):
        (target / folder).symlink_to(ROOT / folder, target_is_directory=True)
    return hashes


def prepare(args):
    output = args.output
    if (output / "manifest.json").exists():
        raise RuntimeError("Comparison already frozen; use a new output for a different treatment")
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.cases.read_text())
    original_meta = output / "original_source.json"
    if not original_meta.exists():
        raise RuntimeError("Missing pre-change source_original snapshot and original_source.json")
    original_hashes = json.loads(original_meta.read_text())["sha256"]
    feedback_hashes = snapshot_source(output / "source_feedback")
    frozen = output / "frozen"
    frozen.mkdir()
    shutil.copy2(ROOT / "videos/narrastream_threearm_20260827/armC_base.yaml", frozen / "base.yaml")
    shutil.copy2(ROOT / "example/narrastream_bench/entity_specs_subset33.json", frozen / "entity_specs.json")
    audits = {r["scene"]: r for r in json.loads((ROOT / "docs/entity_judge_assignment_20260905/baseline_audit.json").read_text())["records"]}
    for case in spec["cases"]:
        scene = case["scene"]
        audit = audits[scene]
        baseline = Path(audit["a_video"])
        if digest(baseline) != audit["a_sha256"]:
            raise RuntimeError(f"Archived baseline changed: {scene}")
        shutil.copytree(ROOT / "example/narrastream_bench/multitext_subset33_b8" / scene, frozen / "data" / scene)
        dest = output / "baseline" / f"{scene}.mp4"
        dest.parent.mkdir(exist_ok=True)
        dest.symlink_to(baseline)
        case["baseline_video"] = str(dest)
        case["baseline_sha256"] = audit["a_sha256"]
        case["review_checklist"] = ["目标缺陷确实改善", "原有角色与身份保留", "动作与场景未明显损坏", "后续镜头未新增明显问题"]
    hashes = {str(p): digest(p) for p in frozen.rglob("*") if p.is_file()}
    hashes.update({str(output / "source_original" / p): h for p, h in original_hashes.items()})
    hashes.update({str(output / "source_feedback" / p): h for p, h in feedback_hashes.items()})
    hashes.update({case["baseline_video"]: case["baseline_sha256"] for case in spec["cases"]})
    manifest = {**spec, "created_unix": time.time(), "source_hashes": hashes,
                "runtime_python": str(ROOT / "venv/bin/python"), "status": "prepared",
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "limitations": ["Exploratory development cases, no statistical quality claim",
                                "Equal edit cap; actual planner and wall costs are reported separately",
                                "Pretrained weight directories and installed environments are shared, not copied",
                                "Original versus feedback compares the complete revision, including the constraint-adapter fix"]}
    verify(manifest)
    write_json(output / "manifest.json", manifest)
    report(args)
    print(json.dumps({"prepared_cases": len(spec["cases"]), "output": str(output)}, ensure_ascii=False))


def verify(manifest):
    changed = [path for path, value in manifest["source_hashes"].items()
               if not Path(path).is_file() or digest(path) != value]
    if changed:
        raise RuntimeError(f"Frozen sources or inputs changed: {changed[:5]}")


def command(output, manifest, scene, arm, gpu, base_url):
    source = output / ("source_original" if arm == "original" else "source_feedback")
    # Each invocation has a private root/cache; parallel case summaries cannot
    # overwrite each other. Both arms begin at the identical archived A.
    run_root = output / "runs" / arm / scene
    flags = [manifest["runtime_python"], "-u", str(source / "scripts/run_agent_edit_demo.py"),
             "--base_url", base_url, "--render_gpu", str(gpu), "--clipseg_device", f"cuda:{gpu}",
             "--base_config", str(output / "frozen/base.yaml"),
             "--data_root", str(output / "frozen/data"), "--entity_specs", str(output / "frozen/entity_specs.json"),
             "--output_root", str(run_root), "--scenes", scene, "--allow_missing_global",
             "--judge", "entity", "--model", manifest["model"], "--judge_seed", "0",
             "--max_edits_per_shot", str(manifest["budget"]["max_edits_per_shot"]),
             "--commit_lookahead_shots", str(manifest["budget"]["commit_lookahead_shots"]),
             "--entry_sigma", "1.0", "--place_absent", "never", "--absent_action", "regenerate",
             "--comparer", "color", "--role_localizer", "indexed", "--uncertain_policy", "accept_unknown",
             "--attribute_masks", "--judge_chunks", "sparse"]
    if arm != "original":
        flags.extend(["--repair_policy", "feedback", "--max_repair_seconds", str(manifest["budget"]["max_repair_seconds"])])
    return source, run_root, flags


def run(args):
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text())
    verify(manifest)
    cases = [c for c in manifest["cases"] if not args.scenes or c["scene"] in args.scenes]
    if not cases or (args.scenes and set(args.scenes) - {c["scene"] for c in cases}):
        raise ValueError("Select scenes from the frozen manifest")
    arms = ["original", "feedback"] if args.arm == "both" else [args.arm]
    if len(set(args.render_gpus)) != len(args.render_gpus):
        raise ValueError("Each render lane needs a distinct GPU")

    def lane(gpu, lane_cases):
        lock_path = Path(f"/tmp/longlive_feedback_render_gpu{gpu}.lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for case in lane_cases:
                for arm in arms:
                    scene = case["scene"]
                    source, run_root, cmd = command(output, manifest, scene, arm, gpu, args.base_url)
                    status_path = run_root / "status.json"
                    if status_path.exists():
                        status = json.loads(status_path.read_text())
                        print(f"{arm}/{scene}: retained existing {status['status']}", flush=True)
                        continue
                    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)], text=True).strip())
                    if free < 60000:
                        raise RuntimeError(f"GPU {gpu} has only {free} MiB free")
                    verify(manifest)
                    render_dir = run_root / scene / "renders"
                    render_dir.mkdir(parents=True)
                    (render_dir / f"baseline-rank0-{scene}-seed0_regular.mp4").symlink_to(Path(case["baseline_video"]).resolve())
                    started = time.time()
                    status = {"scene": scene, "arm": arm, "status": "running", "started_unix": started,
                              "command": cmd, "source": str(source), "render_gpu": gpu}
                    write_json(status_path, status)
                    print(f"GPU {gpu}: {arm}/{scene} started", flush=True)
                    with (run_root / "driver.log").open("x") as log:
                        child = subprocess.Popen(cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        status["pid"] = child.pid
                        write_json(status_path, status)
                        try:
                            code = child.wait(timeout=manifest["budget"]["case_timeout_seconds"])
                            state = "completed" if code == 0 else "failed"
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            code, state = child.returncode, "timeout"
                    try:
                        verify(manifest)
                    except RuntimeError as error:
                        state = "invalid_frozen_inputs"
                        status["error"] = str(error)
                    status.update(status=state, returncode=code, wall_seconds=time.time()-started)
                    write_json(status_path, status)
                    print(f"GPU {gpu}: {arm}/{scene} {state}", flush=True)
    with ThreadPoolExecutor(max_workers=len(args.render_gpus)) as executor:
        jobs = [executor.submit(lane, gpu, cases[i::len(args.render_gpus)]) for i, gpu in enumerate(args.render_gpus)]
        for job in jobs:
            job.result()
    report(args)


def report(args):
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text())
    panels = output / "comparisons"
    panels.mkdir(exist_ok=True)
    rows = []
    for case in manifest["cases"]:
        scene = case["scene"]
        row = {**case, "arms": {}}
        for arm in ("original", "feedback"):
            run_root = output / "runs" / arm / scene
            status = run_root / "status.json"
            state = json.loads(status.read_text()) if status.exists() else {"status": "not_started"}
            summary = run_root / scene / "summary.json"
            if state["status"] == "completed" and summary.exists():
                record = json.loads(summary.read_text())["scenes"][scene]
                state.update({k: record.get(k) for k in ("final_delivery_video", "accepted_repairs", "unresolved_shots", "edit_attempts", "rolled_back_repairs", "repair_plan_calls", "repair_planning_seconds")})
                transcripts = list((run_root / "vlm_cache").glob("*/*.json"))
                state["cached_transcript_count"] = len(transcripts)
            row["arms"][arm] = state
        videos = [case["baseline_video"], *(row["arms"][a].get("final_delivery_video") for a in ("original", "feedback"))]
        dest = panels / f"{scene}.mp4"
        if all(videos) and not dest.exists():
            graph = ";".join(f"[{i}:v]scale=640:352,setsar=1,drawtext=text='{label}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[v{i}]"
                             for i, label in enumerate(("baseline", "original", "feedback v1")))
            graph += ";[v0][v1][v2]hstack=inputs=3[out]"
            cmd = ["ffmpeg", "-nostdin", "-v", "error"]
            for video in videos:
                cmd.extend(["-i", video])
            subprocess.run([*cmd, "-filter_complex", graph, "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "20", "-y", str(dest)], check=True)
        row["comparison_video"] = str(dest) if dest.exists() else None
        rows.append(row)
    write_json(output / "comparison.json", {"quality_claim": False, "cases": rows})
    parts = ["<!doctype html><meta charset='utf-8'><title>Video repair comparison</title>",
             "<style>body{font:16px system-ui;margin:30px;background:#10151c;color:#e8edf4}section{margin:30px 0;padding:20px;background:#1d2631;border-radius:12px}video{width:100%;max-width:1600px}a{color:#9ccaff}td,th{padding:8px;text-align:left}</style>",
             "<h1>视频修复：baseline / 原版 / feedback v1</h1><p>六个案例在新版运行前固定；全部保留。裁判接受数不等于人工确认的质量提升。</p>"]
    for row in rows:
        parts.append(f"<section><h2>{html.escape(row['scene'])}</h2><p>{html.escape(row['reason_zh'])}</p>")
        parts.append("<table><tr><th>版本</th><th>状态</th><th>编辑次数</th><th>耗时（秒）</th></tr>")
        for arm, state in row["arms"].items():
            parts.append(f"<tr><td>{arm}</td><td>{html.escape(state['status'])}</td><td>{html.escape(str(state.get('edit_attempts', '—')))}</td><td>{round(state.get('wall_seconds',0),1)}</td></tr>")
        parts.append("</table>")
        if row["comparison_video"]:
            relative = os.path.relpath(row["comparison_video"], output)
            parts.append(f"<video controls preload='metadata' src='{html.escape(relative)}'></video>")
        else:
            relative = os.path.relpath(row["baseline_video"], output)
            parts.append(f"<p>三组结果齐备后生成并排视频。<a href='{html.escape(relative)}'>查看固定 baseline</a></p>")
        parts.append("<p>检查：" + " · ".join(map(html.escape, row["review_checklist"])) + "</p></section>")
    (output / "index.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=ROOT / "videos/feedback_repair_20260914")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/feedback_repair_cases_20260914.json")
    parser.add_argument("--base-url")
    parser.add_argument("--arm", choices=("original", "feedback", "both"), default="both")
    parser.add_argument("--render-gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == "run" and not args.base_url:
        parser.error("run needs an existing local --base-url")
    {"prepare": prepare, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
