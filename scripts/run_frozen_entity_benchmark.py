#!/usr/bin/env python3
"""Run the existing 33-scene entity loop without changing its decision policy.

Fresh directory only, fixed historical baselines, frozen input/code hashes,
one invocation per scene, and explicit failures in the 33-scene denominator.
This runs generation/control only; completion is NOT a benchmark score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_hash(stream)


def _stream_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def verify_hashes(hashes):
    changed = [path for path, digest in hashes.items() if not Path(path).is_file() or sha256(path) != digest]
    if changed:
        raise RuntimeError(f"Frozen inputs/code changed; refusing to mix implementations: {changed[:8]}")


def scene_command(output, scene, base_url, render_gpu):
    return [sys.executable, "-u", str(ROOT / "scripts/run_agent_edit_demo.py"),
        "--base_url", base_url, "--render_gpu", str(render_gpu),
        "--clipseg_device", f"cuda:{render_gpu}",
        "--base_config", str(output / "frozen/base.yaml"),
        "--data_root", str(output / "frozen/data"),
        "--entity_specs", str(output / "frozen/entity_specs.json"),
        "--output_root", str(output / "armC_current"),
        "--judge_cache_dir", str(output / "controller_cache"),
        "--judge", "entity", "--model", "Qwen/Qwen3-VL-8B-Instruct",
        "--judge_seed", "0", "--entry_sigma", "1.0",
        "--max_edits_per_shot", "3", "--commit_lookahead_shots", "2",
        "--place_absent", "never", "--reference_registration_min_confidence", "0.75",
        "--allow_missing_global", "--scenes", scene]


def prepare(output):
    import yaml

    archive = ROOT / "videos/narrastream_threearm_20260827"
    manifest = json.loads((archive / "manifest.json").read_text())
    scenes = manifest["scenes"]
    if len(scenes) != 33 or len(set(scenes)) != 33:
        raise ValueError("This runner is explicitly for the existing 33-scene iteration suite")
    audit = json.loads((ROOT / "docs/entity_judge_assignment_20260905/baseline_audit.json").read_text())
    sources = {row["scene"]: row for row in audit["records"]}
    if set(sources) != set(scenes):
        raise ValueError("Baseline audit coverage mismatch")
    for scene in scenes:
        row = sources[scene]
        if not row["render_settings_equal"] or not row["authored_inputs_equal"] or sha256(row["a_video"]) != row["a_sha256"]:
            raise ValueError(f"Baseline audit no longer valid: {scene}")
    base = yaml.safe_load((archive / "armC_base.yaml").read_text())
    if base["inference"]["kv_rag"]["enabled"] or base["logging"]["seed"] != 0:
        raise ValueError("C baseline must have KV-RAG off and seed zero")
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen").mkdir()
    (output / "logs").mkdir()
    shutil.copy2(archive / "armC_base.yaml", output / "frozen/base.yaml")
    shutil.copy2(ROOT / "example/narrastream_bench/entity_specs_subset33.json", output / "frozen/entity_specs.json")
    shutil.copytree(ROOT / "example/narrastream_bench/multitext_subset33_b8", output / "frozen/data")
    for scene in scenes:
        render_dir = output / "armC_current" / scene / "renders"
        render_dir.mkdir(parents=True)
        (render_dir / f"baseline-rank0-{scene}-seed0_regular.mp4").symlink_to(sources[scene]["a_video"])
    paths = {ROOT / "inference.py", Path(__file__).resolve(), ROOT / "scripts/run_agent_edit_demo.py"}
    for folder in ("utils", "pipeline", "model", "wan", "wan_5b", "evaluation"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.update((ROOT / "configs").rglob("*.yaml"))
    paths.update(path for path in (output / "frozen").rglob("*") if path.is_file())
    hashes = {str(path): sha256(path) for path in sorted(paths)}
    hashes.update({row["a_video"]: row["a_sha256"] for row in sources.values()})
    record = dict(schema_version=1, suite="NarraStream iteration33", seconds=16, seed=0,
        expected_scene_count=33, scenes=scenes, source_hashes=hashes,
        baseline_policy="exact archived A videos; no fresh baseline generation",
        prefix_check_limitation="Archived A has no latent digest; existing decoded-frame tolerance guard is used for the first repair, not bit-exact latent certification.",
        controller_policy="unchanged live entity judge; calibration not certified; experimental role/appearance candidates not enabled",
        budget=dict(max_edits_per_shot=3, entry_sigma=1.0, commit_lookahead_shots=2, place_absent="never"),
        environment_limitation="Repository code/data/config hashes are checked between scenes; external installed packages and large pretrained weights are not copied into this run.",
        evaluation_complete=False, status="prepared", created_unix=time.time(), results=[])
    write_json(output / "run.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--judge-gpu", type=int, default=3)
    args = parser.parse_args()
    if args.render_gpu == args.judge_gpu:
        parser.error("Use separate judge and renderer GPUs")
    output = args.output.resolve()
    record = prepare(output)
    from scripts.run_agent_edit_demo import start_vlm_server, gpu_free_mib

    server = None
    try:
        if gpu_free_mib(args.render_gpu) < 75000:
            raise RuntimeError("Renderer GPU is in use; refusing to compete with an existing job")
        server_args = argparse.Namespace(gpu=args.judge_gpu,
            sglang_python="/home/chendong/miniconda3/envs/sglang/bin/python",
            model="Qwen/Qwen3-VL-8B-Instruct", mem_fraction_static=0.45, startup_timeout=600)
        server, base_url = start_vlm_server(server_args, output)
        record.update(status="running", judge_base_url=base_url,
            judge_gpu=args.judge_gpu, render_gpu=args.render_gpu, pid=os.getpid())
        write_json(output / "run.json", record)
        for scene in record["scenes"]:
            verify_hashes(record["source_hashes"])
            if shutil.disk_usage(output).free < 50 * 1024**3:
                raise RuntimeError("Less than 50 GiB free; stopping before another scene")
            if gpu_free_mib(args.render_gpu) < 75000:
                raise RuntimeError("Renderer GPU became occupied; stopping without touching other jobs")
            command = scene_command(output, scene, base_url, args.render_gpu)
            row = dict(scene=scene, status="running", started_unix=time.time(), command=command)
            record["results"].append(row)
            write_json(output / "run.json", record)
            with (output / "logs" / f"{scene}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row.update(returncode=result.returncode, wall_seconds=time.time()-row["started_unix"],
                status="driver_completed" if result.returncode == 0 else "driver_failed")
            verify_hashes(record["source_hashes"])
            write_json(output / "run.json", record)
        record["status"] = "generation_complete" if all(r["returncode"] == 0 for r in record["results"]) else "generation_finished_with_failures"
    except Exception as error:
        record.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["updated_unix"] = time.time()
        write_json(output / "run.json", record)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)


if __name__ == "__main__":
    main()
