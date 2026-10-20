#!/usr/bin/env python3
"""Queue unedited replays for failed prefix checks, without changing benchmark outcomes.

One replay per recorded failure, only after the original process group releases
its GPUs. Exact candidate prefix hashes and all archived baseline decoded frames
are compared. This is diagnostic evidence, never an original-A latent certificate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_audited_narrastream_eval import atomic_json, digest
from scripts.run_frozen_entity_benchmark import verify_hashes

TERMINAL = {"generation_complete", "generation_finished_with_failures", "stopped"}
FAILURE = re.compile(r"RuntimeError: (edit_shot\d+_attempt\d+): replay prefix diverged from (branch-\d+) before chunk (\d+):")


def discover_cases(generation, state):
    cases = []
    seen = set()
    for row in state["results"]:
        if row["status"] != "driver_failed":
            continue
        scene = row["scene"]
        if scene in seen or scene not in state["scenes"]:
            raise ValueError(f"Duplicate or out-of-population failure: {scene}")
        seen.add(scene)
        log = generation / "logs" / f"{scene}.log"
        matches = FAILURE.findall(log.read_text())
        if not matches:
            continue
        if len(matches) != 1 or matches[0][1] != "branch-0000":
            raise ValueError(f"Needs a different replay diagnostic: {scene}: {matches}")
        prefix, parent, chunk_end = matches[0]
        folder = generation / "armC_current" / scene
        config = folder / "configs/baseline.yaml"
        candidate = folder / "renders" / f"{prefix}_latents/chunk_digests.json"
        baseline = folder / "renders" / f"baseline-rank0-{scene}-seed0_regular.mp4"
        evidence = [config, candidate, baseline, log]
        cases.append({"scene": scene, "status": "queued", "original_status": "driver_failed",
            "candidate_prefix": prefix, "parent": parent, "chunk_end": int(chunk_end),
            "baseline_config": str(config), "candidate_digests": str(candidate),
            "archived_baseline": str(baseline), "failure_log": str(log),
            "input_hashes": {str(path): digest(path) for path in evidence}})
    return cases


def eligible(state, original_group_active, free_mib):
    return state["status"] in TERMINAL and not original_group_active and free_mib >= 75000


def group_active(pgid):
    result = subprocess.run(["ps", "-eo", "pgid=,stat="], check=True, text=True, capture_output=True)
    return any(int(parts[0]) == pgid and not parts[1].startswith("Z")
               for line in result.stdout.splitlines() if len(parts := line.split()) == 2)


def compare_digests(parent, child, chunk_end):
    for key in ("shape", "dtype", "num_chunks"):
        if parent[key] != child[key]:
            raise ValueError(f"Latent layout mismatch: {key}")
    count = parent["num_chunks"]
    if not 0 < chunk_end <= count or any(len(row["chunk_digests"]) != count for row in (parent, child)):
        raise ValueError("Invalid or incomplete digest coverage")
    mismatch = [i for i in range(chunk_end) if parent["chunk_digests"][i] != child["chunk_digests"][i]]
    return {"basis": "new_baseline_vs_existing_candidate_latent_digests", "chunks": chunk_end,
            "mismatched_chunks": mismatch, "identical": not mismatch}


def compare_videos(archive, replay, expected_frames=384, expected_fps=24):
    import cv2
    import numpy as np
    captures = [cv2.VideoCapture(str(path)) for path in (archive, replay)]
    maes, max_pixel = [], 0
    try:
        if not all(cap.isOpened() for cap in captures):
            raise ValueError("Cannot open baseline videos")
        fps = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
        if any(abs(value - expected_fps) > 0.001 for value in fps):
            raise ValueError(f"Unexpected baseline FPS: {fps}")
        while True:
            a_ok, a = captures[0].read()
            b_ok, b = captures[1].read()
            if a_ok != b_ok:
                raise ValueError("Baseline decoded frame counts differ")
            if not a_ok:
                break
            if a.shape != b.shape:
                raise ValueError("Baseline frame shapes differ")
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            maes.append(float(delta.mean()))
            max_pixel = max(max_pixel, int(delta.max()))
        if len(maes) != expected_frames:
            raise ValueError(f"Expected {expected_frames} baseline frames, decoded {len(maes)}")
        return {"basis": "all_decoded_frames", "frames": len(maes), "fps": fps,
                "max_abs_pixel_error": max_pixel, "mean_frame_mae": sum(maes) / len(maes),
                "max_frame_mae": max(maes), "per_frame_mae": maes, "identical": max_pixel == 0}
    finally:
        for capture in captures:
            capture.release()


def run_case(case, gpu, output):
    import torch
    import yaml
    from scripts.run_agent_edit_demo import build_render_config, render, find_render, latent_chunk_digests
    verify_hashes(case["input_hashes"])
    folder = output / case["scene"]
    folder.mkdir(exist_ok=False)
    original = yaml.safe_load(Path(case["baseline_config"]).read_text())
    inference = original["inference"]
    if inference.get("kv_rag", {}).get("enabled") or any(
        section.get(key) for section in (original, inference) for key in ("edit_renoise", "edit_renoise_plans")
    ):
        raise ValueError("Expected an unedited, KV-RAG-off baseline")
    config = build_render_config(Path(case["baseline_config"]), scene=case["scene"],
        data_dir=Path(original["data"]["data_path"]), output_folder=folder / "renders",
        prefix="baseline_replay", edit=None, target=folder / "baseline_replay.yaml")
    latent_dir = folder / "renders/baseline_replay_latents"
    render(config, gpu=gpu, log_path=folder / "render.log", latent_dir=latent_dir)
    replay = find_render(folder / "renders", "baseline_replay", case["scene"])
    candidate = json.loads(Path(case["candidate_digests"]).read_text())
    latent = torch.load(latent_dir / "latent_0000.pt", map_location="cpu", weights_only=True)
    baseline = {"shape": list(latent.shape), "dtype": str(latent.dtype), "num_chunks": candidate["num_chunks"],
                "chunk_digests": latent_chunk_digests(latent, candidate["num_chunks"])}
    atomic_json(latent_dir / "chunk_digests.json", baseline)
    prefix_report = compare_digests(baseline, candidate, case["chunk_end"])
    decoded = compare_videos(case["archived_baseline"], replay)
    report = {"scene": case["scene"], "gpu": gpu, "replay": str(replay), "prefix": prefix_report,
        "archive_vs_replay": decoded, "original_benchmark_status": "driver_failed",
        "original_A_latent_certified": False, "threshold_changed": False,
        "limitation": "Archived A has no latent evidence. A new replay is not an original-A latent certificate; this diagnostic never reclassifies the failed benchmark scene.",
        "input_hashes": case["input_hashes"], "replay_sha256": digest(replay), "config_sha256": digest(config)}
    verify_hashes(case["input_hashes"])
    atomic_json(folder / "report.json", report)
    return {"report": str(folder / "report.json"), "prefix_identical": prefix_report["identical"],
            "archive_decoded_identical": decoded["identical"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 3])
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("Duplicate GPU assignments")
    generation, output = args.generation.resolve(), args.output.resolve()
    initial = json.loads((generation / "run.json").read_text())
    if not set(args.gpus) <= {initial["render_gpu"], initial["judge_gpu"]}:
        parser.error("Only take over the original generation GPUs")
    pgid = os.getpgid(initial["pid"])
    if pgid != initial["pid"]:
        raise RuntimeError("Original generator must have a dedicated process group")
    cases = discover_cases(generation, initial)
    if not cases:
        raise ValueError("No matching recorded failures")
    verify_hashes(initial["source_hashes"])
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "waiting_for_generation_release", "pid": os.getpid(), "generation": str(generation),
        "original_pgid": pgid, "gpus": args.gpus, "jobs": cases, "started_unix": time.time(),
        "scope": "Failures present at enqueue time only; one diagnostic baseline replay per scene; no repair retries.",
        "script_sha256": digest(__file__), "valid_for_quality_claims": False}
    lock = threading.RLock()
    def save():
        with lock:
            record["updated_unix"] = time.time()
            atomic_json(output / "queue.json", record)
    save()
    from scripts.run_agent_edit_demo import gpu_free_mib
    deadline = time.monotonic() + 48 * 3600

    def worker(gpu):
        while True:
            with lock:
                pending = [case for case in cases if case["status"] == "queued"]
                if not pending:
                    return
            state = json.loads((generation / "run.json").read_text())
            if time.monotonic() > deadline:
                raise TimeoutError("Generation GPU handoff exceeded 48 hours")
            if not eligible(state, group_active(pgid), gpu_free_mib(gpu)):
                time.sleep(20)
                continue
            if shutil.disk_usage(output).free < 50 * 1024**3:
                raise RuntimeError("Less than 50 GiB disk available")
            verify_hashes(initial["source_hashes"])
            with lock:
                pending = [case for case in cases if case["status"] == "queued"]
                if not pending:
                    return
                case = pending[0]
                case.update(status="running", gpu=gpu, started_unix=time.time())
                record["status"] = "running"
                save()
            try:
                result = run_case(case, gpu, output)
                verify_hashes(initial["source_hashes"])
                case.update(status="diagnostic_completed", **result)
            except Exception as error:
                case.update(status="diagnostic_failed", error=f"{type(error).__name__}: {error}")
            finally:
                case["wall_seconds"] = time.time() - case["started_unix"]
                save()
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in args.gpus]
            for future in futures:
                future.result()
        record["status"] = "finished" if all(case["status"] == "diagnostic_completed" for case in cases) else "finished_with_diagnostic_failures"
    except Exception as error:
        record.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
