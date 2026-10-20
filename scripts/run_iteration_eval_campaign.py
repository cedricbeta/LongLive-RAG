#!/usr/bin/env python3
"""Independent evaluation queue for iteration33; no generator/controller changes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_audited_narrastream_eval import atomic_json, digest

NOISE_INDICES = [0, 6, 12, 18, 24, 30]
NOISE_METRICS = ["conditional_adjacent", "conditional_longrange", "entity_grounding", "vlm_score"]
METRICS = ["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness",
           "vtss", "boundary_smoothness", *NOISE_METRICS[:2], "entity_grounding", "dynamic_trajectory", "vlm_score",
           "quality_score", "temporal_score", "instruction_score", "total_score"]
FEATURE_METRICS = [name for name in METRICS[:11] if name not in NOISE_METRICS]
GROUPS = {"quality_score": METRICS[:5], "temporal_score": METRICS[5:8],
          "instruction_score": METRICS[8:11]}


def merge_parts(parts):
    """Merge disjoint metrics, never average partial group scores from workers."""
    if len(parts) != 2:
        raise ValueError("Both feature and judged metric parts are required")
    expected_ids = {row["sample_id"] for row in parts[0]["per_sample"]}
    rows = {key: {"sample_id": key} for key in expected_ids}
    aggregate, coverage = {}, {}
    for part in parts:
        if {row["sample_id"] for row in part["per_sample"]} != expected_ids:
            raise ValueError("Part sample IDs do not match")
        for metric in METRICS[:11]:
            if metric in part["aggregated"]:
                if metric in aggregate:
                    raise ValueError(f"Overlapping metric: {metric}")
                aggregate[metric] = part["aggregated"][metric]
                coverage[metric] = part["aggregated"]["metric_coverage"][metric]
        for row in part["per_sample"]:
            rows[row["sample_id"]].update({k: v for k, v in row.items() if k in METRICS[:11]})
    if set(aggregate) != set(METRICS[:11]):
        raise ValueError("Incomplete metric union")
    for target in [aggregate, *rows.values()]:
        for group, metrics in GROUPS.items():
            values = [target.get(metric) for metric in metrics]
            target[group] = sum(values) / len(values) if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None
        values = [target[group] for group in GROUPS]
        target["total_score"] = sum(values) / 3 if all(v is not None for v in values) else None
    aggregate["metric_coverage"] = coverage
    return {"aggregated": aggregate, "per_sample": [rows[key] for key in sorted(rows)],
            "protocol": "iteration8b_validated_cache_parallel_v3", "group_rule": "equal mean of complete metric groups; missing groups remain null"}


def validate_samples(samples, expected_scenes):
    import cv2
    if [row["scene"] for row in samples] != expected_scenes:
        raise ValueError("Missing/reordered scenes; refusing a reduced denominator")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample ID")
    for row in samples:
        if len(row["prompts"]) != 6 or len(row["segment_paths"]) != 6:
            raise ValueError(f"Invalid prompt/segment count: {row['scene']}")
        for path in [row["source_video"], *row["segment_paths"]]:
            cap = cv2.VideoCapture(str(path))
            expected = 384 if path == row["source_video"] else 64
            good = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if not good or abs(fps - 24.0) > 0.01:
                raise ValueError(f"Unexpected frames/FPS: {path}")


def prepare_current(generation, output, scenes):
    import cv2
    samples = []
    for index, scene in enumerate(scenes):
        path = generation / "armC_current" / scene / "summary.json"
        summary = json.loads(path.read_text())["scenes"][scene]
        video = Path(summary["final_delivery_video"])
        data = generation / "frozen/data" / scene
        prompts = [json.loads((data / f"{shot}.json").read_text())["caption"] for shot in range(6)]
        durations = [int(v) for v in (data / "shot_durations.txt").read_text().split()]
        if durations != [8] * 6:
            raise ValueError(f"Unexpected frozen chunk durations: {scene}")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 384:
            raise ValueError(f"Incomplete delivery video: {scene}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        folder = output / "segments" / scene
        folder.mkdir(parents=True, exist_ok=False)
        paths = []
        try:
            for shot in range(6):
                segment = folder / f"seg_{shot}.mp4"
                writer = cv2.VideoWriter(str(segment), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Segment encoder unavailable")
                try:
                    for _ in range(64):
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError(f"Truncated delivery video: {scene}")
                        writer.write(frame)
                finally:
                    writer.release()
                paths.append(str(segment))
            if cap.read()[0]:
                raise ValueError(f"Unexpected trailing frames: {scene}")
        finally:
            cap.release()
        samples.append({"sample_id": index, "scene": scene, "prompts": prompts,
                        "source_video": str(video), "segment_paths": paths})
    validate_samples(samples, scenes)
    atomic_json(output / "C_current.json", samples)
    return output / "C_current.json"


def paired_interval(a_rows, b_rows, metric):
    a = {row["sample_id"]: row.get(metric) for row in a_rows}
    b = {row["sample_id"]: row.get(metric) for row in b_rows}
    def valid(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    differences = [b[key] - a[key] for key in sorted(a.keys() & b.keys()) if valid(a[key]) and valid(b[key])]
    if not differences:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(0)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
    return {"n": len(differences), "mean": sum(differences) / len(differences), "ci95": [means[50], means[1949]]}


def write_comparison(output, jobs):
    results = {}
    for job in jobs:
        result_path = output / job["name"] / "results_latest.json"
        if job.get("returncode") == 0 and result_path.is_file():
            results[job["name"]] = json.loads(result_path.read_text())
    arms = [name for name in ("A", "B", "C_old", "C_current") if name in results]
    lines = ["# 33 场景迭代评估（非论文正式成绩）", "", "分数乘以 100；失败、缺失和裁判波动需结合 audit.json / campaign.json 阅读。",
             "这不是人工真值；未认证所有静默 fallback。", "", "| 指标 | " + " | ".join(arms) + " |",
             "|---|" + "---:|" * len(arms)]
    for metric in METRICS:
        values = [results[arm]["aggregated"].get(metric) for arm in arms]
        lines.append("| " + metric + " | " + " | ".join(f"{100*v:.2f}" if isinstance(v, (int, float)) else "—" for v in values) + " |")
    comparisons = {}
    if "A" in results:
        for arm in arms:
            if arm != "A":
                comparisons[arm] = {metric: paired_interval(results["A"]["per_sample"], results[arm]["per_sample"], metric) for metric in METRICS}
    noise = {}
    for metric in NOISE_METRICS:
        repetitions = [results[name]["per_sample"] for name in ("noise_0", "noise_1", "noise_2") if name in results]
        if len(repetitions) == 3:
            lookup = [{row["sample_id"]: row.get(metric) for row in rows} for rows in repetitions]
            ranges = []
            for sample_id in NOISE_INDICES:
                values = [row.get(sample_id) for row in lookup]
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                    ranges.append({"sample_id": sample_id, "values": values, "range": max(values)-min(values)})
            noise[metric] = ranges
    atomic_json(output / "comparison.json", {"arms": arms, "paired_bootstrap": comparisons, "noise_repeats": noise,
                                            "ci_limitation": "Scene-resampling uncertainty at seed 0; not multi-generation-seed uncertainty."})
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-gpu", type=int, default=2)
    parser.add_argument("--metric-gpu", type=int, default=1)
    args = parser.parse_args()
    if args.judge_gpu == args.metric_gpu:
        parser.error("Separate judge and metric GPUs required")
    output, generation = args.output.resolve(), args.generation.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "logs").mkdir()
    manifest = json.loads((generation / "run.json").read_text())
    if args.judge_gpu in {manifest["judge_gpu"], manifest["render_gpu"]} or args.metric_gpu in {manifest["judge_gpu"], manifest["render_gpu"]}:
        raise ValueError("Evaluator GPUs overlap the generation job")
    scenes = manifest["scenes"]
    inputs = {}
    for name, archive in [("A", "armA_baseline"), ("B", "armB_concat"), ("C_old", "armC_entity")]:
        samples = json.loads((ROOT / f"videos/narrastream_eval/{archive}/processed/eval_data.json").read_text())
        validate_samples(samples, scenes)
        inputs[name] = output / "inputs" / f"{name}.json"
        atomic_json(inputs[name], samples)
    baseline = json.loads(inputs["A"].read_text())
    atomic_json(output / "inputs/noise.json", [baseline[index] for index in NOISE_INDICES])
    hashes = {str(path): digest(path) for path in [Path(__file__), ROOT / "scripts/run_audited_narrastream_eval.py"]}
    record = {"status": "prepared", "pid": os.getpid(), "started_unix": time.time(), "generation": str(generation),
              "noise_indices": NOISE_INDICES, "noise_repetitions": 3, "noise_metrics": NOISE_METRICS,
              "controller_is_independent": False, "evaluation_does_not_read_controller_verdicts": True,
              "judge_limitation": "Separate scoring process/cache, but same 8B model family as controller; no independent model-family validation.",
              "script_hashes": hashes, "judge_gpu": args.judge_gpu, "metric_gpu": args.metric_gpu, "jobs": []}
    record["protocol"] = "iteration8b_validated_cache_parallel_v3"
    record["queue"] = {
        "features_gpu": [f"{arm}_features" for arm in ("A", "B", "C_old", "C_current")],
        "judged_cpu_with_vlm_gpu": [f"noise_{i}" for i in range(3)] + [f"{arm}_judged" for arm in ("A", "B", "C_old", "C_current")],
    }
    record["queue_policy"] = "Independent failed tasks are recorded and other tasks continue; no automatic semantic/config changes. C waits for complete generation."
    atomic_json(output / "campaign.json", record)
    from scripts.run_agent_edit_demo import start_vlm_server, gpu_free_mib
    server = None
    try:
        if gpu_free_mib(args.metric_gpu) < 75000:
            raise RuntimeError("Metric GPU occupied; refusing to compete")
        server, base_url = start_vlm_server(argparse.Namespace(gpu=args.judge_gpu,
            sglang_python="/home/chendong/miniconda3/envs/sglang/bin/python", model="Qwen/Qwen3-VL-8B-Instruct",
            mem_fraction_static=0.45, startup_timeout=600), output)
        record.update(status="running", judge_base_url=base_url)
        atomic_json(output / "campaign.json", record)
        record_lock = threading.RLock()
        current_input_lock = threading.Lock()
        current_input = []

        def persist():
            with record_lock:
                record["updated_unix"] = time.time()
                atomic_json(output / "campaign.json", record)

        def maybe_merge():
            # Called under record_lock so concurrent workers cannot publish twice.
            jobs = {row["name"]: row for row in record["jobs"]}
            for arm in ("A", "B", "C_old", "C_current"):
                names = [f"{arm}_features", f"{arm}_judged"]
                if arm in jobs or not all(jobs.get(name, {}).get("returncode") == 0 for name in names):
                    continue
                parts = [json.loads((output / name / "results_latest.json").read_text()) for name in names]
                merged = merge_parts(parts)
                atomic_json(output / arm / "results_latest.json", merged)
                audits = {name: json.loads((output / name / "audit.json").read_text()) for name in names}
                atomic_json(output / arm / "audit.json", {"parts": audits, "valid_for_quality_claims": False})
                record["jobs"].append({"name": arm, "status": "merged_audit_required", "returncode": 0,
                                       "observed_fallback_count": sum(len(audit.get("observed_fallbacks", [])) for audit in audits.values())})
            write_comparison(output, record["jobs"])

        def evaluate(name, data, count, cache, metrics, device):
            if any(digest(path) != value for path, value in hashes.items()):
                raise RuntimeError("Evaluation scripts changed during campaign")
            if (device != "cpu" and gpu_free_mib(args.metric_gpu) < 75000) or shutil.disk_usage(output).free < 50 * 1024**3:
                raise RuntimeError("Metric GPU occupied or disk below 50 GiB")
            command = [sys.executable, "-u", str(ROOT / "scripts/run_audited_narrastream_eval.py"),
                "--eval-data", str(data), "--output", str(output / name), "--cache-root", str(cache),
                "--base-url", base_url, "--expected-count", str(count), "--device", device,
                "--api-workers", "4", "--metrics", *metrics]
            if not name.startswith("noise_"):
                command += ["--response-cache", str(output / "cache_main_api")]
            row = {"name": name, "status": "running", "command": command, "started_unix": time.time()}
            with record_lock:
                record["jobs"].append(row)
                persist()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else str(args.metric_gpu), "PYTHONDONTWRITEBYTECODE": "1"}
            with (output / "logs" / f"{name}.log").open("x") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            with record_lock:
                row.update(returncode=completed.returncode, status="completed" if completed.returncode == 0 else "failed",
                           wall_seconds=time.time() - row["started_unix"])
                audit_path = output / name / "audit.json"
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    row["observed_fallback_count"] = len(audit.get("observed_fallbacks", []))
                    row["missing_coverage"] = audit.get("missing_coverage", {})
                persist()
                maybe_merge()
                persist()

        def await_current(lane):
            with record_lock:
                record.setdefault("waiting_lanes", []).append(lane)
                persist()
            deadline = time.monotonic() + 7 * 24 * 3600
            while time.monotonic() < deadline:
                state = json.loads((generation / "run.json").read_text())
                if state["status"] == "generation_complete":
                    if len(state["results"]) != 33 or any(row["status"] != "driver_completed" for row in state["results"]):
                        raise RuntimeError("Generation completion/coverage mismatch")
                    break
                if state["status"] in {"stopped", "generation_finished_with_failures"}:
                    raise RuntimeError("Generation stopped/failed; no scenes omitted from C denominator")
                try:
                    os.kill(state["pid"], 0)
                except ProcessLookupError:
                    raise RuntimeError("Generation supervisor is no longer alive")
                time.sleep(20)
            else:
                raise TimeoutError("Generation wait exceeded seven days")
            with current_input_lock:
                if not current_input:
                    current_input.append(prepare_current(generation, output / "inputs", scenes))
            return current_input[0]

        def lane(kind):
            try:
                device = "cuda:0" if kind == "features" else "cpu"
                metrics = FEATURE_METRICS if kind == "features" else NOISE_METRICS
                if kind == "judged":
                    for repeat in range(3):
                        evaluate(f"noise_{repeat}", output / "inputs/noise.json", 6,
                                 output / f"cache_noise_{repeat}", NOISE_METRICS, device)
                for name in ("A", "B", "C_old"):
                    evaluate(f"{name}_{kind}", inputs[name], 33, output / f"cache_{name}_{kind}", metrics, device)
                data = await_current(kind)
                evaluate(f"C_current_{kind}", data, 33, output / f"cache_C_current_{kind}", metrics, device)
            except Exception as error:
                with record_lock:
                    record.setdefault("lane_errors", {})[kind] = f"{type(error).__name__}: {error}"
                    persist()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, kind) for kind in ("features", "judged")]
            for future in futures:
                future.result()
        bad = record.get("lane_errors") or any(row.get("returncode", 0) != 0 for row in record["jobs"])
        record["status"] = "iteration_finished_with_failures" if bad else "iteration_evaluation_completed_audit_required"
    except Exception as error:
        record.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["updated_unix"] = time.time()
        atomic_json(output / "campaign.json", record)
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)


if __name__ == "__main__":
    main()
