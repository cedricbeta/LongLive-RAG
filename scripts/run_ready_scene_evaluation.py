#!/usr/bin/env python3
"""Score delivered scenes immediately; retain failed/missing scenes in the population.

Reuses completed A/B/old-C scores and the identical v3 metric/API protocol.
Partial tables are explicitly paired on the same delivered sample IDs, not
compared against all-33 baseline means. No generation retry or threshold change.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_audited_narrastream_eval import atomic_json, digest
from scripts.run_iteration_eval_campaign import (
    FEATURE_METRICS, NOISE_METRICS, METRICS, GROUPS, merge_parts, paired_interval, prepare_current,
)


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aggregate_rows(rows):
    aggregate, coverage = {}, {}
    for metric in METRICS[:11]:
        values = [row.get(metric) for row in rows if valid_number(row.get(metric))]
        aggregate[metric] = sum(values) / len(values) if values else None
        coverage[metric] = {"valid_count": len(values), "total_count": len(rows)}
    for group, metrics in GROUPS.items():
        values = [aggregate[metric] for metric in metrics]
        aggregate[group] = sum(values) / len(values) if all(valid_number(v) for v in values) else None
    values = [aggregate[group] for group in GROUPS]
    aggregate["total_score"] = sum(values) / 3 if all(valid_number(v) for v in values) else None
    aggregate["metric_coverage"] = coverage
    return aggregate


def ready_scenes(state, scheduled):
    completed = {row["scene"] for row in state["results"] if row["status"] == "driver_completed"}
    return [scene for scene in state["scenes"] if scene in completed and scene not in scheduled]


def remap_ids(rows, scene_order):
    indices = {scene: index for index, scene in enumerate(scene_order)}
    return [{**row, "sample_id": indices[row["scene"]]} for row in rows]


def publish(output, archive, generation, batches):
    state = json.loads((generation / "run.json").read_text())
    rows = []
    for batch in batches:
        parts = []
        for lane in ("features", "judged"):
            result = batch.get(lane, {})
            if result.get("returncode") != 0:
                break
            parts.append(json.loads((output / result["name"] / "results_latest.json").read_text()))
        if len(parts) == 2:
            rows.extend(merge_parts(parts)["per_sample"])
    rows.sort(key=lambda row: row["sample_id"])
    ids = {row["sample_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate scored sample IDs")
    statuses = {row["scene"]: row["status"] for row in state["results"]}
    population = [{"sample_id": index, "scene": scene, "generation_status": statuses.get(scene, "not_started"),
                   "scored": index in ids} for index, scene in enumerate(state["scenes"])]
    atomic_json(output / "population.json", {"expected": len(population), "scored": len(ids), "scenes": population})
    current = {"aggregated": aggregate_rows(rows), "per_sample": rows, "expected_population": len(population),
               "scored_count": len(rows), "partial": len(rows) != len(population), "valid_for_quality_claims": False}
    atomic_json(output / "C_current/results_latest.json", current)
    paired = {"C_current": current}
    for arm in ("A", "B", "C_old"):
        baseline = json.loads((archive / arm / "results_latest.json").read_text())
        selected = [row for row in baseline["per_sample"] if row["sample_id"] in ids]
        paired[arm] = {"aggregated": aggregate_rows(selected), "per_sample": selected}
    lines = ["# 当前 C：已交付场景的配对诊断（非完整 benchmark）", "",
             f"已评分 {len(rows)}/{len(population)} 场景。四列使用完全相同的 sample ID；缺失/失败另列在 population.json，未冒充成功。",
             "没有完整覆盖、fallback 与独立裁判审核前，不宣称方法有效。", "",
             "| 指标 | A（配对） | B（配对） | 旧 C（配对） | 当前 C |", "|---|---:|---:|---:|---:|"]
    for metric in METRICS:
        values = [paired[arm]["aggregated"][metric] for arm in ("A", "B", "C_old", "C_current")]
        lines.append("| " + metric + " | " + " | ".join(f"{100*v:.2f}" if valid_number(v) else "—" for v in values) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    atomic_json(output / "comparison.json", {"paired_sample_ids": sorted(ids), "expected_population": len(population),
        "paired_aggregates": {arm: value["aggregated"] for arm, value in paired.items()},
        "C_minus_A": {metric: paired_interval(paired["A"]["per_sample"], rows, metric) for metric in METRICS},
        "valid_for_quality_claims": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--archive-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-gpu", type=int, default=1)
    parser.add_argument("--judge-gpu", type=int, default=2)
    args = parser.parse_args()
    generation, archive, output = args.generation.resolve(), args.archive_evaluation.resolve(), args.output.resolve()
    old = json.loads((archive / "campaign.json").read_text())
    if any(row["status"] == "running" for row in old["jobs"]):
        raise RuntimeError("Previous evaluator still has running metric tasks")
    for arm in ("A", "B", "C_old"):
        for lane in ("features", "judged"):
            audit = json.loads((archive / f"{arm}_{lane}/audit.json").read_text())
            for path, value in audit["code_hashes"].items():
                if digest(path) != value:
                    raise RuntimeError(f"Scorer changed since archived baseline evaluation: {path}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    record = {"status": "starting", "pid": os.getpid(), "generation": str(generation), "archive": str(archive),
              "protocol": "v3_metrics_ready_scene_dispatch", "expected_population": 33, "batches": [],
              "metric_gpu": args.metric_gpu, "judge_gpu": args.judge_gpu, "started_unix": time.time(),
              "failure_policy": "Failed generation scenes stay in population with no score; no implicit baseline substitution/retry.",
              "script_hashes": {str(Path(__file__).resolve()): digest(__file__)}}
    lock = threading.RLock()
    def save():
        with lock:
            record["updated_unix"] = time.time()
            atomic_json(output / "queue.json", record)
    save()
    from scripts.run_agent_edit_demo import start_vlm_server, gpu_free_mib
    server = None
    try:
        if gpu_free_mib(args.metric_gpu) < 75000:
            raise RuntimeError("Metric GPU occupied")
        server, base_url = start_vlm_server(argparse.Namespace(gpu=args.judge_gpu,
            sglang_python="/home/chendong/miniconda3/envs/sglang/bin/python", model="Qwen/Qwen3-VL-8B-Instruct",
            mem_fraction_static=0.45, startup_timeout=600), output)
        record.update(status="running", judge_base_url=base_url)
        save()
        queues = {lane: Queue() for lane in ("features", "judged")}

        def worker(lane):
            while True:
                batch = queues[lane].get()
                if batch is None:
                    return
                name = f"batch{batch['index']:02d}_{lane}"
                row = {"name": name, "status": "running", "started_unix": time.time()}
                with lock:
                    batch[lane] = row
                    save()
                try:
                    if shutil.disk_usage(output).free < 50 * 1024**3:
                        raise RuntimeError("Disk below 50 GiB")
                    if lane == "features" and gpu_free_mib(args.metric_gpu) < 75000:
                        raise RuntimeError("Metric GPU occupied")
                    command = [sys.executable, "-u", str(ROOT / "scripts/run_audited_narrastream_eval.py"),
                        "--eval-data", batch["eval_data"], "--output", str(output / name),
                        "--cache-root", str(output / f"cache_{name}"),
                        "--response-cache", str(archive / "cache_main_api"), "--base-url", base_url,
                        "--expected-count", str(len(batch["scenes"])), "--device", "cuda:0" if lane == "features" else "cpu",
                        "--api-workers", "4", "--metrics", *(FEATURE_METRICS if lane == "features" else NOISE_METRICS)]
                    row["command"] = command
                    save()
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.metric_gpu) if lane == "features" else "",
                           "PYTHONDONTWRITEBYTECODE": "1"}
                    with (output / "logs" / f"{name}.log").open("x") as stream:
                        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    row.update(status="completed" if completed.returncode == 0 else "failed", returncode=completed.returncode)
                    audit = json.loads((output / name / "audit.json").read_text())
                    row["observed_fallback_count"] = len(audit.get("observed_fallbacks", []))
                except Exception as error:
                    row.update(status="failed", returncode=1, error=f"{type(error).__name__}: {error}")
                finally:
                    with lock:
                        row["wall_seconds"] = time.time() - row["started_unix"]
                        publish(output, archive, generation, record["batches"])
                        save()

        scheduled = set()
        deadline = time.monotonic() + 48 * 3600
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, lane) for lane in queues]
            try:
                while True:
                    state = json.loads((generation / "run.json").read_text())
                    ready = ready_scenes(state, scheduled)
                    if ready:
                        index = len(record["batches"])
                        data = prepare_current(generation, output / f"inputs/batch{index:02d}", ready)
                        atomic_json(data, remap_ids(json.loads(data.read_text()), state["scenes"]))
                        batch = {"index": index, "scenes": ready, "eval_data": str(data)}
                        with lock:
                            record["batches"].append(batch)
                            scheduled.update(ready)
                            publish(output, archive, generation, record["batches"])
                            save()
                        for queue in queues.values():
                            queue.put(batch)
                    if state["status"] in {"generation_complete", "generation_finished_with_failures", "stopped"}:
                        record["generation_terminal_status"] = state["status"]
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Generation dispatch exceeded 48 hours")
                    os.kill(state["pid"], 0)
                    time.sleep(20)
            finally:
                for queue in queues.values():
                    queue.put(None)
            for future in futures:
                future.result()
        failed = any(batch.get(lane, {}).get("returncode") != 0 for batch in record["batches"] for lane in queues)
        record["status"] = "finished_with_scoring_failures" if failed else "finished_population_audit_required"
    except Exception as error:
        record.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)


if __name__ == "__main__":
    main()
