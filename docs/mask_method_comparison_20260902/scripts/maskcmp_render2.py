"""Render masked-renoise repairs for all four mask methods on today's baselines.
Same repair prompt (caption + judge reference description), sigma 1.0, noise seed 4242 per shot;
only mask_path differs across methods."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

import yaml

REPO = Path("/home/chendong/video-gen/LongLive-RAG")
PILOT = REPO / "videos/narrastream_threearm_20260827/armC_pilot_v4"
HERE = Path(__file__).resolve().parent
RUN = HERE / "run2"
RENDERS = REPO / "videos/maskcmp_20260902/renders2"
METHODS = ["m0_recorded", "m1_short", "m2_box_sam2", "m3_box_direct"]
GPUS = [int(g) for g in os.environ.get("MASKCMP_GPUS", "0,1,2").split(",")]


def jobs():
    items = json.loads((RUN / "testset.json").read_text())
    for it in items:
        scene, shot = it["scene"], it["shot"]
        template = sorted((PILOT / scene / "configs").glob("edit_shot0*.yaml"))[0]
        base = yaml.safe_load(template.read_text())
        for m in METHODS:
            mask = RUN / "masks" / f"{scene}_shot{shot:02d}_{m}.pt"
            if not mask.exists():
                print("missing mask", mask)
                continue
            prefix = f"edit_shot{shot:02d}_{m}"
            out_dir = RENDERS / scene
            if list(out_dir.glob(f"{prefix}-rank*_regular.mp4")):
                continue
            cfg = json.loads(json.dumps(base))
            cfg["output_folder"] = str(out_dir)
            cfg["inference"]["output_folder"] = str(out_dir)
            cfg["inference"]["filename_prefix"] = prefix
            er = cfg["inference"]["edit_renoise"]
            er.update(
                {
                    "enabled": True,
                    "chunk_start": shot * 8,
                    "chunk_end": shot * 8 + 7,
                    "entry_sigma": 1.0,
                    "mask_path": str(mask),
                    "prompt": it["repair_prompt"],
                    "noise_seed": 4242,
                }
            )
            cfg_path = RENDERS.parent / "configs2" / f"{scene}__{prefix}.yaml"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            yield scene, prefix, cfg_path


def worker(gpu: int, q: Queue):
    while True:
        job = q.get()
        if job is None:
            return
        scene, prefix, cfg_path = job
        log = RENDERS.parent / "logs2" / f"{scene}__{prefix}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
        print(f"[gpu{gpu}] start {scene} {prefix}", flush=True)
        with log.open("w") as fh:
            rc = subprocess.run([sys.executable, "inference.py", "--config_path", str(cfg_path)], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[gpu{gpu}] done rc={rc} {scene} {prefix}", flush=True)
        q.task_done()


def main():
    q: Queue = Queue()
    n = 0
    for job in jobs():
        q.put(job)
        n += 1
    print(f"{n} renders queued on GPUs {GPUS}", flush=True)
    threads = [threading.Thread(target=worker, args=(g, q), daemon=True) for g in GPUS]
    for t in threads:
        t.start()
    q.join()
    for _ in threads:
        q.put(None)
    print("all renders done", flush=True)


if __name__ == "__main__":
    main()
