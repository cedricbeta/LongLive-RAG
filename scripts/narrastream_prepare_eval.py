#!/usr/bin/env python3
"""Build a NarraStream-Bench ``eval_data.json`` for one arm of the three-arm campaign.

NarraStream's own preprocess splits by ``int(fps * segment_duration)`` frames, which
cannot express our 64-frame (2.67 s) segments, so this script cuts the segments
itself (same mp4v writer as theirs) and writes the same eval_data.json schema.

    # arms A/B: one render per scene under renders/<scene>__<arm>/
    python scripts/narrastream_prepare_eval.py --arm armA_baseline \
        --video_glob 'videos/narrastream_threearm_20260827/renders/{scene}__armA_baseline/*_regular.mp4' \
        --out videos/narrastream_eval/armA_baseline
    # arm C (entity judge): delivery video from each scene's summary.json
    python scripts/narrastream_prepare_eval.py --arm armC_entity \
        --summary_glob 'videos/narrastream_threearm_20260827/armC_entity/g*/{scene}/summary.json' \
        --out videos/narrastream_eval/armC_entity
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "videos/narrastream_threearm_20260827/manifest.json"
DATA = ROOT / "example/narrastream_bench/multitext_subset33_b8"
FPC = 8


def captions(scene: str) -> list[str]:
    d = DATA / scene
    return [json.load(open(d / f"{k}.json"))["caption"] for k in range(len(list(d.glob("[0-9].json"))))]


def durations(scene: str) -> list[int]:
    return [int(v) for v in (DATA / scene / "shot_durations.txt").read_text().split()]


def split(video: Path, out_dir: Path, chunks_per_shot: list[int]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(3)), int(cap.get(4))
    paths = []
    for idx, chunks in enumerate(chunks_per_shot):
        path = out_dir / f"seg_{idx}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        for _ in range(chunks * FPC):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            n += 1
        writer.release()
        if n == 0:
            raise RuntimeError(f"{video}: no frames for segment {idx}")
        paths.append(str(path))
    cap.release()
    return paths


def resolve_video(args, scene: str) -> Path | None:
    if args.summary_glob:
        for s in glob.glob(str(ROOT / args.summary_glob.format(scene=scene))):
            summary = json.load(open(s))["scenes"].get(scene)
            if summary:
                return Path(summary["final_delivery_video"])
        return None
    m = glob.glob(str(ROOT / args.video_glob.format(scene=scene)))
    return Path(m[0]) if len(m) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--video_glob", default=None, help="glob with {scene}; exactly one match per scene")
    ap.add_argument("--summary_glob", default=None, help="glob with {scene} to run_agent_edit_demo summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    scenes = args.scenes or json.load(open(MANIFEST))["scenes"]
    out = ROOT / args.out
    processed = out / "processed"
    eval_data, missing = [], []
    for idx, scene in enumerate(scenes):
        video = resolve_video(args, scene)
        if video is None or not video.exists():
            missing.append(scene)
            continue
        segs = split(video, processed / f"sample_{idx}", durations(scene))
        eval_data.append({"sample_id": idx, "scene": scene, "prompts": captions(scene), "segment_paths": segs, "source_video": str(video)})
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "eval_data.json").write_text(json.dumps(eval_data, indent=2, ensure_ascii=False))
    print(f"{args.arm}: {len(eval_data)} samples -> {processed / 'eval_data.json'}; missing {len(missing)}: {missing[:5]}")


if __name__ == "__main__":
    main()
