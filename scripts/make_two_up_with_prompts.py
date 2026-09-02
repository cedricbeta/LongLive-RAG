#!/usr/bin/env python3
"""Two-up (baseline | delivery) video with a text band below: the current shot's
authored caption and, for shots the agent repaired, the repair prompt.

    python scripts/make_two_up_with_prompts.py videos/agent_entity_20260902/<scene>

Reads <root>/<scene>/{summary.json, ledger.jsonl, data/<scene>/*.json} produced by
scripts/run_agent_edit_demo.py and writes <root>/<scene>/<scene>/two_up_prompts.mp4.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPC = 8
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def shot_ranges(scene_dir: Path) -> list[tuple[int, int]]:
    durations = [int(v) for v in (scene_dir / "shot_durations.txt").read_text().split()]
    out, start = [], 0
    for chunks in durations:
        out.append((start * FPC, (start + chunks) * FPC))
        start += chunks
    return out


def repair_prompts(ledger: Path, delivered: list[dict]) -> dict[int, tuple[int, str]]:
    """shot -> (attempt, prompt) for the repair attempt that was actually delivered."""
    delivered_attempt = {int(r["shot"]): int(r["attempt"]) for r in delivered if r.get("edited")}
    out: dict[int, tuple[int, str]] = {}
    for line in ledger.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "action" and r.get("edit") and r.get("tool") == "edit_shot":
            shot = int(r["shot_index"])
            attempt = int(r.get("edit_count", 0))
            if delivered_attempt.get(shot) == attempt:
                out[shot] = (attempt, str(r["params"]["prompt"]))
    return out


def main(root: Path) -> Path:
    scene = root.name
    summary = json.load(open(root / "summary.json"))["scenes"][scene]
    out_dir = root / scene
    baseline, final = Path(summary["baseline_video"]), Path(summary["final_delivery_video"])
    data_dir = out_dir / "data" / scene
    captions = {int(p.stem): json.load(open(p)).get("caption", "") for p in data_dir.glob("*.json") if p.stem.isdigit()}
    ranges = shot_ranges(data_dir)
    repairs = repair_prompts(out_dir / "ledger.jsonl", summary.get("delivered", []))
    quarantined = set(summary.get("quarantined_shots", []))

    cap_b, cap_f = cv2.VideoCapture(str(baseline)), cv2.VideoCapture(str(final))
    fps = cap_b.get(cv2.CAP_PROP_FPS) or 24
    tw, th = 640, 352
    band_h = 150
    W, H = 2 * tw, th + band_h
    f_title, f_body = font(20), font(17)
    tmp = out_dir / "two_up_prompts.raw.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    idx = 0
    while True:
        ok_b, fb = cap_b.read()
        ok_f, ff = cap_f.read()
        if not (ok_b and ok_f):
            break
        shot = next((k for k, (s, e) in enumerate(ranges) if s <= idx < e), len(ranges) - 1)
        canvas = Image.new("RGB", (W, H), (12, 12, 12))
        canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(fb, (tw, th)), cv2.COLOR_BGR2RGB)), (0, 0))
        canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(ff, (tw, th)), cv2.COLOR_BGR2RGB)), (tw, 0))
        d = ImageDraw.Draw(canvas)
        d.rectangle((0, 0, 130, 28), fill=(0, 0, 0))
        d.text((8, 4), "baseline", font=f_title, fill="white")
        d.rectangle((tw, 0, tw + 260, 28), fill=(0, 0, 0))
        d.text((tw + 8, 4), "agent delivery", font=f_title, fill="white")
        status = "repaired" if shot in repairs else ("quarantined" if shot in quarantined else "accepted")
        y = th + 6
        d.text((10, y), f"shot {shot}  [{status}]", font=f_title, fill=(255, 220, 80))
        y += 26
        for line in textwrap.wrap("prompt: " + captions.get(shot, ""), width=150)[:3]:
            d.text((10, y), line, font=f_body, fill=(230, 230, 230))
            y += 20
        if shot in repairs:
            attempt, prompt = repairs[shot]
            for line in textwrap.wrap(f"repair (round {attempt}): " + prompt, width=150)[:3]:
                d.text((10, y), line, font=f_body, fill=(130, 220, 130))
                y += 20
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        idx += 1
    writer.release()
    out = out_dir / "two_up_prompts.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
    tmp.unlink()
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(main(Path(arg)))
