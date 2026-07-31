"""Build visual comparison artifacts for the three-arm run.

Per prompt:
  - strip PNG: rows = arms, columns = frames sampled evenly over the video
  - stacked MP4: native / latentmem / vlm_guided vertically stacked
"""
import argparse
import os
import subprocess

import torch
from PIL import Image, ImageDraw
from torchvision.io import read_video

ARMS = ["native", "latentmem", "vlm_guided", "vlm_agent"]
LABELS = {
    "native": "native (sliding window)",
    "latentmem": "LongLive-RAG (latentmem)",
    "vlm_guided": "LongLive-RAG + VLM guide (ours)",
    "vlm_agent": "LongLive-RAG + VLM guide + retry (ours)",
}

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="arms_out")
parser.add_argument("--prompt_indices", default="0,1,2,6")
parser.add_argument("--cols", type=int, default=8)
parser.add_argument("--out", default="arms_out/compare")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

for pi in [int(i) for i in args.prompt_indices.split(",")]:
    paths = {a: os.path.join(args.root, a, f"prompt{pi:03d}.mp4") for a in ARMS}
    if not all(os.path.exists(p) for p in paths.values()):
        print(f"prompt {pi}: missing videos, skip")
        continue

    vids = {}
    for a, p in paths.items():
        v, _, _ = read_video(p, pts_unit="sec", output_format="TCHW")
        vids[a] = v  # uint8 [T, C, H, W]

    T = min(v.shape[0] for v in vids.values())
    idxs = [round(i * (T - 1) / (args.cols - 1)) for i in range(args.cols)]

    thumb_w = 200
    c, h, w = vids[ARMS[0]].shape[1:]
    thumb_h = int(h * thumb_w / w)
    pad, label_h = 2, 18
    W = args.cols * (thumb_w + pad) + pad
    H = len(ARMS) * (thumb_h + pad + label_h) + pad + label_h
    canvas = Image.new("RGB", (W, H), "black")
    draw = ImageDraw.Draw(canvas)

    for ci, fi in enumerate(idxs):
        draw.text((pad + ci * (thumb_w + pad) + 4, 3), f"t={fi/16:.1f}s", fill="white")
    for ri, a in enumerate(ARMS):
        y0 = label_h + pad + ri * (thumb_h + pad + label_h)
        draw.text((4, y0 + 2), LABELS[a], fill="yellow")
        for ci, fi in enumerate(idxs):
            fr = vids[a][fi].permute(1, 2, 0).numpy()
            img = Image.fromarray(fr).resize((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(img, (pad + ci * (thumb_w + pad), y0 + label_h))

    png = os.path.join(args.out, f"prompt{pi:03d}_strip.png")
    canvas.save(png)
    print(f"prompt {pi}: strip -> {png}")

    stacked = os.path.join(args.out, f"prompt{pi:03d}_stacked.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for a in ARMS:
        cmd += ["-i", paths[a]]
    filt = "".join(
        f"[{i}:v]drawtext=text='{LABELS[a]}':x=8:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.5[v{i}];"
        for i, a in enumerate(ARMS)
    ) + "".join(f"[v{i}]" for i in range(len(ARMS))) + f"vstack={len(ARMS)}[out]"
    cmd += ["-filter_complex", filt, "-map", "[out]", "-c:v", "libx264", "-crf", "18", stacked]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        filt = "".join(f"[{i}:v]" for i in range(len(ARMS))) + f"vstack={len(ARMS)}[out]"
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for a in ARMS:
            cmd += ["-i", paths[a]]
        cmd += ["-filter_complex", filt, "-map", "[out]", "-c:v", "libx264", "-crf", "18", stacked]
        subprocess.run(cmd, check=True)
        print("  (drawtext unavailable; stacked video order: native / latentmem / vlm_guided)")
    print(f"prompt {pi}: stacked -> {stacked}")
