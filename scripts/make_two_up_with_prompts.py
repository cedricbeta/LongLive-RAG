"""Two-up baseline-vs-delivery video with a per-shot prompt band.

The band shows the authored caption and, when the entity judge fired, the repair prompt with the
appended script description highlighted, plus what the commit guard decided (delivered / rolled
back). Reads only a scene's summary.json + ledger.jsonl, so it works for any run of
scripts/run_agent_edit_demo.py.

    python scripts/make_two_up_with_prompts.py <scene_dir> [<scene_dir> ...] [--out-name NAME]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPC = 8  # chunks per shot
FRAMES_PER_CHUNK = 8
FRAMES_PER_SHOT = FPC * FRAMES_PER_CHUNK
PANEL = (640, 352)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(name: str, size: int):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def wrap(draw, text: str, font, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def shot_rows(scene_dir: Path) -> tuple[dict, list[dict]]:
    scene = scene_dir.name
    summary = json.loads((scene_dir / "summary.json").read_text())["scenes"][scene]
    delivered = {int(r["shot"]): r for r in summary["delivered"]}
    n_shots = len(delivered)

    # repairs that survived into the delivered render
    delivered_repair = {
        int(plan["chunk_start"]) // FPC: plan["prompt"]
        for plan in summary.get("final_edit_plans", [])
    }
    # every repair the loop attempted, in order
    attempted: dict[int, list[str]] = {}
    for line in (scene_dir / "ledger.jsonl").open():
        row = json.loads(line)
        if row.get("event") == "action" and row.get("tool") == "edit_shot":
            attempted.setdefault(int(row["shot_index"]), []).append(row["params"]["prompt"])

    rows = []
    for shot in range(n_shots):
        caption = json.loads((scene_dir / "data" / scene / f"{shot}.json").read_text())["caption"]
        rec = delivered.get(shot, {})
        repair = delivered_repair.get(shot)
        tries = attempted.get(shot, [])
        if repair is not None:
            status, colour = f"repair delivered (accepted, lookahead passed)", (120, 255, 140)
        elif tries:
            status = f"{len(tries)} repair attempt(s) -> not delivered"
            if rec.get("quarantined"):
                status += ", shot quarantined"
            repair, colour = tries[0], (255, 150, 150)
        else:
            status, colour = "no repair (judge accepted the shot)", (170, 170, 170)
        rows.append(
            {
                "shot": shot,
                "caption": caption.strip(),
                "repair": (repair or "").strip(),
                "status": status,
                "colour": colour,
            }
        )
    return summary, rows


def band(row: dict, width: int, height: int, scene: str, n_shots: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (16, 16, 18))
    d = ImageDraw.Draw(img)
    f_head = _font("DejaVuSans-Bold.ttf", 17)
    f_lab = _font("DejaVuSans-Bold.ttf", 15)
    f_txt = _font("DejaVuSans.ttf", 15)
    pad, y = 16, 10

    d.text((pad, y), f"{scene}   shot {row['shot']}/{n_shots - 1}", font=f_head, fill=(255, 255, 255))
    w = d.textlength(f"{scene}   shot {row['shot']}/{n_shots - 1}", font=f_head)
    d.text((pad + w + 18, y + 1), row["status"], font=f_lab, fill=row["colour"])
    y += 26

    d.text((pad, y), "prompt", font=f_lab, fill=(150, 190, 255))
    for line in wrap(d, row["caption"], f_txt, width - 2 * pad - 92):
        d.text((pad + 92, y), line, font=f_txt, fill=(226, 226, 226))
        y += 19
    y += 6

    d.text((pad, y), "repair", font=f_lab, fill=(255, 205, 120))
    if row["repair"]:
        # the repair prompt is caption + appended script description: dim the shared prefix so the
        # appended sentence is what the eye lands on
        extra = row["repair"][len(row["caption"]):].strip() if row["repair"].startswith(row["caption"]) else ""
        shown = extra or row["repair"]
        prefix = "(caption) + " if extra else ""
        if prefix:
            d.text((pad + 92, y), prefix, font=f_txt, fill=(120, 120, 120))
        x0 = pad + 92 + (d.textlength(prefix, font=f_txt) if prefix else 0)
        first = True
        for line in wrap(d, shown, f_txt, width - 2 * pad - 92 - (x0 - pad - 92)):
            d.text((x0 if first else pad + 92, y), line, font=f_txt, fill=(255, 220, 150))
            y += 19
            first = False
    else:
        d.text((pad + 92, y), "-", font=f_txt, fill=(120, 120, 120))
    return np.array(img)[:, :, ::-1].copy()


def build(scene_dir: Path, out_name: str) -> Path:
    scene = scene_dir.name
    summary, rows = shot_rows(scene_dir)
    n_shots = len(rows)
    base = cv2.VideoCapture(summary["baseline_video"])
    deliv = cv2.VideoCapture(summary["final_delivery_video"])
    fps = base.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(min(base.get(cv2.CAP_PROP_FRAME_COUNT), deliv.get(cv2.CAP_PROP_FRAME_COUNT)))

    width = PANEL[0] * 2
    head_h, band_h = 30, 190
    f_lab = _font("DejaVuSans-Bold.ttf", 16)
    header = Image.new("RGB", (width, head_h), (16, 16, 18))
    hd = ImageDraw.Draw(header)
    hd.text((16, 6), "baseline (LongLive, no repair)", font=f_lab, fill=(200, 200, 200))
    hd.text((PANEL[0] + 16, 6), "delivery (entity judge + cumulative repair + H=2 lookahead)",
            font=f_lab, fill=(140, 230, 160))
    header_bgr = np.array(header)[:, :, ::-1].copy()
    bands = [band(r, width, band_h, scene, n_shots) for r in rows]

    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (width, head_h + PANEL[1] + band_h))
    for i in range(n):
        ok1, fb = base.read()
        ok2, fd = deliv.read()
        if not (ok1 and ok2):
            break
        left = cv2.resize(fb, PANEL)
        right = cv2.resize(fd, PANEL)
        shot = min(i // FRAMES_PER_SHOT, n_shots - 1)
        vw.write(np.vstack([header_bgr, np.hstack([left, right]), bands[shot]]))
    vw.release()
    base.release()
    deliv.release()

    out = scene_dir / out_name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-preset", "veryfast", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    tmp.unlink()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", nargs="+")
    ap.add_argument("--out-name", default="two_up_prompts.mp4")
    args = ap.parse_args()
    for d in args.scene_dirs:
        out = build(Path(d).resolve(), args.out_name)
        print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
