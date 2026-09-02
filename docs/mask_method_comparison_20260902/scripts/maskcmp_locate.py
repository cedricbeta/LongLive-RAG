"""Identity-conditioned per-entity localisation test (no renders).

For every scripted entity, every shot 1..5, every chunk-first frame: ask Qwen3-VL-8B
"where is THIS entity (reference crop + description) in this frame, even if drifted; present=false if absent".
SAM2 turns the box into an 8-frame track. Output: per-shot overlays (one colour per entity), a table of
presence vs caption-expected presence, box well-formedness (IoU with the nearest plain 'person' box) and
track stability. Uses today's baselines (run2/frames).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maskcmp_build as mb  # noqa: E402
import maskcmp_build2 as mb2  # noqa: E402
from utils.agent_vlm import FingerprintedVLMClient  # noqa: E402

OUT = HERE / "run3"
FRAMES = mb2.OUT / "frames"
FPC = 8

ENTITIES = {
    "ns000_in_the_library_reading": [
        {"name": "Noah", "desc": "tall man with a shaved head and a dark green work shirt", "shots": [0, 1, 2, 3, 4, 5]},
    ],
    "ns100_noah_stands_at_the": [
        {"name": "Noah", "desc": "man in a dark green work shirt taking notes", "shots": [0, 1, 2, 3, 4, 5]},
        {"name": "Iris", "desc": "woman with curly red hair and a moss green coat", "shots": [1, 2]},
    ],
    "ns190_in_the_hotel_lobby": [
        {"name": "Iris", "desc": "middle-aged woman with curly red hair and a moss green coat", "shots": [0, 1, 2, 3, 4, 5]},
        {"name": "Kai", "desc": "older man with silver hair and a brown trench coat", "shots": [1, 2]},
    ],
}
COLORS = [(255, 40, 40), (40, 255, 40), (40, 120, 255)]

LOCATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "present": {"type": "boolean"},
        "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}},
    },
    "required": ["present", "bbox_2d"],
}


def wait_server():
    for _ in range(120):
        try:
            urllib.request.urlopen(mb.VLM_URL + "/models", timeout=5).read()
            return
        except Exception:
            time.sleep(10)
    raise RuntimeError("VLM server not up")


def plain_boxes(client, png: Path, W: int, H: int, phrase: str) -> list[tuple[int, int, int, int]]:
    res = client.call(
        prompt=f"Image is {W}x{H} pixels. Return one bounding box bbox_2d=[x1,y1,x2,y2] for every {phrase} visible. Output strict JSON.",
        image_paths=[png], schema=mb.BOX_SCHEMA, call_kind="loc_plain", shot_index=0, temporal_inputs=(),
    )
    return [mb.scale_box(b["bbox_2d"], W, H, "norm1000") for b in res["parsed"]["boxes"]]


def locate(client, ent: dict, ref_png: Path | None, png: Path, W: int, H: int, shot: int) -> dict:
    prompt = {
        "entity": ent["name"],
        "script_description": ent["desc"],
        "images": ("First image: reference crop of this entity from an earlier accepted shot. Last image: the current frame." if ref_png else "Only image: the current frame."),
        "task": (
            f"In the current frame ({W}x{H} pixels) locate THIS SAME entity — the instance playing this role in the "
            "scene — even if its appearance has drifted (hair, clothing, colours or build may differ from the "
            "reference). Return present=false and a zero box if no instance of this entity is in the frame. Never box a "
            "different character. bbox_2d=[x1,y1,x2,y2], whole body including hair."
        ),
        "output": "strict JSON matching the supplied schema",
    }
    res = client.call(
        prompt=json.dumps(prompt, sort_keys=True), image_paths=[p for p in (ref_png, png) if p], schema=LOCATE_SCHEMA,
        call_kind="loc_entity", shot_index=shot, temporal_inputs=(),
    )
    p = res["parsed"]
    box = mb.scale_box(p["bbox_2d"], W, H, "norm1000") if p["present"] else None
    if box and (box[2] - box[0] < 8 or box[3] - box[1] < 8):
        box = None
    return {"present": bool(box), "box": box}


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def shot_dir(scene: str, shot: int) -> Path:
    d = FRAMES / f"{scene}_shot{shot:02d}"
    d.mkdir(exist_ok=True)
    for i in range(64):
        dst = d / f"{i}.jpg"
        if not dst.exists():
            dst.symlink_to(FRAMES / scene / f"{shot * 64 + i}.jpg")
    return d


def main():
    from PIL import Image, ImageDraw

    wait_server()
    OUT.mkdir(exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)
    (OUT / "refs").mkdir(exist_ok=True)
    client = FingerprintedVLMClient(base_url=mb.VLM_URL, model=mb.VLM_MODEL, cache_dir=OUT / "vlm_cache")
    sam = mb.Sam2Runner("cuda:0")
    table = []
    for scene, ents in ENTITIES.items():
        W, H = Image.open(FRAMES / scene / "0.jpg").size
        # reference crops from shot 0 (plain grounding on the script description)
        for ent in ents:
            ent["ref"] = None
            if 0 in ent["shots"]:
                f0 = FRAMES / scene / "chunk00_first.png"
                boxes = plain_boxes(client, f0, W, H, ent["desc"])
                if boxes:
                    x1, y1, x2, y2 = boxes[0]
                    mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                    crop = Image.open(f0).crop((max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)))
                    ent["ref"] = OUT / "refs" / f"{scene}_{ent['name']}.png"
                    crop.save(ent["ref"])
        for shot in range(1, 6):
            sd = shot_dir(scene, shot)
            tracks = {}
            for ent in ents:
                expected = shot in ent["shots"]
                pres, ious, boxes = [], [], []
                soft = {}
                for c in range(shot * FPC, shot * FPC + FPC):
                    png = FRAMES / scene / f"chunk{c:02d}_first.png"
                    r = locate(client, ent, ent["ref"], png, W, H, shot)
                    pres.append(r["present"])
                    persons = plain_boxes(client, png, W, H, "person")
                    if r["box"]:
                        ious.append(max((iou(r["box"], pb) for pb in persons), default=0.0))
                        boxes.append(r["box"])
                    binm = sam.chunk_masks(sd, c - shot * FPC, [r["box"]] if r["box"] else [], (H, W))
                    soft[c] = mb.soft_from_binary(binm)
                tracks[ent["name"]] = soft
                q = mb.qc(soft)
                row = {
                    "scene": scene, "shot": shot, "entity": ent["name"], "expected": expected,
                    "present_chunks": "".join("1" if p else "0" for p in pres),
                    "n_present": sum(pres), "mean_iou_person": float(np.mean(ious)) if ious else None,
                    "box_stability": float(np.mean([iou(a, b) for a, b in zip(boxes[:-1], boxes[1:])])) if len(boxes) > 1 else None,
                    "coverage": q["coverage_mean"], "temporal_iou": q["temporal_iou"],
                }
                table.append(row)
                print(f"{scene[:5]} s{shot} {ent['name']:5s} exp={int(expected)} present={row['present_chunks']} iou_person={row['mean_iou_person'] and round(row['mean_iou_person'],2)} stab={row['box_stability'] and round(row['box_stability'],2)}", flush=True)
            # overlay: frames 0,16,32,48 with each entity's track tinted
            tiles = []
            for lf in (0, 16, 32, 48):
                c = shot * FPC + lf // FPC
                im = np.asarray(Image.open(sd / f"{lf}.jpg").convert("RGB")).astype(np.float32)
                for ei, ent in enumerate(ents):
                    s = tracks[ent["name"]][c][lf % FPC]
                    up = torch.nn.functional.interpolate(s[None, None], size=(H, W), mode="nearest")[0, 0].numpy() >= 0.35
                    col = np.array(COLORS[ei], dtype=np.float32)
                    im[up] = 0.55 * im[up] + 0.45 * col
                tiles.append(Image.fromarray(im.astype(np.uint8)).resize((W // 2, H // 2)))
            row_im = Image.new("RGB", (4 * (W // 2), H // 2 + 28), "black")
            legend = "  ".join(f"{ent['name']}={['red','green','blue'][i]}{'' if shot in ent['shots'] else '(not in caption)'}" for i, ent in enumerate(ents))
            ImageDraw.Draw(row_im).text((8, 6), f"{scene} shot{shot}  {legend}", fill="white")
            for i, t in enumerate(tiles):
                row_im.paste(t, (i * (W // 2), 28))
            row_im.save(OUT / "overlays" / f"{scene}_shot{shot:02d}.png")
    (OUT / "locate_table.json").write_text(json.dumps(table, indent=2))
    # summary
    exp = [r for r in table if r["expected"]]
    nexp = [r for r in table if not r["expected"]]
    print("\nexpected-present rows:", len(exp), "chunks located:", f"{np.mean([r['n_present'] / 8 for r in exp]):.2f}",
          "mean IoU vs person box:", f"{np.mean([r['mean_iou_person'] for r in exp if r['mean_iou_person'] is not None]):.2f}",
          "box stability:", f"{np.mean([r['box_stability'] for r in exp if r['box_stability'] is not None]):.2f}")
    print("expected-absent rows:", len(nexp), "false-present chunk rate:", f"{np.mean([r['n_present'] / 8 for r in nexp]):.2f}" if nexp else "-")


if __name__ == "__main__":
    main()
