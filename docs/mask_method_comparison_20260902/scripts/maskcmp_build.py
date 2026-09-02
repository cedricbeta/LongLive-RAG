"""Mask-method comparison on the drifted shots of the arm-C v4 pilots.

M0 recorded : judge's free-form repaint_region (as logged) -> CLIPSeg
M1 short    : judge re-asked with a constrained schema (1-3 short noun phrases) -> CLIPSeg union
M2 box_sam2 : judge returns bboxes per chunk (Qwen3-VL grounding) -> SAM2 video propagation

Outputs under OUT: masks/<scene>_shot<k>_<method>.pt (edit_renoise payload format),
overlays/<scene>_shot<k>.png (rows = methods), qc.json, testset.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/chendong/video-gen/LongLive-RAG")
sys.path.insert(0, str(REPO))
PILOT = REPO / "videos/narrastream_threearm_20260827/armC_pilot_v4"
OUT = Path(__file__).resolve().parent / "run"
FPC = 8
GRID = (22, 40)
VLM_URL = "http://127.0.0.1:53456/v1"
VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SCENES = {
    "ns000_in_the_library_reading": {"crop": "man"},
    "ns100_noah_stands_at_the": {"crop": "man"},
    "ns190_in_the_hotel_lobby": {"crop": "woman"},
}

from utils.agent_edit import clipseg_soft_masks, read_video_frames, save_mask_payload  # noqa: E402
from utils.agent_vlm import FingerprintedVLMClient  # noqa: E402

SHORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "subject_drifted": {"type": "boolean"},
        "repaint_phrases": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": {"type": "string", "minLength": 0, "maxLength": 30, "pattern": "^[a-z][a-z -]*$"},
        },
    },
    "required": ["subject_drifted", "repaint_phrases"],
}

BOX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "boxes": {
            "type": "array",
            "minItems": 0,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "maxLength": 40},
                    "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}},
                },
                "required": ["label", "bbox_2d"],
            },
        }
    },
    "required": ["boxes"],
}


def baseline_video(scene: str) -> Path:
    return PILOT / scene / "renders" / f"baseline-rank0-{scene}-seed0_regular.mp4"


def load_testset() -> list[dict]:
    items = []
    for scene in SCENES:
        ledger = PILOT / scene / "ledger.jsonl"
        per_shot: dict[int, dict] = {}
        for line in open(ledger):
            r = json.loads(line)
            if r.get("event") == "verifier" and r.get("attempt") == 0 and not r.get("accept"):
                m = r.get("metrics", {})
                per_shot[int(r["shot_index"])] = {
                    "recorded_phrase": m.get("repaint_region", ""),
                    "reference_description": m.get("reference_description", ""),
                    "drift_description": m.get("drift_description", ""),
                }
        for shot, info in sorted(per_shot.items()):
            caption = json.loads((PILOT / scene / "data" / scene / f"{shot}.json").read_text()).get("caption", "")
            items.append(
                {
                    "scene": scene,
                    "shot": shot,
                    "chunk_start": shot * FPC,
                    "chunk_end": shot * FPC + FPC - 1,
                    "caption": caption,
                    "crop_phrase": SCENES[scene]["crop"],
                    **info,
                }
            )
    return items


def ref_frames(scene: str) -> list[Path]:
    return sorted((PILOT / scene / "attempts").glob("shot00_attempt0_f*_idx*.png"))


def cur_frames(scene: str, shot: int) -> list[Path]:
    return sorted((PILOT / scene / "attempts").glob(f"shot{shot:02d}_attempt0_f*_idx*.png"))


def dump_shot_frames(scene: str, shot: int) -> Path:
    """Write the 64 frames of the shot as JPEG 0..63 (SAM2 video input) + PNG per chunk-first frame."""
    d = OUT / "frames" / f"{scene}_shot{shot:02d}"
    if (d / "63.jpg").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    frames = read_video_frames(baseline_video(scene), shot * FPC * FPC, FPC * FPC)
    for i, im in enumerate(frames):
        im.save(d / f"{i}.jpg", quality=95)
        if i % FPC == 0:
            im.save(d / f"chunk{shot * FPC + i // FPC:02d}_first.png")
    return d


def soft_from_binary(masks_hw: np.ndarray) -> torch.Tensor:
    """[F,H,W] binary -> [F,22,40] area-mean soft mask."""
    t = torch.from_numpy(masks_hw.astype(np.float32)).unsqueeze(1)
    return torch.nn.functional.adaptive_avg_pool2d(t, GRID).squeeze(1)


def judge_prompt_common(item: dict) -> dict:
    return {
        "role": "comparative subject-drift judge",
        "rule": (
            "The first images are REFERENCE frames showing the subject as first established in this "
            "scene; the remaining images are the CURRENT provisional shot. Judge ONLY whether an "
            "ESTABLISHED character's identity or appearance changed relative to the reference, or a "
            "character was replaced or duplicated. Characters the current caption introduces or removes "
            "are NOT drift. Ignore pose, camera, framing, lighting."
        ),
        "current_caption": item["caption"],
        "subject": item["crop_phrase"],
    }


def m1_short_phrases(client: FingerprintedVLMClient, item: dict) -> dict:
    prompt = dict(judge_prompt_common(item))
    prompt["repaint_phrases"] = (
        "List 1-3 SHORT noun phrases (1-4 lowercase words each, e.g. 'person', 'woman', "
        "'red-haired woman', 'man in green shirt') naming what in the CURRENT frames must be erased "
        "and repainted: the drifted subject plus any person/object the caption does not call for. "
        "Each phrase must be something a text-prompted segmentation model can find. Never include "
        "characters the caption legitimately places in the shot. Empty list if not drifted."
    )
    prompt["output"] = "strict JSON matching the supplied schema"
    res = client.call(
        prompt=json.dumps(prompt, sort_keys=True),
        image_paths=[*ref_frames(item["scene"]), *cur_frames(item["scene"], item["shot"])],
        schema=SHORT_SCHEMA,
        call_kind="maskcmp_short",
        shot_index=item["shot"],
        temporal_inputs=(),
    )
    return res["parsed"]


def m2_boxes(client: FingerprintedVLMClient, item: dict, frame_png: Path, width: int, height: int, phrases: list[str]) -> dict:
    """Pure grounding: box every instance of the judge-chosen phrases in the current frame."""
    prompt = (
        f"Image is {width}x{height} pixels. Return one bounding box bbox_2d=[x1,y1,x2,y2] for EVERY "
        f"instance of each of these targets visible in the image: {phrases}. Include the whole body and "
        "hair of a person. Empty list if none is visible. Output strict JSON matching the supplied schema."
    )
    res = client.call(
        prompt=prompt,
        image_paths=[frame_png],
        schema=BOX_SCHEMA,
        call_kind="maskcmp_ground",
        shot_index=item["shot"],
        temporal_inputs=(),
    )
    return res["parsed"]


def m3_boxes_direct(client: FingerprintedVLMClient, item: dict, frame_png: Path, width: int, height: int) -> dict:
    """VLM points directly: sees reference frames + the current frame + the judge's drift finding."""
    prompt = json.dumps(
        {
            **judge_prompt_common(item),
            "judge_finding": item.get("drift_description", ""),
            "reference_subject": item.get("reference_description", ""),
            "task": (
                f"The LAST image is one frame of the CURRENT shot ({width}x{height} pixels). The shot was already "
                "judged DRIFTED (see judge_finding). Return a bounding box for EVERY region that must be erased "
                "and repainted so the shot matches the reference: the drifted/replaced subject (whole body and "
                "hair) plus any person or object the current caption does not call for. Never box characters "
                "the caption legitimately places in the shot. bbox_2d=[x1,y1,x2,y2]."
            ),
            "output": "strict JSON matching the supplied schema",
        },
        sort_keys=True,
    )
    res = client.call(
        prompt=prompt,
        image_paths=[*ref_frames(item["scene"]), frame_png],
        schema=BOX_SCHEMA,
        call_kind="maskcmp_box_direct",
        shot_index=item["shot"],
        temporal_inputs=(),
    )
    return res["parsed"]


def scale_box(box: list[int], width: int, height: int, convention: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if convention == "norm1000":
        x1, x2 = x1 / 1000 * width, x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    return int(x1), int(y1), int(x2), int(y2)


class Sam2Runner:
    def __init__(self, device: str):
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self.pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
        self.device = device

    def chunk_masks(self, frames_dir: Path, chunk_local: int, boxes: list[tuple[int, int, int, int]], hw: tuple[int, int]) -> np.ndarray:
        """Binary [8,H,W] union mask for the 8 frames of local chunk `chunk_local` (0..7)."""
        H, W = hw
        if not boxes:
            return np.zeros((FPC, H, W), dtype=bool)
        tmp = frames_dir / f"_chunk{chunk_local}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        for i in range(FPC):
            os.symlink(frames_dir / f"{chunk_local * FPC + i}.jpg", tmp / f"{i}.jpg")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.pred.init_state(video_path=str(tmp))
            for oid, box in enumerate(boxes):
                self.pred.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(box, dtype=np.float32))
            out = np.zeros((FPC, H, W), dtype=bool)
            for fidx, obj_ids, logits in self.pred.propagate_in_video(state):
                m = (logits > 0).any(dim=0)[0].cpu().numpy()
                out[fidx] |= m
            self.pred.reset_state(state)
        shutil.rmtree(tmp)
        return out


def overlay_grid(frames_dir: Path, soft_by_chunk: dict[int, torch.Tensor], chunk_start: int, hw: tuple[int, int], label: str):
    """Return a PIL row: frames 0,16,32,48 of the shot with mask (>=0.35) tinted red."""
    from PIL import Image, ImageDraw

    H, W = hw
    tiles = []
    for local_f in (0, 16, 32, 48):
        chunk = chunk_start + local_f // FPC
        im = Image.open(frames_dir / f"{local_f}.jpg").convert("RGB")
        soft = soft_by_chunk[chunk][local_f % FPC]
        up = torch.nn.functional.interpolate(soft[None, None], size=(H, W), mode="nearest")[0, 0].numpy()
        arr = np.asarray(im).astype(np.float32)
        red = np.zeros_like(arr)
        red[..., 0] = 255
        sel = up >= 0.35
        arr[sel] = 0.55 * arr[sel] + 0.45 * red[sel]
        tile = Image.fromarray(arr.astype(np.uint8)).resize((W // 2, H // 2))
        tiles.append(tile)
    row = Image.new("RGB", (len(tiles) * (W // 2), H // 2 + 28), "black")
    ImageDraw.Draw(row).text((8, 6), label, fill="white")
    for i, t in enumerate(tiles):
        row.paste(t, (i * (W // 2), 28))
    return row


def qc(soft_by_chunk: dict[int, torch.Tensor]) -> dict:
    cov = {int(c): float((s >= 0.35).float().mean()) for c, s in soft_by_chunk.items()}
    binmasks = torch.cat([(soft_by_chunk[c] >= 0.35) for c in sorted(soft_by_chunk)], 0)
    ious = []
    for a, b in zip(binmasks[:-1], binmasks[1:]):
        u = (a | b).sum().item()
        ious.append((a & b).sum().item() / u if u else 1.0)
    return {
        "coverage": cov,
        "coverage_mean": float(np.mean(list(cov.values()))),
        "zero_chunks": int(sum(1 for v in cov.values() if v < 0.005)),
        "temporal_iou": float(np.mean(ious)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true", help="only run one box call and draw it")
    ap.add_argument("--convention", default="auto", choices=["auto", "abs", "norm1000"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    items = load_testset()
    (OUT / "testset.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"{len(items)} drifted shots")
    client = FingerprintedVLMClient(base_url=VLM_URL, model=VLM_MODEL, cache_dir=OUT / "vlm_cache")
    from PIL import Image, ImageDraw

    if args.calib:
        item = items[2]  # ns100 shot1: two people, Iris legit, Noah drifted
        fd = dump_shot_frames(item["scene"], item["shot"])
        png = fd / f"chunk{item['chunk_start']:02d}_first.png"
        W, H = Image.open(png).size
        parsed = m2_boxes(client, item, png, W, H, ['person'])
        print(json.dumps(parsed, indent=2))
        im = Image.open(png).convert("RGB")
        dr = ImageDraw.Draw(im)
        for b in parsed["boxes"]:
            for conv, col in (("abs", "red"), ("norm1000", "lime")):
                dr.rectangle(scale_box(b["bbox_2d"], W, H, conv), outline=col, width=4)
        im.save(OUT / "calib_boxes.png")
        print("wrote", OUT / "calib_boxes.png", "(red=abs pixels, green=0-1000 normalized)")
        return

    sam = Sam2Runner(args.device)
    qc_all = {}
    for item in items:
        scene, shot = item["scene"], item["shot"]
        key = f"{scene}_shot{shot:02d}"
        fd = dump_shot_frames(scene, shot)
        W, H = Image.open(fd / "0.jpg").size
        chunks = list(range(item["chunk_start"], item["chunk_end"] + 1))
        frames_by_chunk = {c: [Image.open(fd / f"{(c - item['chunk_start']) * FPC + i}.jpg").convert("RGB") for i in range(FPC)] for c in chunks}
        methods: dict[str, dict[int, torch.Tensor]] = {}
        meta: dict = {}

        # M0 recorded phrase
        phrase0 = item["recorded_phrase"] or item["crop_phrase"]
        methods["m0_recorded"] = {c: clipseg_soft_masks(frames_by_chunk[c], phrase0, device=args.device) for c in chunks}
        meta["m0_phrase"] = phrase0

        # M1 constrained short phrases
        p1 = m1_short_phrases(client, item)
        phrases = [p for p in p1["repaint_phrases"] if p.strip()] or [item["crop_phrase"]]
        meta["m1_phrases"] = phrases
        meta["m1_drifted"] = p1["subject_drifted"]
        m1 = {}
        for c in chunks:
            softs = [clipseg_soft_masks(frames_by_chunk[c], p, device=args.device) for p in phrases]
            m1[c] = torch.stack(softs, 0).max(0).values
        methods["m1_short"] = m1

        # M2 boxes per chunk -> SAM2
        m2 = {}
        boxes_log = {}
        for c in chunks:
            png = fd / f"chunk{c:02d}_first.png"
            parsed = m2_boxes(client, item, png, W, H, phrases)
            raw = [b["bbox_2d"] for b in parsed["boxes"]]
            conv = args.convention
            if conv == "auto":
                conv = "abs" if any(v > 1000 for b in raw for v in b) else "norm1000"
            boxes = [scale_box(b, W, H, conv) for b in raw]
            boxes_log[c] = {"raw": parsed["boxes"], "convention": conv, "pixels": boxes}
            binm = sam.chunk_masks(fd, c - item["chunk_start"], boxes, (H, W))
            m2[c] = soft_from_binary(binm)
        methods["m2_box_sam2"] = m2
        meta["m2_boxes"] = boxes_log

        # M3 direct pointing with reference in view
        m3 = {}
        boxes_log3 = {}
        for c in chunks:
            png = fd / f"chunk{c:02d}_first.png"
            parsed = m3_boxes_direct(client, item, png, W, H)
            raw = [b["bbox_2d"] for b in parsed["boxes"]]
            boxes = [scale_box(b, W, H, "norm1000") for b in raw]
            boxes_log3[c] = {"raw": parsed["boxes"], "pixels": boxes}
            binm = sam.chunk_masks(fd, c - item["chunk_start"], boxes, (H, W))
            m3[c] = soft_from_binary(binm)
        methods["m3_box_direct"] = m3
        meta["m3_boxes"] = boxes_log3

        rows = []
        for name, soft in methods.items():
            save_mask_payload(soft, OUT / "masks" / f"{key}_{name}.pt", phrase=json.dumps(meta.get(f"{name.split('_')[0]}_phrase", meta.get("m1_phrases") if name.startswith("m1") else name)), video=baseline_video(scene))
            q = qc(soft)
            qc_all[f"{key}/{name}"] = q
            label = f"{key} {name} cov={q['coverage_mean']:.2f} zero={q['zero_chunks']} tIoU={q['temporal_iou']:.2f}"
            if name == "m0_recorded":
                label += f" | '{phrase0[:60]}'"
            if name == "m1_short":
                label += f" | {phrases}"
            rows.append(overlay_grid(fd, soft, item["chunk_start"], (H, W), label))
        grid = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "black")
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.height
        (OUT / "overlays").mkdir(exist_ok=True)
        grid.save(OUT / "overlays" / f"{key}.png")
        (OUT / "masks" / f"{key}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(key, {n: (round(qc_all[f'{key}/{n}']['coverage_mean'], 3), qc_all[f'{key}/{n}']['zero_chunks']) for n in methods})
    (OUT / "qc.json").write_text(json.dumps(qc_all, indent=2))


if __name__ == "__main__":
    main()
