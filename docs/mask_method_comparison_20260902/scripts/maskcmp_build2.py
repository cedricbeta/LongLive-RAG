"""Mask-method comparison, v2: everything derived from TODAY's baseline renders (deterministic within
the current inductor cache), so masks and repaired renders see the same content.

Step A  drift judge (pilot prompt, free-form repaint_region) on shots 1..5 -> drifted set + repair prompt
M0 recorded : judge's free-form repaint_region -> CLIPSeg              (= pilot v4 method)
M1 short    : constrained 1-3 short noun phrases -> CLIPSeg union
M2 box_sam2 : Qwen3-VL grounding of the M1 phrases -> SAM2 per chunk
M3 direct   : Qwen3-VL boxes "what to erase" with reference in view -> SAM2 per chunk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maskcmp_build as mb  # noqa: E402  (reuse schemas, SAM2 runner, overlay, qc)
from utils.agent_edit import build_repair_prompt, clipseg_soft_masks, read_video_frames, save_mask_payload  # noqa: E402
from utils.agent_vlm import DRIFT_JUDGE_SCHEMA, FingerprintedVLMClient  # noqa: E402

REPO = mb.REPO
FPC = 8
OUT = HERE / "run2"
NEWBASE = REPO / "videos/maskcmp_20260902/renders"
SCENES = mb.SCENES


def baseline_video(scene: str) -> Path:
    m = sorted((NEWBASE / scene).glob(f"baseline_rerun-rank*-{scene}-seed0_regular.mp4"))
    assert len(m) == 1, m
    return m[0]


def dump_frames(scene: str) -> Path:
    """All 384 frames as JPEG (SAM2 input), 3 judge PNGs per shot, chunk-first PNGs."""
    d = OUT / "frames" / scene
    if (d / "383.jpg").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    frames = read_video_frames(baseline_video(scene), 0, 384)
    assert len(frames) == 384
    for i, im in enumerate(frames):
        im.save(d / f"{i}.jpg", quality=95)
        shot, local = divmod(i, 64)
        if local in (0, 31, 62):
            im.save(d / f"shot{shot:02d}_f{(0, 31, 62).index(local)}.png")
        if local % FPC == 0:
            im.save(d / f"chunk{i // FPC:02d}_first.png")
    return d


def ref_frames(scene: str) -> list[Path]:
    return [OUT / "frames" / scene / f"shot00_f{k}.png" for k in range(3)]


def cur_frames(scene: str, shot: int) -> list[Path]:
    return [OUT / "frames" / scene / f"shot{shot:02d}_f{k}.png" for k in range(3)]


def caption(scene: str, shot: int) -> str:
    return json.loads((mb.PILOT / scene / "data" / scene / f"{shot}.json").read_text()).get("caption", "")


def pilot_judge(client, scene: str, shot: int) -> dict:
    """Same wording as utils/agent_vlm.py ComparativeDriftJudge (pilot v4)."""
    prompt = json.dumps(
        {
            "role": "comparative subject-drift judge",
            "rule": (
                "The first images are REFERENCE frames showing the subject as first established in this "
                "scene; the remaining images are the CURRENT provisional shot. Judge ONLY whether an "
                "ESTABLISHED character's identity or appearance has changed relative to the reference "
                "(species, color, face, clothing, distinguishing marks), or a character has been replaced "
                "or duplicated. The current shot's authored caption (current_caption below) may "
                "legitimately introduce new characters or remove existing ones — narrative entrances and "
                "exits that the caption calls for are NOT drift. Ignore pose, camera angle, framing, "
                "lighting, and allowed state changes."
            ),
            "current_caption": caption(scene, shot),
            "subject": SCENES[scene]["crop"],
            "reference_image_count": 3,
            "current_image_count": 3,
            "shot_index": shot,
            "drift_description": (
                "When drifted, describe concretely what changed versus the reference in one short "
                "sentence; empty string otherwise."
            ),
            "reference_description": (
                "Describe the subject's appearance AS SEEN IN THE REFERENCE frames in one positive phrase "
                "(colors, clothing, distinguishing marks). This text repaints the subject, so name ONLY "
                "what should appear — never mention the wrong or drifted appearance."
            ),
            "repaint_region": (
                "Short segmentation phrase naming everything in the CURRENT frames that must be erased and "
                "repainted: the drifted subject PLUS any person/object that the current caption does NOT "
                "call for. Never include characters the caption legitimately places in the shot. Use a "
                "broad noun like 'person' only when every visible person is wrong. Empty string if not drifted."
            ),
            "output": "strict JSON matching the supplied schema",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    res = client.call(
        prompt=prompt,
        image_paths=[*ref_frames(scene), *cur_frames(scene, shot)],
        schema=DRIFT_JUDGE_SCHEMA,
        call_kind="v2_drift_judge",
        shot_index=shot,
        temporal_inputs=(),
    )
    p = res["parsed"]
    return {
        "drifted": bool(p["subject_drifted"]) and float(p["confidence"]) >= 0.5,
        "confidence": p["confidence"],
        "drift_description": p["drift_description"],
        "reference_description": p["reference_description"],
        "repaint_region": p["repaint_region"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = FingerprintedVLMClient(base_url=mb.VLM_URL, model=mb.VLM_MODEL, cache_dir=OUT / "vlm_cache")
    from PIL import Image

    # Step A: judge every shot on the new baselines
    items = []
    judge_log = {}
    for scene in SCENES:
        dump_frames(scene)
        for shot in range(1, 6):
            v = pilot_judge(client, scene, shot)
            judge_log[f"{scene}/shot{shot}"] = v
            print(scene[:5], shot, "DRIFT" if v["drifted"] else "ok   ", v["repaint_region"][:60], "|", v["reference_description"][:60])
            if v["drifted"]:
                items.append(
                    {
                        "scene": scene,
                        "shot": shot,
                        "chunk_start": shot * FPC,
                        "chunk_end": shot * FPC + FPC - 1,
                        "caption": caption(scene, shot),
                        "crop_phrase": SCENES[scene]["crop"],
                        "recorded_phrase": v["repaint_region"],
                        "reference_description": v["reference_description"],
                        "drift_description": v["drift_description"],
                        "repair_prompt": build_repair_prompt(caption(scene, shot), v["reference_description"]),
                    }
                )
    (OUT / "judge_log.json").write_text(json.dumps(judge_log, indent=2, ensure_ascii=False))
    (OUT / "testset.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"{len(items)} drifted shots")

    # monkeypatch mb helpers so its m1/m2/m3 functions read the new frames
    mb.ref_frames = ref_frames
    mb.cur_frames = cur_frames
    mb.baseline_video = baseline_video
    sam = mb.Sam2Runner(args.device)
    qc_all = {}
    for item in items:
        scene, shot = item["scene"], item["shot"]
        key = f"{scene}_shot{shot:02d}"
        fd = OUT / "frames" / scene
        W, H = Image.open(fd / "0.jpg").size
        chunks = list(range(item["chunk_start"], item["chunk_end"] + 1))
        frames_by_chunk = {c: [Image.open(fd / f"{c * FPC + i}.jpg").convert("RGB") for i in range(FPC)] for c in chunks}
        # SAM2 runner expects a per-shot dir with local indices 0..63
        shot_dir = OUT / "frames" / f"{key}"
        shot_dir.mkdir(exist_ok=True)
        for i in range(64):
            dst = shot_dir / f"{i}.jpg"
            if not dst.exists():
                dst.symlink_to(fd / f"{shot * 64 + i}.jpg")
        methods, meta = {}, {}

        phrase0 = item["recorded_phrase"] or item["crop_phrase"]
        methods["m0_recorded"] = {c: clipseg_soft_masks(frames_by_chunk[c], phrase0, device=args.device) for c in chunks}
        meta["m0_phrase"] = phrase0

        p1 = mb.m1_short_phrases(client, item)
        phrases = [p for p in p1["repaint_phrases"] if p.strip()] or [item["crop_phrase"]]
        meta["m1_phrases"] = phrases
        methods["m1_short"] = {
            c: torch.stack([clipseg_soft_masks(frames_by_chunk[c], p, device=args.device) for p in phrases], 0).max(0).values
            for c in chunks
        }

        for name, fn in (("m2_box_sam2", lambda png: mb.m2_boxes(client, item, png, W, H, phrases)), ("m3_box_direct", lambda png: mb.m3_boxes_direct(client, item, png, W, H))):
            masks, log = {}, {}
            for c in chunks:
                try:
                    parsed = fn(fd / f"chunk{c:02d}_first.png")
                except Exception as exc:  # VLM emitted unparseable JSON twice -> treat as no boxes, log it
                    print(f"  [{name}] chunk {c}: VLM parse failure ({type(exc).__name__}), using no boxes", flush=True)
                    parsed = {"boxes": [], "error": str(exc)[:200]}
                boxes = [mb.scale_box(b["bbox_2d"], W, H, "norm1000") for b in parsed["boxes"]]
                log[c] = {"raw": parsed["boxes"], "pixels": boxes}
                masks[c] = mb.soft_from_binary(sam.chunk_masks(shot_dir, c - item["chunk_start"], boxes, (H, W)))
            methods[name] = masks
            meta[f"{name}_boxes"] = log

        rows = []
        for name, soft in methods.items():
            save_mask_payload(soft, OUT / "masks" / f"{key}_{name}.pt", phrase=name, video=baseline_video(scene))
            q = mb.qc(soft)
            qc_all[f"{key}/{name}"] = q
            label = f"{key} {name} cov={q['coverage_mean']:.2f} zero={q['zero_chunks']} tIoU={q['temporal_iou']:.2f}"
            if name == "m0_recorded":
                label += f" | '{phrase0[:70]}'"
            if name == "m1_short":
                label += f" | {phrases}"
            rows.append(mb.overlay_grid(shot_dir, soft, item["chunk_start"], (H, W), label))
        grid = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "black")
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.height
        (OUT / "overlays").mkdir(exist_ok=True)
        grid.save(OUT / "overlays" / f"{key}.png")
        (OUT / "masks" / f"{key}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(key, {n: (round(qc_all[f'{key}/{n}']['coverage_mean'], 3), qc_all[f'{key}/{n}']['zero_chunks']) for n in methods}, flush=True)
    (OUT / "qc.json").write_text(json.dumps(qc_all, indent=2))


if __name__ == "__main__":
    main()
