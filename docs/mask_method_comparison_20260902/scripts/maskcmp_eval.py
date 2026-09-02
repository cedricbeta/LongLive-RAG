"""Score the three mask methods on the repaired shots.

identity : DINO cosine between subject crops (CLIPSeg crop on crop_phrase) in the edited shot and the
           shot-0 reference centroid of the same baseline video. Reported together with the unedited
           baseline value of the same shot (floor) and the shot-0 self value (ceiling-ish).
bg_mae   : pixel MAE outside the (dilated) mask vs the unedited baseline -> background lock check.
judge    : the pilot's comparative drift judge (Qwen3-VL-8B, same prompt/schema) on reference + 3
           edited frames -> accept / drifted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/chendong/video-gen/LongLive-RAG")
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maskcmp_build as mb  # noqa: E402
from evaluation.vbench_consistency import build_dino_encoder  # noqa: E402
from utils.agent_vlm import DRIFT_JUDGE_SCHEMA, FingerprintedVLMClient  # noqa: E402

PILOT = mb.PILOT
RENDERS = REPO / "videos/maskcmp_20260902/renders"
FPC = 8
DEVICE = "cuda:0"


def read_frames(video: Path, idxs: list[int]) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"read failed {video} {i}")
        out.append(bgr[:, :, ::-1].copy())
    cap.release()
    return out


class Cropper:
    def __init__(self):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE).eval()

    def crops(self, frames, phrase, margin=0.15):
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        with torch.no_grad():
            inp = self.proc(text=[phrase] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(DEVICE)
            soft = torch.sigmoid(self.model(**inp).logits).cpu().numpy()
        out = []
        for f, m in zip(frames, soft):
            h, w = f.shape[:2]
            mm = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h))) / 255.0
            ys, xs = np.where(mm >= 0.35)
            if len(ys) < 20:
                out.append(f)
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            dy, dx = int((y1 - y0) * margin), int((x1 - x0) * margin)
            out.append(f[max(0, y0 - dy) : min(h, y1 + dy), max(0, x0 - dx) : min(w, x1 + dx)])
        return out


def video_for(scene: str, shot: int, method: str) -> Path | None:
    if method == "baseline":
        return mb.baseline_video(scene)
    if method == "m0_recorded":
        m = sorted((PILOT / scene / "renders").glob(f"edit_shot{shot:02d}-rank*_regular.mp4"))
    else:
        m = sorted((RENDERS / scene).glob(f"edit_shot{shot:02d}_{method}-rank*_regular.mp4"))
    return m[0] if m else None


def mask_for(scene: str, shot: int, method: str):
    if method == "m0_recorded":
        p = PILOT / scene / "edit" / f"shot{shot:02d}_masks.pt"
    else:
        p = mb.OUT / "masks" / f"{scene}_shot{shot:02d}_{method}.pt"
    return torch.load(p, map_location="cpu", weights_only=False)["masks"]


def judge(client: FingerprintedVLMClient, item: dict, frames_png: list[Path]) -> dict:
    prompt = json.dumps(
        {
            **mb.judge_prompt_common(item),
            "drift_description": "one short sentence if drifted, else empty",
            "reference_description": "positive phrase describing the subject in the REFERENCE frames",
            "repaint_region": "short segmentation phrase, empty if not drifted",
            "output": "strict JSON matching the supplied schema",
        },
        sort_keys=True,
    )
    res = client.call(
        prompt=prompt,
        image_paths=[*mb.ref_frames(item["scene"]), *frames_png],
        schema=DRIFT_JUDGE_SCHEMA,
        call_kind="maskcmp_reverify",
        shot_index=item["shot"],
        temporal_inputs=(),
    )
    p = res["parsed"]
    return {"drifted": bool(p["subject_drifted"]) and float(p["confidence"]) >= 0.5, "confidence": p["confidence"], "drift_description": p["drift_description"]}


def main():
    from PIL import Image

    items = json.loads((mb.OUT / "testset.json").read_text())
    dino = build_dino_encoder(device=DEVICE)
    cropper = Cropper()
    client = FingerprintedVLMClient(base_url=mb.VLM_URL, model=mb.VLM_MODEL, cache_dir=mb.OUT / "vlm_cache")
    methods = ["baseline", "m0_recorded", "m1_short", "m2_box_sam2", "m3_box_direct"]
    ref_centroid = {}
    for scene in mb.SCENES:
        crops = cropper.crops(read_frames(mb.baseline_video(scene), list(range(0, 64, 4))), mb.SCENES[scene]["crop"])
        e = dino(crops)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        c = e.mean(0)
        ref_centroid[scene] = c / np.linalg.norm(c)
    rows = []
    frames_dir = mb.OUT / "eval_frames"
    frames_dir.mkdir(exist_ok=True)
    for it in items:
        scene, shot = it["scene"], it["shot"]
        s0, s1 = shot * 64, shot * 64 + 64
        base_frames = np.stack(read_frames(mb.baseline_video(scene), list(range(s0, s1))))
        for method in methods:
            vid = video_for(scene, shot, method)
            if vid is None:
                print("missing render", scene, shot, method)
                continue
            frames = read_frames(vid, list(range(s0, s1, 4)))
            crops = cropper.crops(frames, it["crop_phrase"])
            e = dino(crops)
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            identity = float((e @ ref_centroid[scene]).mean())
            row = {"scene": scene, "shot": shot, "method": method, "identity": identity}
            if method != "baseline":
                masks = mask_for(scene, shot, method)
                all_frames = np.stack(read_frames(vid, list(range(s0, s1))))
                H, W = all_frames.shape[1:3]
                maes_out, maes_in = [], []
                for c in range(shot * FPC, shot * FPC + FPC):
                    soft = masks[c]
                    binm = (soft >= 0.35).float()
                    binm = torch.nn.functional.max_pool2d(binm[None], 3, 1, 1)[0]  # dilate 1 like pipeline
                    up = torch.nn.functional.interpolate(binm[None], size=(H, W), mode="nearest")[0].numpy() > 0.5
                    for f in range(FPC):
                        gi = (c - shot * FPC) * FPC + f
                        diff = np.abs(all_frames[gi].astype(np.float32) - base_frames[gi].astype(np.float32)).mean(-1)
                        if (~up[f]).any():
                            maes_out.append(diff[~up[f]].mean())
                        if up[f].any():
                            maes_in.append(diff[up[f]].mean())
                row["bg_mae_outside"] = float(np.mean(maes_out)) if maes_out else None
                row["mae_inside"] = float(np.mean(maes_in)) if maes_in else None
                pngs = []
                for k, gi in enumerate((s0, s0 + 31, s0 + 62)):
                    p = frames_dir / f"{scene}_shot{shot:02d}_{method}_f{k}.png"
                    if not p.exists():
                        Image.fromarray(read_frames(vid, [gi])[0]).save(p)
                    pngs.append(p)
                row.update({f"judge_{k}": v for k, v in judge(client, it, pngs).items()})
            rows.append(row)
            print(row, flush=True)
    (mb.OUT / "eval.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    # summary table
    import collections

    by = collections.defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    print("\nmethod           n  identity  d_vs_base  bg_mae  judge_accept")
    base_id = {(r["scene"], r["shot"]): r["identity"] for r in rows if r["method"] == "baseline"}
    for m in methods:
        rs = by[m]
        if not rs:
            continue
        ident = np.mean([r["identity"] for r in rs])
        d = np.mean([r["identity"] - base_id[(r["scene"], r["shot"])] for r in rs])
        bg = np.mean([r["bg_mae_outside"] for r in rs if r.get("bg_mae_outside") is not None]) if m != "baseline" else float("nan")
        acc = np.mean([0 if r["judge_drifted"] else 1 for r in rs]) if m != "baseline" else float("nan")
        print(f"{m:16s} {len(rs):2d}  {ident:.3f}    {d:+.3f}    {bg:5.1f}   {acc:.2f}")


if __name__ == "__main__":
    main()
