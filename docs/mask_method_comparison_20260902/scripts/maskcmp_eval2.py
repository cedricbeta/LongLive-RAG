"""Score the four mask methods on today's repaired renders (see maskcmp_eval.py for metric notes)."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maskcmp_build as mb  # noqa: E402
import maskcmp_build2 as mb2  # noqa: E402
import maskcmp_eval as ev  # noqa: E402
from evaluation.vbench_consistency import build_dino_encoder  # noqa: E402
from utils.agent_vlm import FingerprintedVLMClient  # noqa: E402

RENDERS = mb2.REPO / "videos/maskcmp_20260902/renders2"
OUT = mb2.OUT
FPC = 8
METHODS = ["baseline", "m0_recorded", "m1_short", "m2_box_sam2", "m3_box_direct"]


def video_for(scene, shot, method):
    if method == "baseline":
        return mb2.baseline_video(scene)
    m = sorted((RENDERS / scene).glob(f"edit_shot{shot:02d}_{method}-rank*_regular.mp4"))
    return m[0] if m else None


def main():
    from PIL import Image

    mb.ref_frames = mb2.ref_frames  # judge reads today's reference frames
    items = json.loads((OUT / "testset.json").read_text())
    dino = build_dino_encoder(device=ev.DEVICE)
    cropper = ev.Cropper()
    client = FingerprintedVLMClient(base_url=mb.VLM_URL, model=mb.VLM_MODEL, cache_dir=OUT / "vlm_cache")
    ref_centroid = {}
    for scene in mb.SCENES:
        e = dino(cropper.crops(ev.read_frames(mb2.baseline_video(scene), list(range(0, 64, 4))), mb.SCENES[scene]["crop"]))
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        c = e.mean(0)
        ref_centroid[scene] = c / np.linalg.norm(c)
    rows = []
    frames_dir = OUT / "eval_frames"
    frames_dir.mkdir(exist_ok=True)
    for it in items:
        scene, shot = it["scene"], it["shot"]
        s0, s1 = shot * 64, shot * 64 + 64
        base_frames = np.stack(ev.read_frames(mb2.baseline_video(scene), list(range(s0, s1))))
        for method in METHODS:
            vid = video_for(scene, shot, method)
            if vid is None:
                print("missing render", scene, shot, method)
                continue
            e = dino(cropper.crops(ev.read_frames(vid, list(range(s0, s1, 4))), it["crop_phrase"]))
            e = e / np.linalg.norm(e, axis=1, keepdims=True)
            row = {"scene": scene, "shot": shot, "method": method, "identity": float((e @ ref_centroid[scene]).mean())}
            if method != "baseline":
                masks = torch.load(OUT / "masks" / f"{scene}_shot{shot:02d}_{method}.pt", map_location="cpu", weights_only=False)["masks"]
                all_frames = np.stack(ev.read_frames(vid, list(range(s0, s1))))
                H, W = all_frames.shape[1:3]
                # pre-edit determinism check: frame 5 of the render must equal the baseline
                pre = np.abs(ev.read_frames(vid, [5])[0].astype(np.float32) - ev.read_frames(mb2.baseline_video(scene), [5])[0].astype(np.float32)).mean()
                row["pre_edit_mae"] = float(pre)
                maes_out, maes_in = [], []
                for c in range(shot * FPC, shot * FPC + FPC):
                    binm = (masks[c] >= 0.35).float()
                    binm = torch.nn.functional.max_pool2d(binm[None], 3, 1, 1)[0]
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
                        Image.fromarray(ev.read_frames(vid, [gi])[0]).save(p)
                    pngs.append(p)
                row.update({f"judge_{k}": v for k, v in ev.judge(client, it, pngs).items()})
            rows.append(row)
            print(row, flush=True)
    (OUT / "eval.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    base_id = {(r["scene"], r["shot"]): r["identity"] for r in rows if r["method"] == "baseline"}
    print("\nmethod           n  identity  d_vs_base  bg_mae  judge_accept  pre_edit_mae")
    for m in METHODS:
        rs = by[m]
        if not rs:
            continue
        ident = np.mean([r["identity"] for r in rs])
        d = np.mean([r["identity"] - base_id[(r["scene"], r["shot"])] for r in rs])
        if m == "baseline":
            print(f"{m:16s} {len(rs):2d}  {ident:.3f}    {d:+.3f}")
            continue
        bg = np.mean([r["bg_mae_outside"] for r in rs if r.get("bg_mae_outside") is not None])
        acc = np.mean([0 if r["judge_drifted"] else 1 for r in rs])
        pre = np.max([r["pre_edit_mae"] for r in rs])
        print(f"{m:16s} {len(rs):2d}  {ident:.3f}    {d:+.3f}    {bg:5.1f}   {acc:.2f}          {pre:.2f}")


if __name__ == "__main__":
    main()
