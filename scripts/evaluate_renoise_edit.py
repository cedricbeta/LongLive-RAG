#!/usr/bin/env python
"""Evaluate the SDEdit masked-renoise repair arms.

Per arm, against the unedited attn_probe render of the same seed:

  identity   DINO cosine of subject crops (CLIPSeg bbox) in the edited shot
             vs a reference set taken from the video's own canonical shots
             (the look we are trying to restore) -- reported for the original
             shot and every arm, so the delta is the repair effect.
  motion     RAFT dynamic degree of the edited shot vs the original shot
             (repair must not freeze or shake the motion).
  bg_lock    mean |pixel delta| outside the dilated subject mask (should be
             ~0: background latents are pinned during the edit).
  carry      identity of the FOLLOWING shots vs reference (does the repaired
             history propagate through the recache?).

Also writes side-by-side comparison mp4s and a frame-strip figure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.vbench_consistency import build_dino_encoder
from evaluation.video_consistency import build_raft_dynamic_scorer

EDIT_ROOT = ROOT / "videos/renoise_edit_20260821"
ORIG = {
    "sandy_beach_driftwood": ROOT / "videos/attn_probe_20260820/sandy_beach_driftwood/renders/attn_probe-rank0-sandy_beach_driftwood-seed0_regular.mp4",
    "bearded_watchmaker_workshop": ROOT / "videos/attn_probe_20260820/bearded_watchmaker_workshop/renders/attn_probe-rank0-bearded_watchmaker_workshop-seed0_regular.mp4",
}
ARMS = [
    ("drift3_e1", "sandy_beach_driftwood", 30, 35),
    ("drift3_e2", "sandy_beach_driftwood", 30, 35),
    ("drift3_e3", "sandy_beach_driftwood", 30, 35),
    ("drift3_s40", "sandy_beach_driftwood", 30, 35),
    ("watch3_e1", "bearded_watchmaker_workshop", 30, 35),
    ("watch3_e2", "bearded_watchmaker_workshop", 30, 35),
    ("watch3_e3", "bearded_watchmaker_workshop", 30, 35),
    ("drift2_ctrl_e2", "sandy_beach_driftwood", 24, 29),
]
PHRASE = {"sandy_beach_driftwood": "fish", "bearded_watchmaker_workshop": "man"}
# canonical reference frames (original video): the look to restore
REF_FRAMES = {
    "sandy_beach_driftwood": list(range(120, 209, 8)),
    "bearded_watchmaker_workshop": list(range(100, 189, 8)),
}
FPC = 8  # frames per chunk


def read_frames(video: Path, indices: list[int]) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"read failed {video} frame {idx}")
        out.append(bgr[:, :, ::-1].copy())
    cap.release()
    return out


class SubjectCropper:
    def __init__(self, device: str):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = (
            CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            .to(device).eval()
        )
        self.device = device

    def crops(self, frames: list[np.ndarray], phrase: str,
              margin: float = 0.15) -> list[np.ndarray]:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        with torch.no_grad():
            inputs = self.proc(text=[phrase] * len(imgs), images=imgs,
                               return_tensors="pt", padding=True).to(self.device)
            soft = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()
        out = []
        for f, m in zip(frames, soft):
            h, w = f.shape[:2]
            mm = np.array(
                Image.fromarray((m * 255).astype(np.uint8)).resize((w, h))
            ) / 255.0
            ys, xs = np.where(mm >= 0.35)
            if len(ys) < 20:
                out.append(f)
                continue
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            dy, dx = int((y1 - y0) * margin), int((x1 - x0) * margin)
            out.append(f[max(0, y0 - dy):min(h, y1 + dy),
                         max(0, x0 - dx):min(w, x1 + dx)])
        return out


def find_arm_video(arm: str, scene: str) -> Path:
    hits = sorted((EDIT_ROOT / arm).glob("*.mp4"))
    if not hits:
        raise SystemExit(f"no video for arm {arm}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(EDIT_ROOT / "eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dino = build_dino_encoder(device=args.device)
    raft = build_raft_dynamic_scorer(device=args.device)
    cropper = SubjectCropper(args.device)

    # reference centroids per scene
    ref_centroid = {}
    for scene, idxs in REF_FRAMES.items():
        crops = cropper.crops(read_frames(ORIG[scene], idxs), PHRASE[scene])
        emb = dino(crops)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        ref_centroid[scene] = emb.mean(0) / np.linalg.norm(emb.mean(0))

    def identity(video: Path, scene: str, frame_idxs: list[int]) -> float:
        crops = cropper.crops(read_frames(video, frame_idxs), PHRASE[scene])
        emb = dino(crops)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return float((emb @ ref_centroid[scene]).mean())

    results = {}
    for arm, scene, cs, ce in ARMS:
        vid = find_arm_video(arm, scene)
        shot_idxs = list(range(cs * FPC, (ce + 1) * FPC, 4))
        after_idxs = [i for i in range((ce + 1) * FPC, (ce + 1) * FPC + 96, 8)
                      if i < 480]
        shot_all = list(range(cs * FPC, (ce + 1) * FPC))

        orig_frames = np.stack(read_frames(ORIG[scene], shot_all))
        edit_frames = np.stack(read_frames(vid, shot_all))

        # background lock: pixel MAE outside the dilated latent mask
        mask_file = next((EDIT_ROOT / "masks").glob(
            f"*{'shot3' if cs == 30 else 'shot2'}*.pt"))
        # pick the scene-matching mask file
        for mf in (EDIT_ROOT / "masks").glob("*.pt"):
            p = torch.load(mf, map_location="cpu", weights_only=False)
            if p["video"].find(scene) >= 0 and int(sorted(p["masks"])[0]) == cs:
                mask_file, payload = mf, p
                break
        h, w = orig_frames.shape[1:3]
        maes = []
        for ci in range(cs, ce + 1):
            soft = payload["masks"][ci]
            m = (soft >= 0.35).float()
            m = torch.nn.functional.max_pool2d(
                m.unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).numpy()
            for fi in range(FPC):
                g = (ci - cs) * FPC + fi
                diff = np.abs(orig_frames[g].astype(np.float32)
                              - edit_frames[g].astype(np.float32)).mean(-1)
                outside = 1.0 - m[fi]
                if outside.sum() > 0:
                    maes.append(float((diff * outside).sum() / outside.sum()))

        row = {
            "identity_orig": identity(ORIG[scene], scene, shot_idxs),
            "identity_edit": identity(vid, scene, shot_idxs),
            "motion_orig": raft(orig_frames),
            "motion_edit": raft(edit_frames),
            "bg_mae": float(np.mean(maes)),
            "carry_orig": identity(ORIG[scene], scene, after_idxs) if after_idxs else None,
            "carry_edit": identity(vid, scene, after_idxs) if after_idxs else None,
        }
        row["identity_delta"] = row["identity_edit"] - row["identity_orig"]
        results[arm] = row
        print(f"{arm}: id {row['identity_orig']:.3f}->{row['identity_edit']:.3f} "
              f"(d={row['identity_delta']:+.3f})  motion "
              f"{row['motion_orig']:.2f}->{row['motion_edit']:.2f}  "
              f"bg_mae {row['bg_mae']:.2f}  carry "
              f"{row['carry_orig'] if row['carry_orig'] is None else round(row['carry_orig'],3)}"
              f"->{row['carry_edit'] if row['carry_edit'] is None else round(row['carry_edit'],3)}")

        # 2-up video: edited shot plus following context
        a = max(0, cs * FPC - 24)
        b = min(480, (ce + 1) * FPC + 72)
        two_up = out_dir / f"{arm}_2up.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(ORIG[scene]), "-i", str(vid),
            "-filter_complex",
            f"[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,"
            f"drawtext=text='original':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.4[a];"
            f"[1:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS,"
            f"drawtext=text='{arm}':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.4[b];"
            f"[a][b]hstack", "-r", "24", str(two_up),
        ], check=False)

    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out_dir / 'results.json'}")

    # frame strip figure per scene
    for scene, arm_names in (
        ("sandy_beach_driftwood", ["drift3_e1", "drift3_e2", "drift3_e3", "drift3_s40"]),
        ("bearded_watchmaker_workshop", ["watch3_e1", "watch3_e2", "watch3_e3"]),
    ):
        cs, ce = 30, 35
        sample = [cs * FPC + 8, cs * FPC + 24, cs * FPC + 40]
        cols = [("reference", ORIG[scene], [REF_FRAMES[scene][2]] * 3),
                ("original", ORIG[scene], sample)]
        cols += [(a, find_arm_video(a, scene), sample) for a in arm_names]
        fig, axes = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 4.6))
        for c, (label, vid, idxs) in enumerate(cols):
            frames = read_frames(vid, idxs)
            for r in range(3):
                axes[r][c].imshow(frames[r])
                axes[r][c].axis("off")
                if r == 0:
                    axes[r][c].set_title(label, fontsize=8)
        fig.suptitle(f"{scene}: repair arms vs original (rows = 3 frames of the edited shot)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_strip.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print("strips written")


if __name__ == "__main__":
    main()
