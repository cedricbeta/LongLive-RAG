"""Comparative drift relabeling of saved block thumbnails.

The absolute judge ("reject only clear failures") is blind to gradual drift.
This pass re-judges each block AGAINST reference frames from the start of the
same video, producing graded drift labels for negative-memory validation.
"""
import argparse
import glob
import json
import os

from PIL import Image
import torch

from vlm_guide import VLMGuide, _extract_json, _to_data_uri
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="negmem_native/native")
parser.add_argument("--prompt_indices", default="0,2,4,6,7,8")
parser.add_argument("--prompts_file", default="checkpoints/moviegenbench_128_refined.txt")
args = parser.parse_args()

with open(args.prompts_file) as f:
    all_prompts = [l.strip() for l in f if l.strip()]

vlm = VLMGuide()

def load(p):
    return TF.to_tensor(Image.open(p))

for pi in [int(i) for i in args.prompt_indices.split(",")]:
    tdir = os.path.join(args.root, f"prompt{pi:03d}_thumbs")
    thumbs = sorted(glob.glob(os.path.join(tdir, "frame*.jpg")))
    if not thumbs:
        print(f"prompt {pi}: no thumbs, skip")
        continue
    prompt = all_prompts[pi]
    refs = [load(t) for t in thumbs[:2]]
    out = []
    for tp in thumbs:
        fid = int(os.path.basename(tp)[5:8])
        cur = load(tp)
        content = [{"type": "image_url", "image_url": {"url": _to_data_uri(f)}} for f in refs]
        content.append({"type": "image_url", "image_url": {"url": _to_data_uri(cur)}})
        content.append({"type": "text", "text": (
            "The first two images are REFERENCE frames from the start of an AI-generated "
            f"video of: \"{prompt}\"\nThe third image is a LATER segment of the same video.\n\n"
            "Compare the later segment against the reference:\n"
            "- subject_drift: has the main subject's identity/appearance changed "
            "(colors, patterns, shape, duplicated subjects)? none / mild / severe\n"
            "- scene_drift: has the scene, background or composition drifted away "
            "from the reference? none / mild / severe\n"
            "Normal motion, small viewpoint changes and camera movement are 'none'.\n\n"
            "Answer with JSON only: {\"subject_drift\": \"none|mild|severe\", "
            "\"scene_drift\": \"none|mild|severe\", \"reason\": \"<15 words\"}"
        )})
        verdict = None
        for _ in range(3):
            try:
                verdict = _extract_json(vlm._chat(content))
                break
            except Exception as e:
                verdict = {"error": str(e)}
        sd = str(verdict.get("subject_drift", "none")).lower()
        cd = str(verdict.get("scene_drift", "none")).lower()
        out.append({"frame": fid, "block": fid // 3, "subject_drift": sd,
                    "scene_drift": cd, "reason": verdict.get("reason", ""),
                    "drop": ("severe" in (sd, cd))})
    path = os.path.join(args.root, f"prompt{pi:03d}_driftlabels.json")
    with open(path, "w") as fj:
        json.dump(out, fj, indent=1)
    n_sev = sum(1 for r in out if r["drop"])
    n_mild = sum(1 for r in out if not r["drop"] and ("mild" in (r["subject_drift"], r["scene_drift"])))
    print(f"prompt {pi}: {len(out)} blocks -> severe {n_sev}, mild {n_mild}, none {len(out)-n_sev-n_mild}")
print("done")
