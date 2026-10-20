#!/usr/bin/env python3
"""Compare a real zero-mask replay with its saved parent, without judging quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.agent_runtime import file_sha256
from utils.parent_latent_replay import ParentLatentReplay


def compare(config_path, latent_path, video_path):
    config = yaml.safe_load(config_path.read_text())['inference']
    replay = ParentLatentReplay(config['replay_parent'])
    plans = config['edit_renoise_plans']
    if len(plans) != 1:
        raise ValueError('Expected one zero-mask edit plan')
    payload = torch.load(plans[0]['mask_path'], map_location='cpu', weights_only=False)
    if not payload['masks'] or any(torch.count_nonzero(x) for x in payload['masks'].values()):
        raise ValueError('Control contains a nonzero mask')
    child = torch.load(latent_path, map_location='cpu', weights_only=True)
    if child.shape != replay.latent.shape or child.dtype != replay.latent.dtype:
        raise ValueError('Parent and control latent layouts differ')
    if not torch.isfinite(child).all():
        raise ValueError('Nonfinite control latents')
    chunks = []
    for k in range(replay.num_chunks):
        start = k * replay.frames_per_chunk
        a = replay.latent[:, start:start + replay.frames_per_chunk].contiguous()
        b = child[:, start:start + replay.frames_per_chunk].contiguous()
        digest = lambda x: hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
        chunks.append({'chunk': k, 'parent_sha256': digest(a), 'control_sha256': digest(b),
                       'changed_elements': int((a != b).sum()),
                       'mean_absolute_difference': float((a.float() - b.float()).abs().mean())})
    caps = [cv2.VideoCapture(str(p)) for p in [replay.config['video_path'], video_path]]
    if not all(c.isOpened() for c in caps):
        raise ValueError('Cannot open both videos')
    rates = [c.get(cv2.CAP_PROP_FPS) for c in caps]
    frames, changed_frames, hashes = 0, [], [hashlib.sha256(), hashlib.sha256()]
    try:
        while True:
            (ok_a, a), (ok_b, b) = [c.read() for c in caps]
            if ok_a != ok_b:
                raise ValueError('Decoded video frame counts differ')
            if not ok_a:
                break
            if a.shape != b.shape:
                raise ValueError('Decoded frame shapes differ')
            for h, frame in zip(hashes, [a, b]):
                h.update(frame.tobytes())
            if not np.array_equal(a, b):
                changed_frames.append(frames)
            frames += 1
    finally:
        for cap in caps:
            cap.release()
    if not frames:
        raise ValueError('No decoded frames')
    return {'phase': 'engineering_zero_edit_control', 'quality_claim': False,
            'config_sha256': file_sha256(config_path), 'parent': replay.config,
            'control_latent_sha256': file_sha256(latent_path), 'control_video_sha256': file_sha256(video_path),
            'mask_all_zero': True, 'shape': list(child.shape), 'dtype': str(child.dtype),
            'chunks': chunks, 'all_latents_identical': all(c['changed_elements'] == 0 for c in chunks),
            'video': {'decoded_frames': frames, 'fps': rates, 'changed_frames': changed_frames,
                      'parent_decoded_sha256': hashes[0].hexdigest(), 'control_decoded_sha256': hashes[1].hexdigest(),
                      'all_frames_identical': not changed_frames},
            'limits': ['One exposed development case; no visual-quality or population claim.',
                       'Does not directly compare serialized KV, RNG or retrieval states.',
                       'The edited range starts at zero: no nonempty strict-prefix evidence.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ['config', 'latent', 'video', 'output']:
        p.add_argument('--' + field, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    result = compare(args.config, args.latent, args.video)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_latents_identical': result['all_latents_identical'], 'video': result['video']}))
