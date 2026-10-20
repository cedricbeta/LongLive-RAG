#!/usr/bin/env python3
"""Render every frame of an automatic candidate beside the frozen original."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['feedback', 'fixed_noise'], required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    began = time.monotonic()
    root = args.root.resolve()
    assert Path(args.candidate).name == args.candidate
    plan = json.loads((root / 'plan.json').read_text())
    parent = Path(plan['parent'])
    parent_plan = parent.parent.parent / 'plan.json'
    assert sha(parent_plan) == plan['parent_batch_plan_sha256']
    review_frames = json.loads(parent_plan.read_text())['review_frames']
    candidate = root / args.arm / 'run/backend' / args.candidate
    status = json.loads((candidate / 'status.json').read_text())
    audit = json.loads((candidate / 'audit.json').read_text())
    assert status['status'] == 'completed' and status['returncode'] == 0
    assert audit['all_engineering_checks_passed']
    paths = {}
    for name, folder in [('original', parent), ('candidate', candidate)]:
        videos = list((folder / 'videos').glob('*.mp4'))
        assert len(videos) == 1
        paths[name] = videos[0]
    assert sha(paths['candidate']) == audit['video_sha256']
    output = root / 'reviews' / f'{args.arm}_{args.candidate}'
    output.mkdir(parents=True, exist_ok=False)
    caps = {name: cv2.VideoCapture(str(path)) for name, path in paths.items()}
    assert all(c.isOpened() and c.get(cv2.CAP_PROP_FPS) == 16 and c.get(cv2.CAP_PROP_FRAME_COUNT) == 960 for c in caps.values())
    movie = output / 'full_60s_comparison.mp4'
    command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', '1664x510', '-framerate', '16', '-i', '-', '-an',
               '-c:v', 'libx264', '-threads', '2', '-crf', '22', '-pix_fmt',
               'yuv420p', '-movflags', '+faststart', str(movie)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    records, frames = [], []
    pages = [Image.new('RGB', (832, 1620), 'white') for _ in range(3)]
    try:
        for index in range(960):
            row = {'frame': index, 'seconds': index / 16, 'rgb_sha256': {}}
            panes = []
            for name, cap in caps.items():
                ok, bgr = cap.read()
                assert ok and bgr.shape == (480, 832, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                row['rgb_sha256'][name] = hashlib.sha256(rgb.tobytes()).hexdigest()
                if index in review_frames:
                    dest = output / f'{name}_{index:04d}.png'
                    Image.fromarray(rgb).save(dest)
                    records.append({'branch': name, 'frame': index, 'path': dest.name,
                                    'sha256': sha(dest), 'rgb_sha256': row['rgb_sha256'][name]})
                pane = np.full((510, 832, 3), 245, np.uint8)
                pane[30:] = rgb
                label = 'Original' if name == 'original' else f'Automatic {args.arm} candidate; no independent acceptance'
                cv2.putText(pane, f'{label} | {index / 16:05.2f}s', (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .53, (20, 30, 40), 1, cv2.LINE_AA)
                panes.append(pane)
            frames.append(row)
            paired = np.concatenate(panes, axis=1)
            encoder.stdin.write(paired.tobytes())
            if index in review_frames:
                ordinal = review_frames.index(index)
                small = np.full((270, 832, 3), 245, np.uint8)
                small[30:] = cv2.resize(paired[30:], (832, 240), interpolation=cv2.INTER_AREA)
                cv2.putText(small, f'Original | {args.arm} candidate     frame {index} / {index / 16:g}s',
                            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 30, 40), 1, cv2.LINE_AA)
                pages[ordinal // 6].paste(Image.fromarray(small), (0, ordinal % 6 * 270))
        assert all(not cap.read()[0] for cap in caps.values())
    finally:
        for cap in caps.values():
            cap.release()
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
    if code:
        raise RuntimeError(error)
    for i, page in enumerate(pages):
        page.save(output / f'fixed_views_{i + 1}.png')
    check = cv2.VideoCapture(str(movie))
    decoded = 0
    try:
        assert check.isOpened() and check.get(cv2.CAP_PROP_FPS) == 16
        while True:
            ok, frame = check.read()
            if not ok:
                break
            assert frame.shape == (510, 1664, 3)
            decoded += 1
    finally:
        check.release()
    assert decoded == 960
    report = {
        'phase': 'automatic_candidate_complete_video_review', 'arm': args.arm,
        'candidate': args.candidate, 'source_sha256': sha(Path(__file__)),
        'plan_sha256': sha(root / 'plan.json'), 'candidate_audit_sha256': sha(candidate / 'audit.json'),
        'review_frames': review_frames, 'images': records,
        'source_videos': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
        'full_video_frames': frames,
        'comparison_video': {'path': movie.name, 'sha256': sha(movie),
            'frames_decoded': decoded, 'fps': 16, 'seconds': 60, 'resolution': [1664, 510],
            'all_source_frames_included': True, 'source_pixels_resized': False, 'command': command},
        'wall_seconds': time.monotonic() - began, 'human_labels': 0,
        'independent_acceptance': False, 'semantic_quality_claim': False,
        'limits': ['Lossy video comparison, not raw VAE pixels.',
                   'Every frame is included; fixed views alone do not assess continuous motion.',
                   'Program decisions are recorded separately from this developer review.',
                   'A passed engineering audit does not establish visual correctness or audited history bytes.']}
    (output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'images': len(records), 'complete_frames': decoded,
                      'wall_seconds': report['wall_seconds']}))


if __name__ == '__main__':
    main()
