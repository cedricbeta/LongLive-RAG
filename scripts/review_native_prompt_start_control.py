#!/usr/bin/env python3
"""Export a native-resolution two-by-two retention comparison and all fixed times."""
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


def review(root, names, output):
    began = time.monotonic()
    plan = json.loads((root / 'plan.json').read_text())
    assert names == ['B_original', 'P30_prompt', 'P20_prompt', 'RP_local_cue']
    allowed = {'B_original', 'P30_prompt', 'P20_prompt', 'RP_local_cue'}
    assert set(names) <= allowed
    for name in names:
        audit = json.loads((root / name / 'audit.json').read_text())
        assert audit.get('measured_engineering_checks_passed', audit.get('engineering_checks_passed', False))
    output.mkdir(exist_ok=False)
    paths = {name: next((root / name / 'videos').glob('*.mp4')) for name in names}
    caps = {name: cv2.VideoCapture(str(path)) for name, path in paths.items()}
    assert all(c.isOpened() and c.get(cv2.CAP_PROP_FPS) == 16 and c.get(cv2.CAP_PROP_FRAME_COUNT) == 960 for c in caps.values())
    movie = output / 'full_60s_comparison.mp4'
    command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1664x1020',
               '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '22',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    records, frames, pages = [], [], [Image.new('RGB', (832, 1620), 'white') for _ in range(6)]
    try:
        for index in range(960):
            row, panes = {'frame': index, 'seconds': index / 16, 'rgb_sha256': {}}, []
            for name, cap in caps.items():
                ok, bgr = cap.read()
                if not ok or bgr.shape != (480, 832, 3):
                    raise ValueError('Unexpected source video frame')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                row['rgb_sha256'][name] = hashlib.sha256(rgb.tobytes()).hexdigest()
                if index in plan['review_frames']:
                    dest = output / f'{name}_{index:04d}.png'
                    Image.fromarray(rgb).save(dest)
                    records.append({'branch': name, 'frame': index, 'path': dest.name,
                                    'sha256': sha(dest), 'rgb_sha256': row['rgb_sha256'][name]})
                pane = np.full((510, 832, 3), 245, np.uint8)
                pane[30:] = rgb
                label = {'B_original':'B: original', 'P30_prompt':'P30: ordinary cue from30s', 'P20_prompt':'P20: ordinary cue from20s', 'RP_local_cue':'RP: local repair + later cue'}[name]
                cv2.putText(pane, f'{label} | {index / 16:05.2f}s', (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, (20, 30, 40), 1, cv2.LINE_AA)
                panes.append(pane)
            frames.append(row)
            paired = np.concatenate([np.concatenate(panes[:2], 1), np.concatenate(panes[2:], 1)], 0)
            encoder.stdin.write(paired.tobytes())
            if index in plan['review_frames']:
                ordinal = plan['review_frames'].index(index)
                # Smaller review sheets only; the full video keeps native image pixels.
                small = np.full((540, 832, 3), 245, np.uint8)
                small[30:] = cv2.resize(paired, (832, 510), interpolation=cv2.INTER_AREA)
                cv2.putText(small, f'B | P30 above; P20 | RP below   frame {index} / {index / 16:g}s',
                            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 30, 40), 1, cv2.LINE_AA)
                pages[ordinal // 3].paste(Image.fromarray(small), (0, ordinal % 3 * 540))
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
            assert frame.shape == (1020, 1664, 3)
            decoded += 1
    finally:
        check.release()
    assert decoded == 960
    report = {'phase': 'native_same_boundary_prompt_control_review', 'arms': names, 'source_sha256': sha(Path(__file__)),
        'plan_sha256': sha(root / 'plan.json'), 'review_frames': plan['review_frames'], 'images': records,
        'source_videos': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
        'full_video_frames': frames, 'comparison_video': {'path': movie.name, 'sha256': sha(movie),
            'frames_decoded': decoded, 'fps': 16, 'seconds': 60, 'resolution': [1664, 1020], 'layout': 'B,P30 / P20,RP',
            'all_source_frames_included': True, 'source_pixels_resized': False, 'command': command},
        'wall_seconds': time.monotonic() - began, 'candidate_retained': False, 'human_labels': 0,
        'limits': ['MP4 source pixels and comparison are lossy video decodes/encodes, not raw VAE outputs.',
                   'Every frame is included; review sheets alone do not assess continuous motion.',
                   'Fixed-target development probe; no automatic or independent acceptance.']}
    (output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'images': len(records), 'complete_frames': decoded, 'wall_seconds': report['wall_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    review(args.root.resolve(), args.arms, args.output.resolve())
