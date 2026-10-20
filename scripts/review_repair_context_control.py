#!/usr/bin/env python3
"""Keep all pre-frozen views and all video frames for the context diagnostic."""
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
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def review(args):
    began = time.monotonic()
    folder = args.folder.resolve()
    output = folder / 'frame_review'
    output.mkdir(exist_ok=False)
    plan = json.loads((folder / 'plan.json').read_text())
    off = json.loads((folder / 'off/context_audit.json').read_text())
    assert off['off_conformance_passed']
    assert json.loads((folder / 'on/context_audit.json').read_text())['native_engineering_checks_passed']
    branches = [('original', Path(plan['parent'])), ('full_context', Path(plan['reference'])), ('current_block', folder / 'on')]
    paths = {name: next((path / 'videos').glob('*.mp4')) for name, path in branches}
    videos = {name: cv2.VideoCapture(str(path)) for name, path in paths.items()}
    assert all(v.isOpened() for v in videos.values())
    assert all(v.get(cv2.CAP_PROP_FPS) == 16 and v.get(cv2.CAP_PROP_FRAME_COUNT) == 960
               and v.get(cv2.CAP_PROP_FRAME_WIDTH) == 832 and v.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480 for v in videos.values())
    movie = output / 'three_way_comparison.mp4'
    command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1248x270',
               '-framerate', '16', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '22',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    records, frames = [], []
    try:
        for index in range(960):
            rgb = {}
            for name, video in videos.items():
                ok, bgr = video.read()
                if not ok:
                    raise RuntimeError('Truncated source video: ' + name)
                rgb[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            difference = np.abs(rgb['current_block'].astype(np.int16) - rgb['full_context'].astype(np.int16))
            row = {'frame': index, 'seconds': index / 16,
                'rgb_sha256': {name: hashlib.sha256(frame.tobytes()).hexdigest() for name, frame in rgb.items()},
                'current_block_vs_full_context_mae_255': float(difference.mean()),
                'current_block_vs_full_context_changed_pixels': int((difference != 0).any(2).sum())}
            frames.append(row)
            if index in plan['review_frames']:
                for name, frame in rgb.items():
                    dest = output / f'{name}_{index:04d}.png'
                    Image.fromarray(frame).save(dest)
                    records.append({'branch': name, 'frame': index, 'seconds': index / 16,
                                    'path': dest.name, 'sha256': sha(dest), 'rgb_sha256': row['rgb_sha256'][name]})
            panes = []
            for name, frame in rgb.items():
                pane = np.full((270, 416, 3), 245, np.uint8)
                pane[30:] = cv2.resize(frame, (416, 240), interpolation=cv2.INTER_AREA)
                label = {'original': 'Original parent', 'full_context': 'Full context (C = OFF)',
                         'current_block': 'Current block only'}[name]
                cv2.putText(pane, f'{label} | {index/16:05.2f}s', (9, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 30, 40), 1, cv2.LINE_AA)
                panes.append(pane)
            encoder.stdin.write(np.concatenate(panes, 1).tobytes())
        assert all(not v.read()[0] for v in videos.values())
    finally:
        for video in videos.values():
            video.release()
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
    if code:
        raise RuntimeError('FFmpeg failed: ' + error)
    record = {'protocol': 'native_context_full_video_review_v1', 'plan_sha256': sha(folder / 'plan.json'),
        'source_sha256': sha(Path(__file__)), 'source_videos': {n: {'path': str(p), 'sha256': sha(p)} for n, p in paths.items()},
        'actual_frames_per_branch': 960, 'fps': 16, 'native_resolution': [832, 480], 'review_frames': plan['review_frames'],
        'images': records, 'full_video_frames': frames,
        'comparison_video': {'path': movie.name, 'sha256': sha(movie), 'resolution': [1248, 270], 'frames': 960,
            'command': command, 'all_source_frames_included': True, 'resize': 'OpenCV INTER_AREA to 416x240 per pane'},
        'wall_seconds': time.monotonic() - began, 'candidate_retained': False, 'human_labels': 0,
        'limits': ['PNG frames decode the actual lossy source MP4s; these are not raw VAE values.',
            'Pixel differences are not garment/identity/motion accuracy.', 'Full-context C is reused because OFF passed exact complete conformance.']}
    (output / 'review.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'images': len(records), 'source_frames': len(frames), 'wall_seconds': record['wall_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    review(p.parse_args())
