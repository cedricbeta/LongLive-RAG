#!/usr/bin/env python3
"""Export a labeled native development pair; these artifacts are not evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
from PIL import Image, ImageDraw, ImageOps


def export(parent, candidate, output, frames):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    canvas = Image.new('RGB', (1040, 36 + 326 * len(frames)), '#181818')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), 'Original parent | Native automatic candidate (development; see recorded guard decision)', fill='white')
    dimensions = []
    for column, video in enumerate([parent, candidate]):
        cap = cv2.VideoCapture(str(video))
        try:
            meta = {'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 'fps': cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
            assert meta == {'frames': 960, 'fps': 16.0, 'width': 832, 'height': 480}, meta
            dimensions.append(meta)
            for row, frame in enumerate(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError('Unable to decode requested comparison frame')
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                canvas.paste(ImageOps.pad(image, (520, 300)), (520 * column, 36 + 326 * row))
                draw.text((8 + column * 520, 339 + 326 * row), f'Frame {frame} / {frame / 16:.2f}s', fill='white')
        finally:
            cap.release()
    canvas.save(output / 'current_pair.png')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin',
        '-threads', '4', '-i', str(parent), '-threads', '4', '-i', str(candidate),
        '-filter_complex_threads', '2', '-filter_complex',
        '[0:v]drawtext=text=Original parent:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:x=10:y=10[a];'
        '[1:v]drawtext=text=Native candidate:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:x=10:y=10[b];'
        '[a][b]hstack=inputs=2[v]', '-map', '[v]', '-an', '-c:v', 'libx264', '-threads', '4', '-crf', '23',
        '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'parent_candidate.mp4')]
    with (output / 'ffmpeg.log.txt').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    shutil.copy2(candidate, output / 'candidate.mp4')
    result = {'stage': 'artifact_export_not_evaluation', 'source_video_sha256': [sha(parent), sha(candidate)],
              'source_metadata': dimensions, 'sampled_export_frames': frames, 'ffmpeg_command': command,
              'wall_seconds': time.monotonic() - started, 'new_model_calls': 0, 'new_generated_videos': 0,
              'files_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
              'limits': ['Contact-sheet resize and comparison-video re-encoding are presentation only.',
                         'candidate.mp4 is an exact source copy. No rating, acceptance or identity label is inferred.']}
    (output / 'artifact_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[352, 400, 448])
    args = parser.parse_args()
    result = export(args.parent.resolve(), args.candidate.resolve(), args.output.resolve(), args.frames)
    print(json.dumps({'wall_seconds': result['wall_seconds'], 'files': list(result['files_sha256'])}))
