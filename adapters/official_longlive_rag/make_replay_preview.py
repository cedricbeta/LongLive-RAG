#!/usr/bin/env python3
"""Fixed-time side-by-side artifact; no labels of improvement or preservation."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from repair_adapter import file_hash


def main(args):
    indices = args.frames
    caps = [cv2.VideoCapture(str(p)) for p in [args.parent, args.candidate]]
    assert all(c.isOpened() and c.get(cv2.CAP_PROP_FPS) == 16 for c in caps)
    canvas = Image.new('RGB', (832, len(indices) * 270), 'white')
    font = ImageFont.truetype('DejaVuSans.ttf', 16)
    draw = ImageDraw.Draw(canvas)
    try:
        for row, index in enumerate(indices):
            for column, cap in enumerate(caps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Missing actual video frame')
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tile = Image.fromarray(rgb).resize((416, 240), Image.Resampling.LANCZOS)
                canvas.paste(tile, (column * 416, row * 270 + 30))
                name = 'Parent' if column == 0 else 'Candidate: quality unjudged'
                draw.text((column * 416 + 6, row * 270 + 5), f'{name}  {index / 16:g}s', fill='black', font=font)
        canvas.save(args.output)
        args.output.with_suffix('.json').write_text(json.dumps({
            'frame_indices': indices, 'fps': 16, 'parent_sha256': file_hash(args.parent),
            'candidate_sha256': file_hash(args.candidate), 'source_sha256': file_hash(Path(__file__)),
            'preview_sha256': file_hash(args.output),
            'scope': 'Post-run development preview; not human labels or independent quality evidence.'}, indent=2) + '\n')
    finally:
        for cap in caps:
            cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'candidate', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[200, 440, 880])
    main(parser.parse_args())
