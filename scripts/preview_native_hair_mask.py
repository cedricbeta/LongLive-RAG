#!/usr/bin/env python3
"""Inspect three fixed source frames with the existing automatic mask recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

adapter = next(p / 'adapters/official_longlive_rag' for p in Path(__file__).resolve().parents
               if (p / 'adapters/official_longlive_rag/prepare_mask.py').is_file())
sys.path.insert(0, str(adapter))
from prepare_mask import contextual_part
from repair_adapter import file_hash


def main(args):
    start = time.monotonic()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    plan = json.loads(args.plan.read_text())
    parent = Path(plan['parent'])
    video, = (parent / 'videos').glob('*.mp4')
    assert file_hash(video) == plan['parent_video_sha256']
    model_path = args.model.resolve()
    torch.set_num_threads(8)
    processor = CLIPSegProcessor.from_pretrained(model_path, local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(model_path, local_files_only=True).eval()
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        for index in plan['mask_preflight_frames']:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            assert ok
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    inputs = processor(text=[p for p in [args.entity_phrase, 'hair', 'face'] for _ in frames],
                       images=frames * 3, return_tensors='pt', padding=True)
    with torch.inference_mode():
        soft = torch.nn.functional.interpolate(model(**inputs).logits.sigmoid()[:, None], size=(60, 104),
            mode='bilinear', align_corners=False)[:, 0].reshape(3, len(frames), 60, 104)
    part, boxes = contextual_part(model, processor, frames, soft[0], 'hair', .35)
    soft = torch.stack([soft[0], part, soft[2]])
    binary = (soft[0] >= .35) & (soft[1] >= .35) & (soft[2] < .35)
    rows = []
    sheet = Image.new('RGB', (1664, len(frames) * 272), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (index, frame) in enumerate(zip(plan['mask_preflight_frames'], frames)):
        raw = np.asarray(frame)
        frame.save(out / f'original_{index:04d}.png')
        maps = [soft[0, i] >= .35, soft[1, i] >= .35, soft[2, i] >= .35, binary[i]]
        for column, (label, mask) in enumerate(zip(['entity', 'crop hair', 'face exclusion', 'final hair'], maps)):
            overlay = raw.copy()
            pixels = cv2.resize(mask.numpy().astype('uint8'), (832, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[pixels] = (.55 * overlay[pixels] + .45 * np.array([255, 0, 255])).astype('uint8')
            image = Image.fromarray(overlay)
            image.save(out / f'{label.replace(" ", "_")}_{index:04d}.png')
            sheet.paste(image.resize((416, 240)), (column * 416, i * 272 + 32))
            draw.text((column * 416 + 8, i * 272 + 8), f'{label} | frame {index}', fill='black')
        rows.append({'frame': index, 'rgb_sha256': hashlib.sha256(raw.tobytes()).hexdigest(), 'box': boxes[i],
            'part_cells': int((soft[1, i] >= .35).sum()),
            'person_part_cells': int(((soft[0, i] >= .35) & (soft[1, i] >= .35)).sum()),
            'removed_face_overlap_cells': int(((soft[0, i] >= .35) & (soft[1, i] >= .35) & (soft[2, i] >= .35)).sum()),
            'final_cells': int(binary[i].sum())})
    sheet.save(out / 'components.png')
    np.savez_compressed(out / 'arrays.npz', soft=soft.numpy(), final=binary.numpy())
    record = {'status': 'completed', 'rows': rows, 'wall_seconds': time.monotonic() - start,
        'entity_phrase': args.entity_phrase, 'source_sha256': file_hash(Path(__file__)),
        'source_helper_sha256': file_hash(adapter / 'prepare_mask.py'),
        'model_files_sha256': {p.name: file_hash(p) for p in model_path.iterdir() if p.is_file()},
        'plan_sha256': file_hash(args.plan), 'ground_truth_masks': False, 'new_video_generations': 0}
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'rows': rows, 'seconds': record['wall_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--entity-phrase', required=True)
    main(p.parse_args())
