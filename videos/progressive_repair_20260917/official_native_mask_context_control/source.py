#!/usr/bin/env python3
"""Programmatic CLIPSeg mask projection for the continuous 16-fps timing path.

This is an explicit CLIPSeg-only localization adapter, not an identity detector
or a validated replacement for the existing SAM2/person-track method. All four
exported frames associated with a latent are segmented; their union is bounded
by the entity phrase and excludes the selected protected part in each frame.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from repair_adapter import file_hash


def exported_frames(latent_index, total_frames):
    # The omitted warm-up frame has no exported pixel. Its estimate uses the
    # first visible frame; it overlaps latent 1's evidence and is not independent.
    indices = [0] if latent_index == 0 else list(range(4 * latent_index - 4, 4 * latent_index))
    if min(indices) < 0 or max(indices) >= total_frames:
        raise ValueError('Latent projection exceeds the actual exported video')
    return indices


def entity_crop_box(entity, image_size, threshold):
    """Tight program-derived box with one grid cell of context; no manual box."""
    ys, xs = torch.where(entity >= threshold)
    if not len(xs):
        return None
    height, width = entity.shape
    pixel_width, pixel_height = image_size
    x0, x1 = max(0, int(xs.min()) - 1), min(width, int(xs.max()) + 2)
    y0, y1 = max(0, int(ys.min()) - 1), min(height, int(ys.max()) + 2)
    return (x0 * pixel_width // width, y0 * pixel_height // height,
            (x1 * pixel_width + width - 1) // width, (y1 * pixel_height + height - 1) // height)


def contextual_part(model, processor, frames, entity_soft, phrase, threshold):
    boxes = [entity_crop_box(soft, frame.size, threshold) for soft, frame in zip(entity_soft, frames)]
    valid = [i for i, box in enumerate(boxes) if box is not None]
    result = torch.zeros_like(entity_soft)
    if valid:
        inputs = processor(text=[phrase] * len(valid), images=[frames[i].crop(boxes[i]) for i in valid],
                           return_tensors='pt', padding=True)
        with torch.inference_mode():
            logits = model(**inputs).logits
            if logits.ndim == 2:
                logits = logits[None]
            logits = logits[:, None]
            for output_index, index in enumerate(valid):
                x0, y0, x1, y1 = boxes[index]
                local = torch.nn.functional.interpolate(logits[output_index:output_index + 1].sigmoid(),
                    size=(y1 - y0, x1 - x0), mode='bilinear', align_corners=False)
                full = torch.zeros((1, 1, frames[index].height, frames[index].width), dtype=local.dtype)
                full[:, :, y0:y1, x0:x1] = local
                result[index] = torch.nn.functional.interpolate(full, size=entity_soft.shape[-2:],
                    mode='bilinear', align_corners=False)[0, 0]
    return result, boxes


def run(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'started_unix': time.time(), 'pid': __import__('os').getpid()}
    (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    try:
        parent = args.parent.resolve()
        audit = json.loads((parent / 'audit.json').read_text())
        timing_path = parent / 'diagnostics/timing_0000.json'
        timing = json.loads(timing_path.read_text())
        if timing['protocol'] != 'continuous_vae_drop_first_frame_v1':
            raise ValueError('Unsupported timing coordinate system')
        segment = timing['segments'][args.segment]
        first = segment['start_latent_frame']
        last = first + segment['latent_frames']
        video, = list((parent / 'videos').glob('*.mp4'))
        latent = parent / 'diagnostics/latents_0000.pt'
        if file_hash(video) != audit['video_sha256'] or file_hash(latent) != audit['latent_sha256']:
            raise ValueError('Parent run artifacts changed')
        grid = audit['latent_shape'][-2:]
        assets = {p.name: file_hash(p) for p in args.model.iterdir() if p.is_file()}
        manifest = {'parent_video_sha256': audit['video_sha256'], 'parent_latents_sha256': audit['latent_sha256'],
                    'parent_timing_sha256': file_hash(timing_path), 'latent_range': [first, last], 'grid': grid,
                    'method': 'CLIPSeg entity intersection part minus excluded region; temporal max union',
                    'entity_phrase': args.entity_phrase, 'part_phrase': args.part_phrase,
                    'exclude_phrase': args.exclude_phrase, 'threshold': args.threshold,
                    'part_context': args.part_context,
                    'model_path': str(args.model.resolve()), 'model_file_sha256': assets,
                    'source_sha256': file_hash(Path(__file__)), 'independent_quality_evaluation': False,
                    'manual_pixel_coordinates': False,
                    'limitations': ['Text segmentation is not calibrated entity assignment.',
                                    'Grid projection approximates VAE receptive fields; latent pinning does not guarantee pixel pinning.',
                                    'Warm-up latent zero, if selected, borrows frame-zero evidence.',
                                    'CLIPSeg-only localization differs from the existing SAM2/person-track path.']}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        torch.set_num_threads(8)
        processor = CLIPSegProcessor.from_pretrained(str(args.model), local_files_only=True)
        model = CLIPSegForImageSegmentation.from_pretrained(str(args.model), local_files_only=True).eval()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) != 16:
            raise ValueError('Expected readable 16-fps parent video')
        masks, rows = [], []
        try:
            for index in range(first, last):
                indices = exported_frames(index, timing['export_frames'])
                frames = []
                for frame in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, bgr = cap.read()
                    if not ok:
                        raise ValueError('Missing source frame: ' + str(frame))
                    frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                phrases = [args.entity_phrase, args.part_phrase, args.exclude_phrase]
                inputs = processor(text=[phrase for phrase in phrases for _ in frames],
                                   images=frames * len(phrases), return_tensors='pt', padding=True)
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    soft = torch.nn.functional.interpolate(logits.sigmoid()[:, None], size=grid,
                        mode='bilinear', align_corners=False)[:, 0].reshape(3, len(frames), *grid)
                boxes = None
                if args.part_context == 'entity_box':
                    # Keep entity and protected-face maps identical to the full
                    # frame path. Only the part model's image context changes.
                    soft[1], boxes = contextual_part(model, processor, frames, soft[0],
                                                    args.part_phrase, args.threshold)
                binary = (soft[0] >= args.threshold) & (soft[1] >= args.threshold) & (soft[2] < args.threshold)
                projected = binary.any(dim=0)
                masks.append(projected)
                rows.append({'latent_frame': index, 'source_export_frames': indices,
                             'source_rgb_sha256': [hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in frames],
                             'latent_grid_coverage': float(projected.float().mean())})
                if boxes is not None:
                    rows[-1]['program_entity_boxes_xyxy'] = boxes
                (folder / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
                if index in {first, (first + last) // 2, last - 1}:
                    image = np.asarray(frames[0]).copy()
                    expanded = cv2.resize(projected.numpy().astype('uint8'), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    image[expanded] = (0.55 * image[expanded] + 0.45 * np.array([255, 0, 255])).astype('uint8')
                    Image.fromarray(image).save(folder / f'mask_latent_{index:04d}.png')
                print(index, rows[-1]['latent_grid_coverage'], flush=True)
        finally:
            cap.release()
        mask = torch.stack(masks)
        torch.save({**manifest, 'mask': mask}, folder / 'mask.pt')
        result = {'mask_sha256': file_hash(folder / 'mask.pt'), 'nonempty': bool(mask.any()),
                  'mean_coverage': float(mask.float().mean()), 'rows': rows,
                  'wall_seconds': time.monotonic() - started, 'device': 'cpu',
                  'validated_entity_mask': False, 'quality_claim': False}
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(status='completed', wall_seconds=time.monotonic() - started)
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}', wall_seconds=time.monotonic() - started)
        raise
    finally:
        (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--segment', type=int, required=True)
    parser.add_argument('--entity-phrase', required=True)
    parser.add_argument('--part-phrase', required=True)
    parser.add_argument('--exclude-phrase', default='face')
    parser.add_argument('--threshold', type=float, default=0.35)
    parser.add_argument('--part-context', choices=['full_frame', 'entity_box'], default='full_frame')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('threshold must lie strictly between zero and one')
    run(args)
