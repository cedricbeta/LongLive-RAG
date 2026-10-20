#!/usr/bin/env python3
"""Retrospective CLIPSeg component diagnostic; does not choose a repair mask."""
import argparse
import json
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from repair_adapter import file_hash


def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    processor = CLIPSegProcessor.from_pretrained(args.model, local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model, local_files_only=True).eval()
    phrases = ['woman', 'clothing', 'coat', 'face']
    # Same first/middle/last latent locations used in the frozen mask overlays.
    indices = [160, 240, 316]
    cap = cv2.VideoCapture(str(args.video))
    rows, arrays = [], []
    fig, axes = plt.subplots(3, 5, figsize=(15, 6.3), constrained_layout=True)
    try:
        for row, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('Missing frozen source frame')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inputs = processor(text=phrases, images=[Image.fromarray(rgb)] * len(phrases),
                               return_tensors='pt', padding=True)
            with torch.inference_mode():
                logits = model(**inputs).logits
                soft = torch.nn.functional.interpolate(logits.sigmoid()[:, None], size=(60, 104),
                    mode='bilinear', align_corners=False)[:, 0].numpy()
            arrays.append(soft)
            binary = soft >= 0.35
            stats = {'frame': index, 'component_area': {p: float(b.mean()) for p, b in zip(phrases, binary)},
                     'component_max': {p: float(s.max()) for p, s in zip(phrases, soft)}}
            for part, part_index in [('clothing', 1), ('coat', 2)]:
                intersection = binary[0] & binary[part_index]
                final = intersection & ~binary[3]
                stats[part] = {'entity_part_area': float(intersection.mean()), 'after_face_exclusion_area': float(final.mean()),
                    'fraction_removed_by_face': float((intersection & binary[3]).sum() / max(1, intersection.sum()))}
            rows.append(stats)
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f'Export frame {index}')
            for col, phrase in enumerate(phrases, start=1):
                im = axes[row, col].imshow(soft[col - 1], vmin=0, vmax=1, cmap='viridis')
                axes[row, col].set_title(phrase)
            for axis in axes[row]:
                axis.axis('off')
        fig.colorbar(im, ax=axes[:, 1:], shrink=0.8, label='CLIPSeg sigmoid output (uncalibrated)')
        fig.savefig(args.output / 'components.png', dpi=140)
        np.savez_compressed(args.output / 'soft_maps.npz', soft=np.stack(arrays), frames=indices)
        report = {'kind': 'retrospective_localization_component_diagnostic', 'frames': indices, 'phrases': phrases,
                  'threshold': 0.35, 'grid': [60, 104], 'source_video_sha256': file_hash(args.video),
                  'source_sha256': file_hash(Path(__file__)), 'model_file_sha256':
                  {p.name: file_hash(p) for p in Path(args.model).iterdir() if p.is_file()}, 'rows': rows,
                  'wall_seconds': time.monotonic() - started, 'device': 'cpu',
                  'limits': ['Three exposed development frames, not mask ground truth.',
                             'No new mask, threshold or protection rule is selected by this diagnostic.']}
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        cap.release()
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
