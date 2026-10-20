#!/usr/bin/env python3
"""Display all measured arms without selecting one by coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run, output):
    output.mkdir(parents=True, exist_ok=False)
    data = np.load(run / 'measured_masks.npz')
    rows = json.loads((run / 'result.json').read_text())['rows']
    folder, = (run / 'entities/frames').iterdir()
    conditions = [('generic', 'per_frame'), ('literal_head', 'per_frame'), ('generic', 'propagated'), ('literal_head', 'propagated')]
    for name in sorted({r['entity'] for r in rows}):
        fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
        for row, frame in enumerate([0, 80, 159]):
            frame_image = np.array(Image.open(folder / f'chunk{frame:02d}.png').convert('RGB'))
            for col, (prompt_mode, time_mode) in enumerate(conditions):
                mask = data[f'{name}_{prompt_mode}_{time_mode}_final'][frame]
                display = np.array(Image.fromarray(mask).resize((frame_image.shape[1], frame_image.shape[0]), Image.Resampling.NEAREST))
                overlay = np.zeros((*display.shape, 4))
                overlay[display] = [1., .28, .05, .55]
                ax = axes[row, col]
                ax.imshow(frame_image)
                ax.imshow(overlay)
                ax.axis('off')
                ax.set_title(f'{prompt_mode} / {time_mode}\nframe {frame}, {mask.sum()} cells', fontsize=10)
        fig.suptitle(f'{name}: measured garment masks, four frozen conditions\nAutomatic first-frame seeds; unchanged thresholds/person/face protection. Area is not accuracy; no arm adopted.', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(output / (name + '_four_conditions.png'), dpi=120)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)
