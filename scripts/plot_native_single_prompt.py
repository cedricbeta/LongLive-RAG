#!/usr/bin/env python3
"""Plot every frozen ten-second sample and the final frame of the duration pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root, name, output):
    if output.exists():
        raise FileExistsError('Use a fresh figure path')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    folder = root / 'results' / name
    audit_path = folder / 'audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['engineering_checks_passed'] and audit['single_prompt_unchanged']
    frames = audit['sample_frames']
    assert [r['frame'] for r in frames] == list(range(0, 1920, 160)) + [1919]
    inputs = {str(audit_path.relative_to(root)): sha(audit_path)}
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, frames):
        path = folder / row['path']
        assert sha(path) == row['sha256']
        inputs[str(path.relative_to(root))] = row['sha256']
        ax.imshow(plt.imread(path))
        ax.set_title(f"Frame {row['frame']} / {row['seconds']:g} s", fontsize=11)
    fig.suptitle('MovieGenBench prompt 0, seed 0, 120 seconds: fixed samples, no quality labels', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'inputs_sha256': inputs,
        'figure_sha256': sha(output), 'source_sha256': sha(Path(__file__)),
        'matplotlib_version': matplotlib.__version__, 'actual_human_labels': 0,
        'new_model_calls': 0, 'new_generated_videos': 0}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='official_moviegen_000_120s_v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plot(args.root.resolve(), args.name, args.output.resolve())
