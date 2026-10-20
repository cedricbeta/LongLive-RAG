#!/usr/bin/env python3
"""Prepare fresh OFF/ZERO replay attempts against a reproduced timed parent."""
import argparse
from argparse import Namespace
import json
from pathlib import Path
import shutil
import sys

import torch
import yaml

from prepare_repair import prepare
from repair_adapter import file_hash


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    parent = args.parent.resolve()
    reference = torch.load(parent / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
    timing = json.loads((parent / 'diagnostics/timing_0000.json').read_text())
    segment = timing['segments'][1]
    first = segment['start_latent_frame']
    last = first + segment['latent_frames']
    video, = list((parent / 'videos').glob('*.mp4'))
    payload = {'parent_latents_sha256': file_hash(parent / 'diagnostics/latents_0000.pt'),
               'parent_video_sha256': file_hash(video),
               'parent_timing_sha256': file_hash(parent / 'diagnostics/timing_0000.json'),
               'latent_range': [first, last], 'grid': list(reference.shape[-2:]),
               'mask': torch.zeros(last - first, *reference.shape[-2:], dtype=torch.bool)}
    torch.save(payload, root / 'zero_mask.pt')
    prepared = prepare(Namespace(parent=parent, checkout=args.checkout, output=root / 'zero',
        parent_branch=args.parent_branch, assets_manifest=args.assets_manifest, mask=root / 'zero_mask.pt',
        prompt=args.prompt, noise_seed=4243, gpu=args.zero_gpu, timeout=1800))
    output = root / 'off'
    output.mkdir()
    config = yaml.safe_load((root / 'zero/config.yaml').read_text())
    config.pop('repair_request')
    config.update(data_path=str(output / 'initial_prompt.txt'), output_folder=str(output / 'videos'),
                  adapter_diagnostics_dir=str(output / 'diagnostics'))
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(root / 'zero/initial_prompt.txt', output / 'initial_prompt.txt')
    manifest = json.loads((root / 'zero/manifest.json').read_text())
    for key in ['action', 'action_fingerprint']:
        manifest.pop(key)
    manifest.update(gpu=args.off_gpu, repair_enabled=False, phase='official_rag_disabled_repair_control',
        config_sha256=file_hash(output / 'config.yaml'),
        command=[sys.executable, '-u', 'inference.py', '--config_path', str(output / 'config.yaml')])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__).with_name('run_prepared.py'), output / 'supervisor.py')
    print(json.dumps({'off': [sys.executable, str(output / 'supervisor.py'), str(output)],
                      'zero': prepared['command'],
                      'audit': [sys.executable, str(Path(__file__).with_name('audit_repair_controls.py').resolve()),
                                '--root', str(root), '--parent', str(parent)]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'checkout', 'output', 'assets-manifest', 'prompt']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent-branch', required=True)
    parser.add_argument('--off-gpu', type=int, required=True)
    parser.add_argument('--zero-gpu', type=int, required=True)
    torch.set_num_threads(4)
    main(parser.parse_args())
