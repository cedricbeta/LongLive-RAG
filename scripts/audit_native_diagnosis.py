#!/usr/bin/env python3
"""Audit actual diagnostic region projection and export portable measured arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent if BASE.name == 'implementation' else BASE
sys.path[:0] = [str(BASE), str(ROOT / 'adapters/official_longlive_rag')]
from native_region_preview import preview


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run, output):
    output.mkdir(parents=True, exist_ok=False)
    source = run / 'diagnosis_calls.json'
    records = json.loads(source.read_text())
    files = {sha(p): p for p in run.rglob('*.pt')}
    rows = []
    for call_index, record in enumerate(records):
        parent = Path(record['native_run'])
        video, = (parent / 'videos').glob('*.mp4')
        latent = parent / 'diagnostics/latents_0000.pt'
        timing_path = parent / 'diagnostics/timing_0000.json'
        timing = json.loads(timing_path.read_text())
        for index, option in enumerate(record['verdict']['metrics']['repair_options']):
            native = torch.load(files[option['mask_sha256']], map_location='cpu', weights_only=True)
            assert native['parent_video_sha256'] == sha(video)
            assert native['parent_latents_sha256'] == sha(latent)
            assert native['parent_timing_sha256'] == sha(timing_path)
            frames = torch.load(files[native['source_mask_sha256']], map_location='cpu', weights_only=True)
            protection = torch.load(files[native['source_protected_sha256']], map_location='cpu', weights_only=True)
            for i, mapping in enumerate(native['source_frame_mapping']):
                union = torch.stack([frames['masks'][k][0] >= .35 for k in mapping['export_frames']]).any(0)
                exclusion = torch.stack([protection['masks'][k][0] >= .35 for k in mapping['export_frames']]).any(0)
                assert torch.equal(native['mask'][i], union & ~exclusion)
                assert not (native['mask'][i] & exclusion).any()
            # A parent and its child can both offer regions for the same shot.
            # Keep every call distinct instead of overwriting the parent's arrays.
            name = f"call{call_index:03d}_shot{record['shot']:02d}_option{index:02d}"
            array = output / (name + '.npz')
            np.savez_compressed(array, native_mask=native['mask'].numpy(),
                export_frame_indices=np.array(sorted(frames['masks'])),
                frame_masks=np.stack([frames['masks'][k].numpy() for k in sorted(frames['masks'])]),
                protected=np.stack([protection['masks'][k].numpy() for k in sorted(protection['masks'])]))
            metadata = {k: v for k, v in native.items() if k != 'mask'}
            (output / (name + '_metadata.json')).write_text(json.dumps(metadata, indent=2) + '\n')
            row = {'call_index': call_index, 'shot': record['shot'],
                   'attempt': record['attempt'], 'native_run': str(parent),
                   'option_id': option['id'], 'array_file': array.name,
                   'original_tensor_sha256': option['mask_sha256'], 'array_sha256': sha(array),
                   'native_latent_count': len(native['mask']),
                   'empty_latent_frames': int((native['mask'].flatten(1).sum(1) == 0).sum()),
                   'coverage': float(native['mask'].float().mean()),
                   'exact_projection_minus_protection': True, 'protected_overlap_cells': 0}
            if index == 0:
                row['preview'] = preview(option, video=str(video), parent_latents=str(latent),
                                         output_dir=output, timing=timing)
            rows.append(row)
    audit = {'diagnosis_sha256': sha(source), 'source_script_sha256': sha(Path(__file__)),
             'masks': rows, 'new_model_calls': 0, 'new_video_generations': 0,
             'quality_claim': False, 'ground_truth_masks': False}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'native_masks': len(rows), 'all_exact': all(r['exact_projection_minus_protection'] for r in rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    main(args.run.resolve(), args.output.resolve())
