#!/usr/bin/env python3
"""Reconstruct terminal native evidence from a portable export, without models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def verify(root, name):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    folder = root / 'results' / name
    export = read(folder / 'export_manifest.json')
    for path, expected in export['files_sha256'].items():
        assert sha(folder / path) == expected, path
    manifest = read(folder / 'run/manifest.json')
    for path, expected in manifest['source_sha256'].items():
        assert sha(folder / 'run/source' / path) == expected, path
    plan = read(folder / 'plan.json')
    for path, expected in plan['source_sha256'].items():
        candidates = [folder / 'run/source' / path, folder / 'policy_source' / path]
        assert any(p.exists() and sha(p) == expected for p in candidates), path
    origin = Path(manifest['arguments']['output']).parent
    images = set()
    for path in (folder / 'run/vlm_cache').rglob('*.json'):
        call = read(path)
        assert len(call['ordered_image_paths']) == len(call['ordered_image_hashes'])
        for original, expected in zip(call['ordered_image_paths'], call['ordered_image_hashes']):
            relative = Path(original).relative_to(origin)
            assert sha(folder / relative) == expected, str(relative)
            images.add(str(relative))
    assert len(images) == export['actual_request_images_verified']
    audit = read(folder / 'audit/audit.json')
    slices = 0
    for row in audit['masks']:
        path = folder / 'audit' / row['array_file']
        assert sha(path) == row['array_sha256']
        metadata = read(path.with_name(path.stem + '_metadata.json'))
        with np.load(path) as arrays:
            positions = {int(x): i for i, x in enumerate(arrays['export_frame_indices'])}
            for i, mapping in enumerate(metadata['source_frame_mapping']):
                ix = [positions[k] for k in mapping['export_frames']]
                region = (arrays['frame_masks'][ix, 0] >= .35).any(0)
                protected = (arrays['protected'][ix, 0] >= .35).any(0)
                assert np.array_equal(arrays['native_mask'][i], region & ~protected)
                slices += 1
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'summary.json'
        subprocess.run([sys.executable, str(root / 'implementation/scripts/summarize_native_feedback.py'),
                        '--root', str(root), '--name', name, '--output', str(output)],
                       cwd=tmp, check=True, capture_output=True, text=True)
        assert output.read_bytes() == (folder / 'summary.json').read_bytes()
    summary = read(folder / 'summary.json')
    for path, expected in summary['inputs_sha256'].items():
        assert sha(root / path) == expected, path
    artifacts = root / 'artifacts' / name
    binding = read(artifacts / 'artifact_binding.json')
    for path, expected in binding['artifact_files_sha256'].items():
        assert sha(artifacts / path) == expected, path
    assert sha(root / binding['source_script']) == binding['source_script_sha256']
    for item in binding['candidate_video_bindings']:
        video = root / item['published_video']
        evidence = read(root / item['engineering_audit'])
        assert sha(video) == evidence['video_sha256'] == item['video_sha256']
    return {'run': name, 'export_files_verified': len(export['files_sha256']),
            'frozen_run_sources_verified': len(manifest['source_sha256']),
            'frozen_plan_sources_verified': len(plan['source_sha256']),
            'actual_request_images_verified': len(images),
            'native_mask_projection_slices_checked': slices,
            'summary_byte_exact': True, 'candidate_videos_verified': len(binding['candidate_video_bindings']),
            'new_model_calls': 0, 'new_generated_videos': 0, 'quality_claim': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root.resolve(), args.name)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
