#!/usr/bin/env python3
"""Audit a completed single-prompt timing pilot without assigning quality labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent if BASE.name == 'implementation' else BASE
sys.path.insert(0, str(ROOT / 'adapters/official_longlive_rag'))
from audit_timing import audit


def main(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['phase'] == 'official_moviegen_single_prompt_development'
    assert not manifest['correction_enabled'] and not manifest['held_out']
    result = audit(folder)
    timing = result['timing']
    prompt = (folder / 'initial_prompt.txt').read_text().rstrip('\n')
    assert len(timing['segments']) == len(timing['schedule']) == 1
    assert timing['schedule'] == {'0': [prompt]}
    assert not result['switch_checks']
    assert timing['export_frames'] == manifest['expected_output']['frames']
    video, = (folder / 'videos').glob('*.mp4')
    frames = list(range(0, timing['export_frames'], 160)) + [timing['export_frames'] - 1]
    cap = cv2.VideoCapture(str(video))
    samples = []
    try:
        for index in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            assert ok, index
            path = folder / 'sample_frames' / f'frame_{index:04d}.png'
            assert cv2.imwrite(str(path), image)
            samples.append({'frame': index, 'seconds': index / 16,
                            'path': str(path.relative_to(folder)),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    finally:
        cap.release()
    result.update(phase='official_moviegen_single_prompt_engineering',
                  source_benchmark=manifest['source_benchmark'],
                  single_prompt_unchanged=True, sample_frames=samples,
                  actual_human_labels=0, independent_quality_evaluation=False,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['limits'][0] = 'One exposed MovieGenBench development prompt; not held-out benchmark performance.'
    (folder / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('phase', 'engineering_checks_passed', 'ffprobe',
        'retrieval_query_count', 'wall_seconds', 'single_prompt_unchanged')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    main(args.folder.resolve())
