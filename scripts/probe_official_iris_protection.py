#!/usr/bin/env python3
"""Development-only paired protection on the official Iris videos; no adoption."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.agent_attribute_diagnosis import freeze_requirements
from utils.agent_progressive_repair import PairedVisualProtection
from utils.agent_runtime import ShotObservation, TemporalInput, file_sha256, fingerprint
from utils.agent_vlm import FingerprintedVLMClient


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time()}
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    try:
        assert args.segment in (2, 4), 'Frozen current / bounded later diagnostic only'
        config = yaml.safe_load((args.parent / 'config.yaml').read_text())
        prompts = [row['prompts'][0] for row in config['timed_segments']]
        required = freeze_requirements(' '.join(prompts), [
            {'part': 'hair', 'aspect': 'color', 'expected': 'red'},
            {'part': 'hair', 'aspect': 'style', 'expected': 'curly'},
            {'part': 'garment', 'aspect': 'color', 'expected': 'moss green'},
            {'part': 'garment', 'aspect': 'type', 'expected': 'coat'}])
        constraints = [dict(row, kind='attribute', entity='Iris', parameters={
            'part': row['part'], 'aspect': row['aspect'], 'expected': row['expected']}) for row in required]
        target = [row for row in constraints if row['id'] == '4d74f1ec43a851fa']
        assert len(target) == 1
        indices = [args.segment * 160 + x for x in (32, 80, 128)]
        videos = {}
        observations = []
        for role, folder in [('parent', args.parent), ('candidate', args.candidate)]:
            assert json.loads((folder / 'status.json').read_text())['status'] == 'completed'
            assert json.loads((folder / 'manifest.json').read_text())['source_benchmark_index'] == 90
            assert yaml.safe_load((folder / 'config.yaml').read_text())['timed_segments'] == config['timed_segments']
            video, = (folder / 'videos').glob('*.mp4')
            videos[role] = file_sha256(video)
            paths = []
            capture = cv2.VideoCapture(str(video))
            try:
                for frame in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    assert ok
                    path = out / f'{role}_{frame:04d}.png'
                    cv2.imwrite(str(path), image)
                    paths.append(str(path))
            finally:
                capture.release()
            hashes = [file_sha256(path) for path in paths]
            observations.append(ShotObservation(shot_index=args.segment, attempt=0, frames=tuple(paths),
                crops=(), telemetry={}, inputs=tuple(TemporalInput('paired_diagnostic_frame', path, digest,
                    args.segment) for path, digest in zip(paths, hashes)), input_fingerprint=fingerprint(hashes)))
        source_hashes = {}
        for name in ['scripts/probe_official_iris_protection.py', 'utils/agent_progressive_repair.py',
                     'utils/agent_vlm.py', 'utils/agent_runtime.py', 'utils/agent_attribute_diagnosis.py']:
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
            source_hashes[name] = file_sha256(dest)
        fixed = {'source': 'Unmodified authored prompts through the checked segment',
                 'raw_prompts': prompts[:args.segment + 1],
                 'current_prompt': prompts[args.segment], 'segment_zero_based': args.segment,
                 'entity_name': 'Iris', 'literal_appearance_predicates': required}
        manifest = {'source_sha256': source_hashes, 'video_sha256': videos,
                    'frame_indices': indices, 'fixed_script': fixed, 'model': 'Qwen/Qwen3-VL-8B-Instruct',
                    'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                    'protocol': 'existing_explicit_domain_scopes_v2', 'max_model_calls': 1,
                    'candidate_retained': False, 'automatic_controller': False,
                    'limits': ['Known unreliable development guard, not adopted or independent.',
                               'Three aligned still frames cannot certify full action, motion or video quality.',
                               'No planner explanation, manual reference or inferred age/build supplied.',
                               'Segment 4 is the sole sampled later window; no whole-suffix guarantee.']}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        client = FingerprintedVLMClient(base_url=args.base_url, model=manifest['model'],
            cache_dir=out / 'vlm_cache', seed=0, max_tokens=2048, max_parse_retries=0)
        parent, candidate = observations
        review = PairedVisualProtection(client, protocol='v2').review(parent, candidate, parent,
            SimpleNamespace(to_dict=lambda: fixed), target_requirements=target,
            protected_requirements=[row for row in constraints if row not in target])
        result = {'review': review, 'candidate_retained': False, 'quality_claim': False,
                  'wall_seconds': time.monotonic() - started, 'independent_final_evaluation': False}
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(status='completed', wall_seconds=time.monotonic() - started)
        print(json.dumps(review['checks']), flush=True)
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}',
                      wall_seconds=time.monotonic() - started)
        raise
    finally:
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'candidate', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--segment', type=int, choices=[2, 4], required=True)
    main(parser.parse_args())
