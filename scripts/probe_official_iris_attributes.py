#!/usr/bin/env python3
"""Atomic appearance diagnostic for exposed ns090, using only its raw script."""
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
from utils.agent_attribute_diagnosis import AttributeEvidenceComparer, freeze_requirements, validate_required_aspects
from utils.agent_runtime import TemporalInput, file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time()}
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    try:
        run = args.run.resolve()
        assert json.loads((run / 'status.json').read_text())['status'] == 'completed'
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['source_benchmark_index'] == 90
        config = yaml.safe_load((run / 'config.yaml').read_text())
        prompts = [x['prompts'][0] for x in config['timed_segments']]
        description = ' '.join(prompts)
        # This diagnostic explicitly enumerates the four literal appearance
        # predicates in the exposed raw script; no archived age/body additions.
        rows = [{'part': 'hair', 'aspect': 'color', 'expected': 'red'},
                {'part': 'hair', 'aspect': 'style', 'expected': 'curly'},
                {'part': 'garment', 'aspect': 'color', 'expected': 'moss green'},
                {'part': 'garment', 'aspect': 'type', 'expected': 'coat'}]
        required = freeze_requirements(description, rows)
        validate_required_aspects(description, required)
        for row in required:
            first = next(i for i, prompt in enumerate(prompts) if row['expected'] in prompt)
            assert first <= args.segment, 'Do not evaluate requirements not yet introduced in this diagnostic'
        video, = list((run / 'videos').glob('*.mp4'))
        capture = cv2.VideoCapture(str(video))
        indices = [args.segment * 160 + offset for offset in (32, 80, 128)]
        images = []
        try:
            for frame in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                assert ok
                path = out / f'frame_{frame:04d}.png'
                cv2.imwrite(str(path), image)
                images.append(path)
        finally:
            capture.release()
        client = FingerprintedVLMClient(base_url=args.base_url, model='Qwen/Qwen3-VL-8B-Instruct',
            cache_dir=out / 'vlm_cache', seed=0, max_tokens=2048, max_parse_retries=0)
        comparer = AttributeEvidenceComparer(client, max_rechecks=1)
        entity = SimpleNamespace(name='Iris', description=description)
        comparer._requirements[(entity.name, entity.description)] = required
        source_hashes = {}
        for name in ['scripts/probe_official_iris_attributes.py', 'utils/agent_attribute_diagnosis.py',
                     'utils/agent_vlm.py', 'utils/agent_runtime.py', 'utils/agent_entity_judge.py']:
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
            source_hashes[name] = file_sha256(dest)
        provenance = {'run_manifest_sha256': file_sha256(run / 'manifest.json'),
            'video_sha256': file_sha256(video), 'raw_script': prompts, 'frozen_requirements': required,
            'requirement_selection': 'Four explicitly enumerated literal appearance predicates; no inferred age/build',
            'source_benchmark_index': 90, 'held_out': False, 'segment_zero_based': args.segment,
            'frame_indices': indices, 'input_sha256': {p.name: file_sha256(p) for p in images},
            'source_sha256': source_hashes, 'sampling': client.sampling_config,
            'model': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
            'automatic_entity_tracking': False, 'input_scope': 'Whole frames of a single-subject development script',
            'limits': ['Same development judge; not independent final evaluation.',
                       'Appearance-only observations cannot establish identity, action or quality preservation.',
                       'No age/build constraint or manual pixel mask is supplied.',
                       'No candidate is selected or promoted by this script.']}
        (out / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
        inputs = tuple([TemporalInput(kind='fixed_authored_script', identifier=str(run / 'config.yaml'),
                    sha256=file_sha256(run / 'config.yaml'), available_after_shot=-1)] +
                [TemporalInput(kind='current_shot_frame', identifier=str(p), sha256=file_sha256(p),
                    available_after_shot=args.segment) for p in images])
        result = comparer.compare(entity, None, images, shot_index=args.segment, inputs=inputs)
        report = {'phase': 'official_iris_appearance_diagnostic', 'result': result,
                  'wall_seconds': time.monotonic() - start, 'candidate_promoted': False,
                  'independent_final_evaluation': False, 'quality_claim': False}
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        status.update(status='completed', wall_seconds=time.monotonic() - start)
        print(json.dumps(result['attribute_assessments'], indent=2), flush=True)
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}',
                      wall_seconds=time.monotonic() - start)
        raise
    finally:
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--segment', type=int, default=2)
    main(parser.parse_args())
