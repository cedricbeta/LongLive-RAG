#!/usr/bin/env python3
"""Run frozen existing identity controls through CPU face evidence, once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.agent_face_evidence import FaceEvidence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'started_unix': time.time(), 'pid': os.getpid(), 'device': 'cpu'}
    (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    try:
        control = json.loads(args.control_manifest.read_text())
        assets = json.loads((args.model_dir / 'manifest.json').read_text())
        assert all(sha(args.model_dir / item['file']) == item['sha256'] for item in assets['files'])
        for name, digest in control['input_sha256'].items():
            assert sha(args.control_manifest.parent / name) == digest
        cases = []
        for case in control['cases']:
            if not args.include_unlabelled and case['expected_control_status'] is None:
                continue
            row = dict(case)
            for key in ['frames_a', 'frames_b']:
                row[key] = [str(args.control_manifest.parent / 'inputs' / Path(p).name) for p in row[key]]
            cases.append(row)
        manifest = {'phase': 'synthetic_character_face_evidence_controls', 'cases': cases,
            'control_manifest_sha256': sha(args.control_manifest), 'model_assets': assets,
            'cv2_version': cv2.__version__, 'threads': 4, 'score_threshold': 0.9,
            'nms_threshold': 0.3, 'top_k': 5000, 'cosine_threshold': 0.363,
            'source_sha256': {}, 'independent_final_evaluation': False,
            'limits': ['Tiny constructed development controls do not calibrate population identity error.',
                       'Unlabelled archived repair is excluded unless explicitly requested at launch.',
                       'Detector confidence does not certify identity or facial visibility quality.',
                       'Aligned embeddings may still respond to nuisance appearance and pose.',
                       'No script attributes or planner text are passed to either vision network.',
                       'No candidate is accepted and the existing guard is unchanged.']}
        for name in ['scripts/run_face_evidence_probe.py', 'utils/agent_face_evidence.py']:
            dest = folder / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
            manifest['source_sha256'][name] = sha(dest)
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        engine = FaceEvidence(args.model_dir)
        observed, rows = {}, []
        evidence_dir = folder / 'evidence'
        evidence_dir.mkdir()
        for case in cases:
            for path in case['frames_a'] + case['frames_b']:
                key = sha(path)
                if key not in observed:
                    record, crop = engine.observe(path)
                    record.update(image_sha256=key, source_filename=Path(path).name)
                    shutil.copy2(path, evidence_dir / (key + '.png'))
                    if crop is not None:
                        cv2.imwrite(str(evidence_dir / (key + '_aligned.png')), crop)
                    observed[key] = record
            result = engine.compare([observed[sha(p)] for p in case['frames_a']],
                                    [observed[sha(p)] for p in case['frames_b']])
            expected = case['expected_control_status']
            row = {'id': case['id'], 'expected_control_status': expected, **result,
                   'control_matches_expectation': None if expected is None else
                    result['diagnostic_decision']['status'] == expected}
            rows.append(row)
            (folder / 'partial_results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(case['id'], row['diagnostic_decision'], flush=True)
        (folder / 'observations.json').write_text(json.dumps(observed, indent=2) + '\n')
        result = {'rows': rows, 'constructed_controls': sum(r['expected_control_status'] is not None for r in rows),
                  'controls_matching_expectation': sum(r['control_matches_expectation'] is True for r in rows),
                  'wall_seconds': time.monotonic() - started, 'device': 'cpu',
                  'integrated_into_acceptance': False, 'candidate_promoted': False,
                  'independent_final_evaluation': False, 'limits': manifest['limits']}
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(status='completed', wall_seconds=time.monotonic() - started)
    except Exception as error:
        status.update(status='execution_failed', error=f'{type(error).__name__}: {error}',
                      wall_seconds=time.monotonic() - started)
        raise
    finally:
        (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-manifest', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-unlabelled', action='store_true')
    run(parser.parse_args())
