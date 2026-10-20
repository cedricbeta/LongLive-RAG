#!/usr/bin/env python3
"""Reconstruct an observation probe's frames, transcripts and costs without models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def audit(root, folder, telemetry):
    import cv2
    import numpy as np
    from PIL import Image
    inputs = {}
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(path):
        inputs[str(path.relative_to(folder.parent))] = digest(path)
        return json.loads(path.read_text())
    plan, status = read(folder / 'plan.json'), read(folder / 'status.json')
    assert status['status'] == 'completed' and status['returncode'] == 0
    assert digest(folder / 'runner.py') == plan['source_sha256']
    model = read(folder / 'model_manifest.json')
    assert digest(folder / 'model_manifest.json') == plan['model_manifest_sha256']
    for path in sorted((folder / 'model_source').iterdir()):
        assert digest(path) == model['files'][path.name]['sha256']
        inputs[str(path.relative_to(folder.parent))] = digest(path)
    result = read(folder / 'result.json')
    partial = read(folder / 'partial_results.json')
    assert result['rows'] == partial and len(partial) == len(plan['cases']) == plan['max_calls']
    decoded = 0
    source_videos = {}
    for video in plan['videos']:
        path = root / video['repository_path']
        assert digest(path) == video['sha256']
        source_videos[video['repository_path']] = video['sha256']
        cap = cv2.VideoCapture(str(path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 960 and cap.get(cv2.CAP_PROP_FPS) == 16
        for case in plan['cases']:
            if case['branch'] != video['branch'] or case['control'] is not None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, case['frame'])
            ok, frame = cap.read()
            assert ok
            with Image.open(folder / case['image']) as image:
                assert np.array_equal(np.asarray(image.convert('RGB')), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            decoded += 1
        cap.release()
    descriptions, controls = [], []
    total_call_seconds = 0
    for case, row in zip(plan['cases'], partial):
        assert row['id'] == case['id'] and row == read(folder / 'transcripts' / (case['id'] + '.json'))
        path = folder / case['image']
        assert digest(path) == case['image_sha256'] == row['image_sha256']
        inputs[str(path.relative_to(folder.parent))] = digest(path)
        assert row['prompt'] == plan['prompt'] and row['generation'] == plan['generation']
        assert row['seed'] == plan['seed_per_call']
        if row['status'] == 'completed':
            assert json.loads(row['reply']) == row['parsed']
            parsed = row['parsed']
            assert set(parsed) == {'people_count', 'clothing', 'hair', 'uncertainty'}
            assert type(parsed['people_count']) is int and parsed['people_count'] >= 0
            assert all(isinstance(parsed[k], str) for k in ('clothing', 'hair', 'uncertainty'))
        elif row['status'] == 'parse_failed':
            assert 'reply' in row and 'error' in row
        else:
            assert row['status'] == 'execution_failed' and 'error' in row
        if 'output_token_ids' in row:
            assert len(row['output_token_ids']) <= plan['generation']['max_new_tokens']
        total_call_seconds += row['wall_seconds']
        descriptions.append({**case, 'status': row['status'], 'reply': row.get('reply'),
            'parsed': row.get('parsed'), 'error': row.get('error'),
            'wall_seconds': row['wall_seconds']})
        if case['control'] == 'same_image_repeat':
            original_case = next(c for c in plan['cases'] if c['branch'] == case['branch']
                and c['frame'] == case['frame'] and c['control'] is None)
            original = next(r for r in partial if r['id'] == original_case['id'])
            assert original_case['image_sha256'] == case['image_sha256']
            controls.append({'id': case['id'], 'kind': case['control'],
                'same_input_tensors': original.get('input_tensors') == row.get('input_tensors'),
                'same_output_tokens': original.get('output_token_ids') == row.get('output_token_ids'),
                'same_reply': original.get('reply') == row.get('reply')})
        elif case['control'] == 'no_person':
            with Image.open(path) as image:
                assert np.all(np.asarray(image) == 127)
            controls.append({'id': case['id'], 'kind': case['control'],
                'reported_zero_people': row.get('parsed', {}).get('people_count') == 0})
    assert decoded == sum(c['control'] is None for c in plan['cases'])
    assert result['actual_calls'] == sum(r.get('model_call_started', False) for r in partial)
    assert result['completed'] == sum(r['status'] == 'completed' for r in partial)
    tplan, measured = read(telemetry / 'plan.json'), read(telemetry / 'summary.json')
    tstatus = read(telemetry / 'target_terminal_status.json')
    assert tstatus == status and measured['target_status_sha256'] == digest(telemetry / 'target_terminal_status.json')
    assert measured['samples_sha256'] == digest(telemetry / 'samples.jsonl')
    assert tplan['target_plan_sha256'] == digest(folder / 'plan.json')
    assert tplan['target_manifest_sha256'] == digest(folder / 'manifest.json')
    assert tplan['source_sha256'] == digest(telemetry / 'monitor_source.py')
    spec = importlib.util.spec_from_file_location('probe_telemetry', telemetry / 'monitor_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [json.loads(s) for s in (telemetry / 'samples.jsonl').read_text().splitlines()]
    inputs[str((telemetry / 'samples.jsonl').relative_to(folder.parent))] = digest(telemetry / 'samples.jsonl')
    assert module.summarize(samples, status) == {k: v for k, v in measured.items() if k not in ('samples_sha256', 'target_status_sha256')}
    return {'phase': 'target_blind_different_family_development_observations',
        'model': plan['model'], 'model_revision': plan['model_revision'],
        'actual_calls': result['actual_calls'], 'completed': result['completed'],
        'supervisor_seconds': status['wall_seconds'], 'worker_seconds': result['worker_seconds'],
        'model_load_seconds': result['model_load_seconds'], 'sum_call_wall_seconds': total_call_seconds,
        'cost_scope': 'Nested wall intervals; do not add. Supervisor includes model-file verification, loading and all calls. Initial CPU preparation and failed observer import are separate.',
        'actual_frames_reproduced': decoded, 'controls': controls, 'descriptions': descriptions,
        'telemetry': measured, 'source_video_sha256': source_videos, 'inputs_sha256': inputs,
        'audit_source_sha256': digest(Path(__file__)), 'models_reexecuted_by_audit': False,
        'model_weights_rehashed_by_audit': False, 'held_out': False, 'actual_human_labels': 0,
        'adopted_into_acceptance': False, 'new_generated_videos': 0,
        'limits': ['One exposed script and correlated branch/time views, not independent samples or calibrated accuracy.',
            'Target-blind observations use a different model family and a simpler prompt/runtime than the controller; this is not a single-factor model comparison.',
            'Raw descriptions are model outputs, not human labels or automatic repair decisions.',
            'Sparse frames do not establish motion, identity preservation or whole-video correctness.',
            'No original/candidate acceptance decision is changed. The observer is read-only; energy includes idle and utilization is not kernel GPU time.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root.resolve(), args.folder.resolve(), args.telemetry.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['actual_calls', 'completed', 'supervisor_seconds', 'actual_frames_reproduced', 'controls']}))
