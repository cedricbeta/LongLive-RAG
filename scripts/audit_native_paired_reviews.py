#!/usr/bin/env python3
"""Replay paired-review inputs, failures, controls and costs without model calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def audit(root, folder, telemetry):
    import cv2
    import numpy as np
    from PIL import Image
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(folder.parent))] = sha(path)
        return json.loads(path.read_text())
    plan, status, result = [read(folder / name) for name in ('plan.json', 'status.json', 'result.json')]
    assert status['status'] == 'completed' and status['returncode'] == 0
    assert sha(folder / 'runner.py') == plan['source_sha256']
    assert sha(folder / 'support.py') == plan['support_source_sha256']
    assert sha(folder / 'compiled_script.json') == plan['compiled_script_sha256']
    model = read(folder / 'model_manifest.json')
    assert sha(folder / 'model_manifest.json') == plan['model_manifest_sha256']
    for path in (folder / 'model_source').iterdir():
        assert sha(path) == model['files'][path.name]['sha256']
        inputs[str(path.relative_to(folder.parent))] = sha(path)
    if 'paired_engine_sha256' in plan:
        assert sha(folder / 'paired_engine.py') == plan['paired_engine_sha256']
        assert sha(folder / 'reference_plan.json') == plan['reference_plan_sha256']
        assert sha(folder / 'reference_result.json') == plan['reference_result_sha256']
    rows = read(folder / 'partial_results.json')
    assert rows == result['rows']
    assert len(plan['cases']) == plan['max_calls']
    assert result['unattempted_case_ids'] == [c['id'] for c in plan['cases'][len(rows):]]
    validator = module(folder / 'runner.py', 'recorded_paired_runner').validate
    sent_images = set()
    descriptions, controls = [], []
    malformed = 0
    for case, row in zip(plan['cases'], rows):
        assert row == read(folder / 'transcripts' / (case['id'] + '.json'))
        assert row['id'] == case['id'] and row['prompt'] == case['prompt'] and row['images'] == case['images']
        assert row['generation'] == plan['generation'] and row['seed'] == plan['seed_per_call']
        for item in row['images']:
            assert sha(folder / item['image']) == item['sha256']
            inputs[str((folder / item['image']).relative_to(folder.parent))] = item['sha256']
            sent_images.add(item['image'])
        if row['status'] != 'execution_failed':
            assert row['model_call_started'] and len(row['output_token_ids']) <= plan['generation']['max_new_tokens']
            try:
                parsed = validator(json.loads(row['reply']), case['frames'])
            except (ValueError, TypeError, KeyError):
                assert row['status'] == 'parse_failed'
                malformed += 1
            else:
                assert row['status'] == 'completed' and parsed == row['parsed']
                malformed = 0
        record = {'id': row['id'], 'branches': case['branches'], 'frames': case['frames'],
            'scope': case['scope'], 'control': case['control'], 'pair_group': case['pair_group'],
            'status': row['status'], 'parsed': row.get('parsed'), 'reply': row.get('reply'),
            'error': row.get('error'), 'wall_seconds': row['wall_seconds']}
        descriptions.append(record)
        if case['control'] is not None:
            n = len(case['images']) // 2
            identical = [x['sha256'] for x in case['images'][:n]] == [x['sha256'] for x in case['images'][n:]]
            assert identical
            controls.append({**record, 'aligned_input_image_bytes_identical': True})
    assert result['actual_calls'] == sum(row.get('model_call_started', False) for row in rows)
    assert result['completed'] == sum(row['status'] == 'completed' for row in rows)
    if result['stop_reason'] == 'three_consecutive_malformed_replies':
        assert malformed == 3 and all(r['status'] == 'parse_failed' for r in rows[-3:])
    elif result['stop_reason'] == 'all_cases_attempted':
        assert len(rows) == plan['max_calls']
    else:
        assert result['stop_reason'] == 'execution_failure' and rows[-1]['status'] == 'execution_failed'
    # Decode all planned source frames, including those never sent after early stop.
    decoded = 0
    for video in plan['videos']:
        path = root / video['repository_path']
        assert sha(path) == video['sha256']
        cap = cv2.VideoCapture(str(path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 960 and cap.get(cv2.CAP_PROP_FPS) == 16
        for window in plan['windows'].values():
            for frame in window['frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                assert ok
                with Image.open(folder / 'inputs' / f"{video['branch']}_{frame}.png") as stored:
                    assert np.array_equal(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.asarray(stored))
                decoded += 1
        cap.release()
    with Image.open(folder / 'inputs/gray.png') as gray:
        assert gray.size == (832, 480) and np.all(np.asarray(gray) == 127)
    measured, tp = read(telemetry / 'summary.json'), read(telemetry / 'plan.json')
    assert read(telemetry / 'target_terminal_status.json') == status
    assert measured['target_status_sha256'] == sha(telemetry / 'target_terminal_status.json')
    assert measured['samples_sha256'] == sha(telemetry / 'samples.jsonl')
    assert tp['target_plan_sha256'] == sha(folder / 'plan.json')
    assert tp['target_manifest_sha256'] == sha(folder / 'manifest.json')
    assert tp['source_sha256'] == sha(telemetry / 'monitor_source.py')
    samples = [json.loads(s) for s in (telemetry / 'samples.jsonl').read_text().splitlines()]
    assert module(telemetry / 'monitor_source.py', 'paired_telemetry').summarize(samples, status) == {
        k: v for k, v in measured.items() if k not in ('samples_sha256', 'target_status_sha256')}
    inputs[str((telemetry / 'samples.jsonl').relative_to(folder.parent))] = sha(telemetry / 'samples.jsonl')
    groups = {}
    for row in descriptions:
        if row['control'] is None:
            groups.setdefault(row['pair_group'], []).append(row)
    return {'phase': plan['protocol'], 'actual_calls': result['actual_calls'], 'valid_replies': result['completed'],
        'stop_reason': result['stop_reason'], 'unattempted_case_ids': result['unattempted_case_ids'],
        'all_planned_source_frames_reconstructed': decoded, 'unique_images_actually_sent': len(sent_images),
        'comparison_groups': groups, 'controls': controls, 'all_attempted_responses': descriptions,
        'supervisor_seconds': status['wall_seconds'], 'worker_seconds_nested': result['worker_seconds'],
        'model_load_seconds_nested': result['model_load_seconds'], 'telemetry': measured,
        'inputs_sha256': inputs, 'audit_source_sha256': sha(Path(__file__)),
        'models_reexecuted': False, 'weights_rehashed': False, 'candidate_adopted': False,
        'actual_human_labels': 0, 'independent_script_count': 1, 'held_out': False,
        'scope': 'Exact inputs, transcript validity/failures, incomplete coverage and raw device measurements. Raw model descriptions are not accuracy labels. No identity, action or protection acceptance is established.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'folder', 'telemetry', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root.resolve(), args.folder.resolve(), args.telemetry.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('actual_calls', 'valid_replies', 'stop_reason', 'supervisor_seconds', 'unique_images_actually_sent')}))
