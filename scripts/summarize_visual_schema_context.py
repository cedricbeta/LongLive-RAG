#!/usr/bin/env python3
"""Audit a fixed visual schema-text comparison, never interpret transitions as accuracy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from utils.agent_attribute_diagnosis import assessment_schema, aggregate
from utils.agent_vlm import _image_data_uri
from probe_visual_schema_context import evidence_first_schema


def summarize(folder):
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    plan = read(folder / 'plan.json')
    variant = plan.get('comparison_arm', 'schema_in_message')
    arms = ['sampling_only', variant]
    status = read(folder / 'run/status.json')
    assert status['status'] in ('completed', 'timeout', 'execution_failed')
    rows = read(folder / 'run/results.json')
    http = read(folder / 'run/http_attempts.json')
    assert len(http) <= plan['max_requests']
    assert len({(r['packet'], r['arm']) for r in http}) == len(http), 'No transport retry is authorized'
    by_key = {(r['packet'], r['arm']): r for r in rows}
    transport = {(r['packet'], r['arm']): r for r in http}
    audits, transitions, packets, initial_aggregates = [], [], [], []
    for packet in plan['packets']:
        archive = read(folder / packet['archived_call'])
        prompt = archive['request']['messages'][0]['content'][0]['text']
        requirements = json.loads(prompt)['requirements']
        original_schema = assessment_schema(requirements)
        assert original_schema == archive['schema']
        images = [folder / p for p in packet['images']]
        for image in images:
            inputs[str(image.relative_to(folder))] = hashlib.sha256(image.read_bytes()).hexdigest()
        assert [inputs[str(p.relative_to(folder))] for p in images] == archive['ordered_image_hashes']
        record = {'packet': packet['id'], 'branch': packet['branch'], 'images': packet['images'],
                  'requirements': requirements, 'archived_parsed': archive['parsed'], 'arms': {}}
        for arm in arms:
            schema = evidence_first_schema(original_schema) if arm == 'evidence_first' else original_schema
            row = by_key.get((packet['id'], arm))
            if row is None:
                record['arms'][arm] = {'status': 'not_executed'}
                continue
            record['arms'][arm] = row
            request = transport.get((packet['id'], arm))
            actual_prompt = prompt
            if arm == 'schema_in_message':
                actual_prompt += '\nRequired response JSON schema (field names and allowed values):\n' + json.dumps(schema, ensure_ascii=False, sort_keys=True)
            payload = {'model': plan['model'], 'messages': [{'role': 'user', 'content':
                [{'type': 'text', 'text': actual_prompt}] + [{'type': 'image_url', 'image_url': {'url': _image_data_uri(p)}} for p in images]}],
                'temperature': 0, 'seed': 0, 'max_tokens': 2048,
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': archive['call_kind'], 'schema': dict(schema), 'strict': True}}}
            if request is not None:
                assert hashlib.sha256(json.dumps(payload).encode()).hexdigest() == request['request_bytes_sha256']
            if row['status'] != 'completed':
                audits.append({'packet': packet['id'], 'arm': arm, 'result': row['status'],
                               'request_bytes_verified': request is not None})
                continue
            transcript = read(folder / 'run' / row['transcript'])
            assert not transcript['parse_attempts'] and transcript['schema_enforced_by_server']
            expected_request = json.loads(json.dumps(payload))
            expected_request['messages'][0]['content'][1:] = [
                {'type': 'image_url', 'image_url': {'url': '<base64 omitted>'}} for _ in images]
            assert transcript['request'] == expected_request
            assert transcript['ordered_image_hashes'] == archive['ordered_image_hashes']
            assert transcript['parsed'] == row['parsed']
            raw_answer = json.loads(transcript['response']['choices'][0]['message']['content'])
            record['arms'][arm] = {**row, 'output_field_orders': [list(v) for v in raw_answer.values()]}
            if arm == 'sampling_only':
                assert transcript['cache_fingerprint'] == archive['cache_fingerprint']
                assert transcript['request'] == archive['request']
            else:
                assert transcript['cache_fingerprint'] != archive['cache_fingerprint']
            audits.append({'packet': packet['id'], 'arm': arm, 'result': 'completed',
                           'request_bytes_verified': True, 'images_exact': True,
                           'sampling_field_constraints_requirements_unchanged': True})
        a, b = record['arms']['sampling_only'], record['arms'][variant]
        for requirement in requirements:
            key = requirement['id']
            transitions.append({'packet': packet['id'], **requirement,
                'archive': archive['parsed'][key]['status'],
                'sampling_only': a['parsed'][key]['status'] if a['status'] == 'completed' else a['status'],
                variant: b['parsed'][key]['status'] if b['status'] == 'completed' else b['status']})
        packets.append(record)
    for branch in ['parent', 'candidate']:
        frames = [p for p in packets if p['branch'] == branch and len(p['images']) == 1]
        for arm in arms:
            if all(p['arms'][arm]['status'] == 'completed' for p in frames):
                reviews = [{'path': p['images'][0], 'sha256': inputs[p['images'][0]],
                            'assessments': p['arms'][arm]['parsed']} for p in frames]
                initial_aggregates.append({'branch': branch, 'arm': arm,
                    'rows': aggregate(frames[0]['requirements'], reviews),
                    'scope': 'Frozen single-frame aggregation only; no adaptive production run or acceptance.'})
    fresh = [r for r in rows if r['arm'] == 'sampling_only' and r['status'] == 'completed']
    return {'worker_status': status, 'comparison_arm': variant, 'actual_http_requests': len(http), 'packets': packets,
        'request_audit': audits, 'transitions': transitions,
        'status_transition_counts': dict(Counter(r['sampling_only'] + ' -> ' + r[variant] for r in transitions)),
        'fresh_baseline_parsed_exact_to_archive': sum(r['parsed_equals_archive'] for r in fresh),
        'fresh_baseline_completed_packets': len(fresh),
        'fresh_baseline_status_equal_to_archive': sum(r['archive'] == r['sampling_only'] for r in transitions),
        'initial_frame_aggregates': initial_aggregates,
        'failures': [r for r in rows if r['status'] != 'completed'],
        'recorded_http_wall_seconds': sum(r['wall_seconds'] for r in http),
        'held_out': False, 'quality_claim': False, 'adopted_arm': None, 'actual_human_labels': 0,
        'new_video_generations': 0, 'audit_new_model_calls': 0,
        'limits': ['Seven correlated packets from one exposed script; transitions are not accuracy or independent samples.',
                   'No prior verdict is replaced and no candidate selection changes.',
                   'The joint packet remains fixed even when an arm would choose a different adaptive recheck.',
                   'All actual HTTP request byte hashes are reconstructed; raw historical HTTP bytes were not saved.',
                   'HTTP, worker and server times overlap; do not add them.'],
        'inputs_sha256': inputs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.folder.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['actual_http_requests', 'status_transition_counts',
                                           'fresh_baseline_status_equal_to_archive']}))
