#!/usr/bin/env python3
"""Summarize a terminal exported native window without generating quality labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(root, name):
    folder = root / 'results' / name
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    status = read(folder / 'run/status.json')
    assert status['status'] in ('completed', 'execution_failed', 'timeout'), 'Do not summarize a live run as final'
    plan = read(folder / 'plan.json')
    manifest = read(folder / 'run/manifest.json')
    calls = [read(p) for p in sorted((folder / 'run/vlm_cache').rglob('*.json'))]
    assert all(not r['parse_attempts'] for r in calls), 'Retry costs need explicit treatment'
    diagnosis = read(folder / 'run/diagnosis_calls.json')
    branches = read(folder / 'run/backend/branches.json')
    ledger_path = folder / 'run/ledger.jsonl'
    inputs[str(ledger_path.relative_to(root))] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    renders = []
    for candidate in sorted((folder / 'run/backend').glob('candidate_*')):
        if not candidate.is_dir() or not (candidate / 'status.json').exists():
            continue
        row = {'folder': candidate.name, 'status': read(candidate / 'status.json')}
        if (candidate / 'audit.json').exists():
            audit = read(candidate / 'audit.json')
            row['engineering_audit'] = audit
        renders.append(row)
    diagnoses = []
    for record in diagnosis:
        metrics = record['verdict']['metrics']
        constraints = metrics['constraint_assessments']
        diagnoses.append({'shot': record['shot'], 'native_run': Path(record['native_run']).name,
            'wall_seconds': record['wall_seconds'], 'accept': record['verdict']['accept'],
            'reason': record['verdict']['reason'],
            'constraint_counts_including_aggregate': dict(Counter(r['status'] for r in constraints)),
            'unresolved_constraints': [r for r in constraints if r['status'] != 'satisfied'],
            'repair_dependencies': metrics.get('repair_dependencies'),
            'native_regions_offered': len(metrics.get('native_mask_projection', [])),
            'repair_options': len(metrics.get('repair_options', []))})
    futures = [{'event': e['event'], 'shot': e.get('shot_index'), 'attempt': e.get('attempt'),
                'report': e['lookahead']} for e in events if isinstance(e.get('lookahead'), dict)]
    checked = sorted({r['shot_index'] for f in futures for r in f['report'].get('checked', [])})
    result = {'run': name, 'worker_status': status, 'held_out': False, 'quality_claim': False,
        'current_shots_requested': manifest['current_shots_requested'],
        'current_shots_diagnosed': sorted({r['shot'] for r in diagnoses} & set(manifest['current_shots_requested'])),
        'future_shots_actually_checked': checked, 'future_reports': futures,
        'unassessed_prefix': list(range(min(manifest['current_shots_requested']))),
        'recorded_model_calls': len(calls), 'call_kinds': dict(Counter(r['call_kind'] for r in calls)),
        'recorded_call_wall_seconds': sum(r['wall_seconds'] for r in calls),
        'diagnoses': diagnoses, 'native_render_jobs': renders,
        'completed_native_videos': sum(r['status']['status'] == 'completed' for r in renders),
        'original_branch_id': branches['original_branch_id'],
        'current_best_branch_id': branches['current_best_branch_id'],
        'fully_verified_branch_id': branches['fully_verified_branch_id'],
        'selected_parent_changed': branches['current_best_branch_id'] != branches['original_branch_id'],
        'partial_retention_events': [e for e in events if e['event'] == 'partial_improvement_retained'],
        'repair_outcomes': [e for e in events if e['event'] == 'repair_finished'],
        'execution_failures': [e for e in events if e['event'] == 'progressive_execution_failed'],
        'current_guard_reports': [e for e in events if e['event'] == 'progressive_current_guard'],
        'offline_window_declaration': [e for e in events if e['event'] == 'offline_window_start'],
        'identity_references_promoted': branches['identity_references_promoted'],
        'actual_human_labels': 0, 'audit_new_model_calls': 0,
        'limits': ['Development checker outputs are not independent quality labels.',
                   'Aggregate and atomic constraint rows overlap; do not count them as independent defects.',
                   'Requested lookahead is not evidence that future checks ran.',
                   'Compiler, initial diagnosis, renderer and guard failures remain in separate records.',
                   'Call times are contained in worker time; do not add them.'],
        'frozen_limits': plan['limits']}
    for name in ['memory_audit.json', 'segmentation_cost.json', 'selected.json']:
        path = folder / 'run' / name
        if path.exists():
            result[name.removesuffix('.json')] = read(path)
    result['inputs_sha256'] = inputs
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root.resolve(), args.run)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['run', 'recorded_model_calls', 'completed_native_videos',
                                           'current_best_branch_id', 'future_shots_actually_checked']}))
