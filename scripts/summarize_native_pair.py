#!/usr/bin/env python3
"""Compare both terminal native arms under frozen common budget ceilings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def summarize(root):
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    names = ['native_iris_mixed_evidence_v1', 'native_iris_fixed_noise_v1']
    baseline_status = read(root / 'results/official_iris_60s_baseline_prepared/status.json')
    baseline_audit = read(root / 'results/official_iris_60s_baseline_prepared/audit.json')
    assert baseline_status['status'] == 'completed'
    original_video, = (root / 'artifacts/official_iris_60s_baseline_prepared/videos').glob('*.mp4')
    inputs[str(original_video.relative_to(root))] = hashlib.sha256(original_video.read_bytes()).hexdigest()
    assert inputs[str(original_video.relative_to(root))] == baseline_audit['video_sha256']
    original_seconds = baseline_status['wall_seconds']
    rows, initial_actions, initial_videos = [], [], []
    for name in names:
        folder = root / 'results' / name
        data = read(folder / 'summary.json')
        plan = read(folder / 'plan.json')
        lifecycle = read(folder / 'lifecycle.json')
        branches = read(folder / 'run/backend/branches.json')
        selected, = [b for b in branches['branches'] if b['branch_id'] == branches['current_best_branch_id']]
        assert selected['video'] == data['worker_status']['selected_video']
        selected_digest = selected['artifacts'][selected['video']]
        assert data['worker_status']['status'] != 'running'
        assert plan['max_seconds'] == 3000 and plan['max_edits_per_shot'] == 2
        assert plan['strength'] == 1 and plan['noise_seed_base'] == 4242
        assert plan['parent'] == 'official_iris_60s_baseline_prepared'
        assert plan['compiler'] == 'names_v6' and plan['compiler_schema_context']
        assert plan['lookahead_shots'] == 1 and plan['render_timeout'] == 1800
        logs = folder / 'server/sglang.log.txt'
        inputs[str(logs.relative_to(root))] = hashlib.sha256(logs.read_bytes()).hexdigest()
        codes = Counter(re.findall(r'POST /v1/chat/completions HTTP/1\.1" (\d{3})', logs.read_text()))
        edits = []
        ledger = folder / 'run/ledger.jsonl'
        inputs[str(ledger.relative_to(root))] = hashlib.sha256(ledger.read_bytes()).hexdigest()
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        for e in events:
            if e['event'] == 'action' and e.get('tool') == 'edit_shot':
                edits.append(e)
        actions = data['actual_actions']
        if actions:
            assert actions[0]['actual']['parent_video_sha256'] == baseline_audit['video_sha256']
        initial_actions.append(actions[0]['actual'] if actions else None)
        first_render = next((r for r in data['native_render_jobs'] if r.get('engineering_audit')), None)
        initial_videos.append(first_render['engineering_audit']['video_sha256'] if first_render else None)
        candidates = []
        for diagnosis in data['diagnoses']:
            if diagnosis['native_run'] == plan['parent']:
                continue
            candidates.append({'native_run': diagnosis['native_run'], 'shot': diagnosis['shot'],
                'accept': diagnosis['accept'], 'unresolved_constraints': diagnosis['unresolved_constraints']})
        render_seconds = sum(r['status'].get('wall_seconds', 0) for r in data['native_render_jobs'])
        worker_seconds = data['worker_status']['wall_seconds']
        lifecycle_seconds = lifecycle['observed_process_lifecycle_seconds']
        assert data['worker_status']['started_unix'] >= lifecycle['server_exit_mtime'] - lifecycle_seconds
        assert data['worker_status']['finished_unix'] <= lifecycle['server_exit_mtime']
        rows.append({'run': name, 'status': data['worker_status']['status'],
            'worker_wall_seconds': data['worker_status']['wall_seconds'],
            'render_supervisor_wall_seconds': render_seconds,
            'trial_lifecycle_wall_seconds': lifecycle_seconds,
            'original_plus_trial_lifecycle_seconds': original_seconds + lifecycle_seconds,
            'unused_worker_ceiling_seconds': max(0, 3000 - worker_seconds),
            'worker_ceiling_overrun_seconds': max(0, worker_seconds - 3000),
            'declared_device_slot_seconds_proxy': worker_seconds + render_seconds + lifecycle_seconds,
            'original_plus_device_slot_seconds_proxy': original_seconds + worker_seconds + render_seconds + lifecycle_seconds,
            'measured_kernel_gpu_seconds': None,
            'completed_edit_transaction_wall_seconds': sum(r['result']['wall_seconds'] for r in edits),
            'edit_transaction_scope': 'Preparation, native supervisor, engineering audit and branch registration; contained in worker time.',
            'recorded_model_calls': data['recorded_model_calls'], 'call_kinds': data['call_kinds'],
            'recorded_call_wall_seconds': data['recorded_call_wall_seconds'],
            'server_http_response_codes': dict(codes),
            'http_successes_equal_recorded_calls': codes.get('200', 0) == data['recorded_model_calls'],
            'segmentation_video_pass_seconds': sum(r['wall_seconds'] for r in data.get('segmentation_cost', [])),
            'diagnosis_wall_seconds': sum(r['wall_seconds'] for r in data['diagnoses']),
            'server_lifecycle_seconds': lifecycle['observed_process_lifecycle_seconds'],
            'completed_native_videos': data['completed_native_videos'],
            'actual_action_kinds': [r['kind'] for r in actions],
            'actual_noise_seeds': [r['actual']['noise_seed'] for r in actions],
            'retained_improvements': len(data['partial_retention_events']),
            'selected_parent_changed': data['selected_parent_changed'],
            'delivered_registered_video_sha256': selected_digest,
            'delivered_video_matches_published_original': selected_digest == baseline_audit['video_sha256'],
            'fully_verified_branch_id': data['fully_verified_branch_id'],
            'paired_current_protection_calls': sum(r['protection'] is not None for r in data['current_guard_reports']),
            'future_shots_actually_checked': data['future_shots_actually_checked'],
            'termination': data['repair_outcomes'], 'execution_failures': data['execution_failures'],
            'identity_references_promoted': data['identity_references_promoted'],
            'evidence_routing': data['evidence_routing'], 'changed_action_audits': data['changed_action_audits'],
            'candidate_diagnoses': candidates})
    return {'protocol': 'frozen_common_ceiling_development_pair_v1',
        'common_worker_ceiling_seconds': 3000, 'common_maximum_renders': 2,
        'common_original_generation': {'run': 'official_iris_60s_baseline_prepared',
            'video_sha256': baseline_audit['video_sha256'], 'historical_supervisor_seconds': original_seconds,
            'scope': 'One earlier real baseline is shared by both repair arms. Its recorded cost is shown separately and is not charged as a newly repeated render.',
            'new_baseline_generations_for_this_comparison': 0},
        'cost_definitions': {
            'trial_lifecycle_wall_seconds': 'Exclusive judge launcher start through exit marker, enclosing the entire worker interval. Includes cold startup, worker, idle and shutdown.',
            'declared_device_slot_seconds_proxy': 'One segmentation slot for worker wall time + one judge slot for server lifecycle + one render slot for each native supervisor. An occupancy accounting proxy, not sampled utilization or kernel timing.',
            'original_plus_values': 'Sum of separately measured original generation and additional trial intervals; a reconstructed full-pipeline cost, not one newly timed contiguous execution.',
            'worker_ceiling': 'No new operation after the ceiling; already in-flight work may finish. Any observed overrun is reported.'},
        'initial_actual_actions_equal': initial_actions[0] is not None and initial_actions[0] == initial_actions[1],
        'initial_video_bytes_equal': initial_videos[0] is not None and initial_videos[0] == initial_videos[1],
        'final_videos_identical_to_original_and_each_other': all(r['delivered_video_matches_published_original'] for r in rows),
        'rows': rows, 'held_out': False, 'quality_claim': False, 'actual_human_labels': 0,
        'independent_final_evaluation': False, 'new_model_calls_for_summary': 0,
        'limits': ['One previously exposed script and one paired seed schedule, no independent samples or confidence interval.',
            'Shared ceilings do not imply equal actual compute; unused budget is not fabricated.',
            'Worker, diagnosis, model-call, segmentation and server durations overlap; do not add them.',
            'Render supervisor duration includes startup, generation, replay and file output; it is not kernel GPU time.',
            'A development checker may contradict itself; its labels do not establish independent visual accuracy.',
            'A requested lookahead does not prove a future check was reached.'],
        'inputs_sha256': inputs}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'initial_actions_equal': result['initial_actual_actions_equal'],
                      'initial_videos_equal': result['initial_video_bytes_equal'],
                      'arms': [r['status'] for r in result['rows']]}))
