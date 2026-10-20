#!/usr/bin/env python3
"""Report both compiler failures without assigning unexecuted policy outcomes."""
import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(root / 'plan.json')
parent = Path(plan['parent'])
audit = read(parent / 'audit.json')
video = parent / audit['video_path']
assert sha(video) == audit['video_sha256']
assert read(root / 'queue_status.json')['status'] == 'completed'
rows = []; replies = []
for arm in plan['arms']:
    folder = root / arm
    status, worker = read(folder / 'status.json'), read(folder / 'run/status.json')
    assert status['status'] == 'failed' and worker['status'] == 'execution_failed'
    assert worker['error'] == 'ValueError: A noncharacter cannot have depicted character presence'
    assert sha(Path(worker['selected_video'])) == audit['video_sha256']
    assert not (folder / 'run/backend').exists() and not (folder / 'run/verifier').exists()
    assert not (folder / 'run/compiled_script.json').exists()
    assert not (folder / 'run/selected.json').exists()
    cache_files = list((folder / 'run/vlm_cache').rglob('*.json'))
    assert len(cache_files) == 1
    call = read(cache_files[0]); parsed = call['parsed']
    content = call['response']['choices'][0]['message']['content']; replies.append(content)
    invalid = {key: [shot for shot, state in value['presence'].items() if state != 'not_required']
        for key, value in parsed.items() if value['kind'] == 'noncharacter' and any(state != 'not_required' for state in value['presence'].values())}
    assert invalid == {'n004': ['1', '4', '5'], 'n006': ['3']}
    owned = read(folder / 'owned_processes.json')
    assert not owned['live_owned_at_end']
    assert not Path('/proc', str(worker['pid'])).exists()
    device = {}
    for gpu in [0, 2]:
        target = read(folder / f'gpu{gpu}_telemetry/target_terminal_status.json')
        report = read(folder / f'gpu{gpu}_telemetry/summary.json')
        assert target['wall_seconds'] == status['wall_seconds'] and target['status'] == 'failed'
        raw_other = report['observed_foreign_compute_pids_inside']
        known = {row['pid']: row for row in owned['captured']}
        assert all(pid in known for pid in raw_other)
        device[str(gpu)] = {'energy_bounds': report['device_energy_bounds'],
            'peak_observed_memory_used_bytes': report['peak_observed_memory_used_bytes'],
            'raw_other_pids_than_worker': raw_other,
            'other_pid_ownership_records': [known[pid] for pid in raw_other]}
    rows.append({'arm': arm, 'outcome': 'compiler_failure_before_visual_diagnosis',
        'worker_seconds': worker['wall_seconds'], 'arm_lifecycle_seconds': status['wall_seconds'],
        'server_startup_seconds': status['server_startup_seconds'], 'text_calls': 1,
        'call_seconds': call['wall_seconds'], 'usage': call['response']['usage'],
        'response_sha256': hashlib.sha256(content.encode()).hexdigest(), 'cache_record_sha256': sha(cache_files[0]),
        'invalid_noncharacter_presence': invalid, 'visual_calls': 0, 'segmenter_model_loads': 0,
        'candidate_generations': 0, 'policy_difference_exercised': False, 'future_checks': 0,
        'fallback_reference_video_sha256': audit['video_sha256'], 'device_telemetry': device,
        'all_owned_processes_terminal': True})
assert replies[0] == replies[1]
summary = {
    'status': 'closed_both_arms_failed_in_compiler', 'case': 'ns200', 'arms': rows,
    'plan_sha256': sha(root / 'plan.json'), 'first_plan_frozen_before_baseline_viewing': True,
    'same_raw_reply_across_two_fresh_servers': True, 'total_actual_text_calls': 2,
    'total_actual_visual_calls': 0, 'total_candidates': 0, 'unused_candidate_slots': 4,
    'remaining_case_attempts': 0, 'new_originals_in_this_pair': 0,
    'automatic_retained_improvements': 0, 'human_labels': 0, 'independent_evaluation': False,
    'quality_acceptance': False, 'full_comparative_pilot_completed': False, 'submission_ready': False,
    'worker_seconds_sum': sum(row['worker_seconds'] for row in rows),
    'arm_lifecycle_seconds_sum': sum(row['arm_lifecycle_seconds'] for row in rows),
    'queue_seconds': read(root / 'queue_status.json')['wall_seconds'],
    'declared_two_gpu_slot_seconds_sum': 2 * sum(row['arm_lifecycle_seconds'] for row in rows),
    'slot_accounting_limit': 'Reserved device-duration proxy including idle GPU 0, not kernel GPU time or exact incremental computation. Energy/proxy telemetry is separate.',
    'prior_setup_failure': '../native_ns200_progressive_pair_v1/startup_failure.json',
    'prior_setup_model_calls': 0, 'asset_preflight_seconds': read(root / 'assets_preflight.json')['wall_seconds'],
    'conclusion': 'The unchanged compiler rejects contradictory pronoun presence before the two action policies diverge. Neither safety nor repair superiority was tested. Both fallback references point to the original.',
    'implementation_limit': 'The explicit shared-GPU ownership route reaches the server and text compiler. SAM2/CLIPSeg were not loaded, so simultaneous judge/segmenter residency is still untested.',
    'next_action': 'Use a separate development protocol to make character kind and presence structurally compatible, preserving this failure. Do not rewrite replies, waive the invariant or rerun this closed pair.'}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ['status', 'total_actual_text_calls', 'total_candidates', 'worker_seconds_sum', 'arm_lifecycle_seconds_sum']}))
