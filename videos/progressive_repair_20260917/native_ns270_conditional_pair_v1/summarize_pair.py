#!/usr/bin/env python3
"""Close the frozen pair from actual terminal records, without new model calls."""
import collections, hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, queue = read(root / 'plan.json'), read(root / 'queue_status.json')
assert queue['status'] == 'completed' and len(queue['arms']) == 2
parent = Path(plan['parent'])
parent_audit = read(parent / 'audit.json')
rows = []
for arm in plan['arms']:
    folder = root / arm
    status, worker = read(folder / 'status.json'), read(folder / 'run/status.json')
    assert status['status'] == worker['status'] == 'completed'
    owned = read(folder / 'owned_processes.json')
    assert owned['live_owned_at_end'] == []
    known = {row['pid'] for row in owned['captured']}
    calls = [read(p) for p in sorted((folder / 'run/vlm_cache').glob('*/*.json'))]
    kinds = dict(collections.Counter(row['call_kind'] for row in calls))
    selected, = read(folder / 'run/selected.json')
    assert selected['shot'] == 1 and selected['accepted'] is False and selected['quarantined'] is True
    assert selected['delivered']['attempt'] == 0
    assert sha(Path(worker['selected_video'])) == parent_audit['video_sha256']
    assert not list((folder / 'run/backend').glob('candidate_*'))
    ledger = [json.loads(line) for line in (folder / 'run/ledger.jsonl').read_text().splitlines()]
    finished, = [row for row in ledger if row['event'] == 'repair_finished']
    assert finished['render_attempts'] == finished['retained_improvements'] == 0
    assert finished['termination'] == 'no_supported_confirmed_target'
    diagnosis, = read(folder / 'run/diagnosis_calls.json')
    attributes = diagnosis['verdict']['metrics']['entities']['Iris']['attribute_assessments']
    assert all(row['status'] == 'unknown' for row in attributes)
    segmentation = read(folder / 'run/segmentation_cost.json')
    assert not segmentation
    masks = [{'path': str(p.relative_to(root)), 'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in (folder / 'run/verifier').rglob('repaint_masks.pt')]
    telemetry = {}
    for gpu in [0, 2]:
        m = read(folder / f'gpu{gpu}_telemetry/summary.json')
        unexplained = set(m['observed_foreign_compute_pids_inside']) - known
        assert not unexplained, unexplained
        telemetry[str(gpu)] = {'device_energy_bounds': m['device_energy_bounds'],
            'peak_observed_memory_used_bytes': m['peak_observed_memory_used_bytes'],
            'integrated_sampled_utilization_seconds': m['integrated_sampled_utilization_seconds'],
            'nonworker_compute_pids_are_owned_server_processes': True, 'unexplained_observed_compute_pids': []}
    rows.append({'arm': arm, 'worker_seconds': worker['wall_seconds'], 'lifecycle_seconds': status['wall_seconds'],
        'startup_seconds': status['server_startup_seconds'], 'model_calls_by_kind': kinds,
        'total_model_calls': len(calls), 'text_calls': sum(not c['ordered_image_paths'] for c in calls),
        'visual_calls': sum(bool(c['ordered_image_paths']) for c in calls),
        'model_call_seconds_sum_nested_in_worker': sum(c['wall_seconds'] for c in calls),
        'diagnosis_seconds_nested_in_worker': diagnosis['wall_seconds'], 'selection': selected,
        'selected_original_sha256': parent_audit['video_sha256'], 'render_attempts': 0,
        'retained_improvements': 0, 'termination': finished['termination'], 'attributes': attributes,
        'clipseg_part_mask_calls': 0, 'sam2_person_mask_files': masks,
        'paired_protection_or_future_checks_reached': False, 'device_telemetry': telemetry})
summary = {'status': 'closed_both_arms_unknown_after_contradictory_recheck', 'case': 'ns270',
    'plan_sha256': sha(root / 'plan.json'), 'current_window_seconds': [10, 20], 'arms': rows,
    'total_model_calls': sum(r['total_model_calls'] for r in rows),
    'total_text_calls': sum(r['text_calls'] for r in rows), 'total_visual_calls': sum(r['visual_calls'] for r in rows),
    'total_candidate_generations': 0, 'automatic_retained_improvements': 0,
    'unused_candidate_slots_closed': 4, 'remaining_case_attempts': 0,
    'queue_seconds': queue['wall_seconds'], 'worker_seconds_sum': sum(r['worker_seconds'] for r in rows),
    'lifecycle_seconds_sum': sum(r['lifecycle_seconds'] for r in rows),
    'reserved_two_device_seconds_proxy': 2 * sum(r['lifecycle_seconds'] for r in rows),
    'selection_frozen_before_baseline_viewing': True, 'first_window_remains_unassessed': True,
    'quality_acceptance': False, 'independent_evaluation': False, 'human_labels': 0, 'held_out': False,
    'full_action_policy_comparison_exercised': False, 'submission_ready': False,
    'conclusion': 'Each arm sees mixed first/middle/last hair evidence, then a contradictory joint recheck. Both keep unknown and return the original without rendering or retaining a repair. Later red curls already occur in the native original and cannot be credited to repair.',
    'next_action': 'Close this case without another checker prompt, target/window change, fixed-target generation or retry. Continue remaining frozen cases.',
    'limits': ['This is a completed two-arm diagnostic comparison, with zero opportunities to compare generated feedback and fixed-noise actions.',
        'SAM2 person masks were written, but no CLIPSeg attribute-mask call or repair action occurred.',
        'Sparse source frames show an onset mismatch and later matching appearance; no exact transition duration or motion/identity acceptance is established.',
        'Summed model time is nested in worker time; startup and cleanup are inside lifecycle time. These are not additive.',
        'The two-device reservation proxy includes idle GPU 0; sampled device energy includes idle power and is not kernel GPU time.']}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['status', 'total_model_calls', 'worker_seconds_sum', 'lifecycle_seconds_sum', 'queue_seconds']}))
