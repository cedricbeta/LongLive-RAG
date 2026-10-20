#!/usr/bin/env python3
"""Rebuild the sixth-batch numerical summary from terminal runs and fixed review."""
import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(root / 'plan.json')
assert plan['batch_indices'] == [180, 30]
assert read(root / 'queue_status.json')['status'] == 'completed'
assert sha(root.parent / 'native_development_pilot12_v1/plan.json') == plan['prior_cohort_plan_sha256']
assert sha(root.parent / 'native_development_pilot12_batch5_v1/summary.json') == plan['previous_batch_summary_sha256']
rows = []
for name in ['ns180', 'ns030']:
    folder = root / name / 'baseline'
    case = next(c for c in plan['cases'] if c['id'] == name)
    audit, status = read(folder / 'audit.json'), read(folder / 'status.json')
    telemetry = read(root / (name + '_telemetry') / 'summary.json')
    switches = read(folder / 'diagnostics/switches_0000.json')
    assert len(switches) == 5
    assert all(row['prompt_sha256'] == [hashlib.sha256(case['prompts'][i].encode()).hexdigest()] for i, row in enumerate(switches, 1))
    assert status['status'] == 'completed' and audit['engineering_checks_passed']
    assert sha(folder / audit['video_path']) == audit['video_sha256']
    assert read(root / (name + '_telemetry') / 'target_terminal_status.json') == status
    rows.append({'id': name, 'generation_seconds': audit['generation_wall_seconds'],
        'audit_seconds': audit['audit_wall_seconds'], 'video_sha256': audit['video_sha256'],
        'latent_sha256': audit['latent_sha256'], 'all_five_actual_prompt_switch_hashes_match': True,
        'full_frames_decoded': audit['frames_decoded'], 'fixed_frames_viewed': plan['review_frames'],
        'device_energy_bounds': telemetry['device_energy_bounds'],
        'observed_foreign_compute_pids': telemetry['observed_foreign_compute_pids_inside']})
summary = {
    'phase': 'sixth_two_originals_for_preselected_twelve_script_pilot', 'selection_plan_sha256': sha(root / 'plan.json'),
    'prior_cohort_plan_sha256': plan['prior_cohort_plan_sha256'], 'previous_batch_summary_sha256': plan['previous_batch_summary_sha256'],
    'planned_scripts': 12, 'selected_indices': [c['index'] for c in plan['cases']],
    'this_batch_attempted': 2, 'completed': 2, 'total_completed_originals': 12,
    'completed_original_indices': plan['previously_completed_original_indices'] + plan['batch_indices'],
    'remaining_selected_originals_unlaunched': 0, 'new_repairs': 0, 'new_control_generations': 0,
    'remaining_this_batch_generation_budget': 0, 'full_comparative_pilot_completed': False,
    'full_comparative_protocol_frozen': False, 'runs': rows,
    'generation_seconds_sum': sum(row['generation_seconds'] for row in rows),
    'preparation_seconds': read(root / 'preparation.json')['wall_seconds'],
    'queue_seconds': read(root / 'queue_status.json')['wall_seconds'],
    'developer_observations_sha256': sha(root / 'developer_observations.json'),
    'held_out': False, 'human_labels': 0, 'automatic_retained_improvements': 0,
    'quality_acceptance': False, 'submission_ready': False,
    'limits': ['Originals only; the separately frozen ns180 automatic pair has a separate budget and records.',
        'Developer fixed-frame observations are not independent human labels or full action/identity certification.',
        'Summed concurrent durations differ from queue elapsed time. Device energy includes idle power and sampled utilization is not kernel GPU time.',
        'Inspection/publication time and optional history/resident tensor byte hashes are unmeasured.']}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ['total_completed_originals', 'generation_seconds_sum', 'queue_seconds']}))
