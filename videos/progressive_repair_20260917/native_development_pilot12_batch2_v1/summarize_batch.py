#!/usr/bin/env python3
"""Bind two completed originals and disclose the fixed-frame developer review."""
import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, v: p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')
plan = read(root / 'plan.json')
prior = root.parent / 'native_development_pilot12_v1'
assert sha(prior / 'plan.json') == plan['prior_cohort_plan_sha256']
assert read(prior / 'summary.json')['completed'] == 2
assert plan['batch_indices'] == [10, 70]
observations = {
    'reviewer': 'assistant/developer', 'human_raters': 0, 'human_labels': 0,
    'evidence_scope': 'All 36 predeclared stills viewed in two montages. Continuous motion and name-to-person assignments are not certified.',
    'independent_evaluation': False, 'automatic_acceptance': False,
    'runs': {
        'ns010': {
            'authored_requirements': 'First prompt specifies curly red hair and moss green coat for Iris. Second introduces an older man with silver hair and brown trench coat. Later prompts specify book interaction, his departure and book placement.',
            'observations': [
                'The initial solo woman has dark brown/wavy-looking hair in the fixed views, not clearly the authored curly red hair. Her outer garment is green/teal; exact moss shade and garment classification are not calibrated.',
                'An older gray/silver-haired man in a brown outer garment is visible at 15 through 30 seconds and absent at 35 seconds and later fixed views. Names and exact character identity are not independently established.',
                'The woman holds a book in fixed views around 35–40 seconds and handles it in later views. Exact touching, door closure, departure and final placement are not certified from these stills.',
                'Window light and color patches change visibly around 30 seconds and toward the end.'
            ],
            'conclusion': 'An authored hair mismatch remains; some cast and prop changes are visible. Original only, no repair or overall quality acceptance.'
        },
        'ns070': {
            'authored_requirements': 'Named participation changes over six prompts: three initially, a departure/arrival, four sharing tasks, three named again at 40–50 seconds, then five sitting together. No ages, genders, hair or clothing appearance requirements occur in the actual script.',
            'observations': [
                'Three child-looking people sit and work in the fixed views through 20 seconds. This is not an age violation: the authored script supplies no ages and the cast registry is not a hidden requirement.',
                'Four people are visible from the 25-second fixed view onward; the additional white-shirt figure is partly occluded at 25 seconds and clearer later.',
                'Four are still visible at 45 and 49.9375 seconds although that paragraph names three; the wording does not explicitly forbid extras. Role and identity assignments remain unresolved.',
                'Only four are visible at 55 and 59.9375 seconds whereas the final paragraph asks all five named people to sit together. This is a fixed-view participation concern, not a certified per-frame count metric.',
                'No clear standing departure/arrival is established by the sampled 10–20-second views. Sparse samples cannot prove those actions never occurred. Notebook, projector and laptop role assignments are not certified.'
            ],
            'conclusion': 'Keep this script in the cohort despite zero explicit appearance candidates. Temporal participation and object roles need evaluation; no repair or overall quality acceptance.'
        }
    }
}
write(root / 'developer_observations.json', observations)
runs = []
owned = []
launch = read(root / 'launch.json')
for name in ('ns010', 'ns070'):
    folder = root / name / 'baseline'
    case = next(c for c in plan['cases'] if c['id'] == name)
    switches = read(folder / 'diagnostics/switches_0000.json')
    assert len(switches) == 5
    for i, row in enumerate(switches, 1):
        assert row['prompt_sha256'] == [hashlib.sha256(case['prompts'][i].encode()).hexdigest()]
    audit = read(folder / 'audit.json')
    status = read(folder / 'status.json')
    telemetry = read(root / (name + '_telemetry') / 'summary.json')
    assert audit['engineering_checks_passed'] and status['status'] == 'completed'
    assert read(root / (name + '_telemetry') / 'target_terminal_status.json') == status
    assert audit['frames_decoded'] == 960
    assert sha(folder / audit['video_path']) == audit['video_sha256']
    pids = {**next(row for row in launch if row['case'] == name), 'worker_pid': status['pid']}
    for key in ('supervisor_pid', 'observer_pid', 'worker_pid'):
        assert not Path('/proc', str(pids[key])).exists(), pids
    owned.append(pids)
    runs.append({
        'id': name, 'generation_seconds': audit['generation_wall_seconds'],
        'audit_seconds': audit['audit_wall_seconds'], 'video_sha256': audit['video_sha256'],
        'latent_sha256': audit['latent_sha256'], 'all_five_actual_prompt_switch_hashes_match': True,
        'full_frames_decoded': 960, 'fixed_frames_viewed': plan['review_frames'],
        'device_energy_bounds': telemetry['device_energy_bounds'],
        'observed_foreign_compute_pids': telemetry['observed_foreign_compute_pids_inside']})
summary = {
    'phase': 'second_two_originals_for_preselected_twelve_script_pilot',
    'selection_plan_sha256': sha(root / 'plan.json'), 'prior_cohort_plan_sha256': sha(prior / 'plan.json'),
    'planned_scripts': 12, 'selected_indices': [c['index'] for c in plan['cases']],
    'second_batch_attempted': 2, 'completed': 2, 'total_completed_originals': 4,
    'completed_original_indices': [40, 130, 10, 70], 'remaining_selected_originals_unlaunched': 8,
    'new_repairs': 0, 'new_control_generations': 0, 'remaining_second_batch_generation_budget': 0,
    'full_comparative_pilot_completed': False, 'full_comparative_protocol_frozen': False,
    'runs': runs, 'generation_seconds_sum': sum(x['generation_seconds'] for x in runs),
    'preparation_seconds': read(root / 'preparation.json')['wall_seconds'],
    'queue_seconds': read(root / 'queue_status.json')['wall_seconds'],
    'all_owned_jobs_terminal_at_summary': owned,
    'developer_observations_sha256': sha(root / 'developer_observations.json'),
    'held_out': False, 'human_labels': 0, 'automatic_retained_improvements': 0,
    'quality_acceptance': False, 'submission_ready': False,
    'limits': [
        'Two official LongLive-RAG originals, not repair efficacy or a completed strategy pilot.',
        'Same frozen cohort and authored prompts; sampling metadata never supplies hidden appearance.',
        'Fixed stills do not certify continuous actions, identity or all-frame people counts.',
        'Summed concurrent generation times are not elapsed queue time; device energy includes idle power.',
        'Developer review and publication time are unmeasured. Optional history/resident byte hashes are not audited.'
    ]
}
write(root / 'summary.json', summary)
print(json.dumps({k: summary[k] for k in ['completed', 'total_completed_originals', 'generation_seconds_sum', 'queue_seconds', 'new_repairs']}))
