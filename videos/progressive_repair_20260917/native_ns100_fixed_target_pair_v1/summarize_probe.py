#!/usr/bin/env python3
"""Close the two-video budget using actual terminal records and developer views."""
import hashlib, json
from pathlib import Path
import yaml

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, queue = read(root / 'plan.json'), read(root / 'queue_status.json')
assert queue['status'] == 'completed' and len(queue['arms']) == 2
parent = Path(plan['parent'])
base_config = yaml.safe_load((parent / 'config.yaml').read_text())
base_audit = read(parent / 'audit.json')
rows = []
for arm in ['P0_prompt', 'R0_local']:
    folder = root / arm / 'baseline'
    status, manifest = read(folder / 'status.json'), read(folder / 'manifest.json')
    audit = read(folder / 'audit.json')
    assert status['status'] == 'completed'
    assert not Path('/proc', str(status['pid'])).exists()
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    assert config['timed_segments'][1:] == base_config['timed_segments'][1:]
    assert config['seed'] == base_config['seed'] == 0
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    video, = (folder / 'videos').glob('*.mp4')
    assert sha(video) == audit['video_sha256']
    for i, row in enumerate(read(folder / 'diagnostics/switches_0000.json'), 1):
        assert row['prompt_sha256'] == [hashlib.sha256(config['timed_segments'][i]['prompts'][0].encode()).hexdigest()]
    telemetry = read(root / (arm + '_telemetry') / 'summary.json')
    assert read(root / (arm + '_telemetry') / 'target_terminal_status.json') == status
    row = {'arm': arm, 'generation_seconds': status['wall_seconds'], 'audit_seconds': audit['audit_wall_seconds'],
        'video_sha256': audit['video_sha256'], 'latent_sha256': audit['latent_sha256'],
        'device_energy_bounds': telemetry['device_energy_bounds'],
        'peak_observed_memory_used_bytes': telemetry['peak_observed_memory_used_bytes'],
        'observed_foreign_compute_pids': telemetry['observed_foreign_compute_pids_inside'],
        'five_actual_later_prompt_hashes_match': True, 'automatic_acceptance': False, 'candidate_retained': False}
    if arm == 'P0_prompt':
        assert audit['engineering_checks_passed'] and audit['frames_decoded'] == 960
        assert 'repair_request' not in config
        assert config['timed_segments'][0]['prompts'] == [plan['target_expression']]
        row.update(additional_private_denoiser_calls=0, mask_calls=0)
    else:
        assert audit['measured_engineering_checks_passed'] and audit['outside_mask_latents_exact']
        assert config['repair_request']['prompts'] == [plan['target_expression']]
        assert config['timed_segments'] == base_config['timed_segments']
        assert audit['edited_range'] == [0, 41]
        assert audit['nonempty_prefix_tested'] is False
        row.update(additional_private_denoiser_calls=audit['additional_denoiser_calls'],
            repair_seconds_nested_in_generation=audit['repair_stage_seconds_nested_in_generation'],
            outside_mask_latents_exact=True, edited_latents_changed=len(audit['changed_edited_latents']),
            suffix_latents_changed=len(audit['changed_suffix_latents']), nonempty_prefix_tested=False)
    rows.append(row)
views = read(root / 'developer_observations.json')
assert views['all_54_predeclared_views_seen']
summary = {'status': 'closed_two_fixed_target_capacity_generations', 'case': 'ns100',
    'plan_sha256': sha(root / 'plan.json'), 'arms': rows, 'new_generation_attempts': 2,
    'remaining_generation_budget': 0, 'remaining_mask_budget': 0, 'automatic_acceptance': False,
    'automatic_retained_improvements': 0, 'human_labels': 0, 'independent_evaluation': False,
    'held_out': False, 'quality_acceptance': False, 'submission_ready': False,
    'generation_seconds_sum': sum(row['generation_seconds'] for row in rows), 'queue_seconds': queue['wall_seconds'],
    'mask_worker_seconds': read(root / 'garment_mask/status.json')['wall_seconds'],
    'mask_supervised_seconds': read(root / 'mask_supervision.json')['wall_seconds'],
    'preparation_seconds': read(root / 'preparation.json')['wall_seconds'],
    'comparison_build_seconds': read(root / 'comparison/manifest.json')['wall_seconds'],
    'prompt_control_audit_recovery': read(root / 'prompt_audit_recovery.json'),
    'initial_prompt_audit_failure_retained': True,
    'baseline_generation_seconds_previously_incurred': read(parent / 'status.json')['wall_seconds'],
    'preceding_automatic_pair_seconds_separate': read(Path(plan['automatic_pair_folder']) / 'queue_status.json')['wall_seconds'],
    'developer_observations_sha256': sha(root / 'developer_observations.json'),
    'conclusion': views['conclusion'], 'limits': [
        'Developer-specified literal target after the automatic checker error; not autonomous detection/retention.',
        'Two new videos with matched target text/time, not a matched-compute formal comparison or isolated mask ablation.',
        'Full subsequent videos are retained; sparse stills do not certify continuous motion, identity or actions.',
        'Mask includes notebook/hand spill and a small first-frame hair patch; it was not manually adjusted.',
        'Mask, preparation, render and display costs are separate. Worker/model/device intervals overlap; sampled energy includes idle power.',
        'Outside-mask latent equality is not pixel or semantic preservation. No nonempty prefix exists when editing starts at zero.',
        'History/resident tensor byte hashes and inspection/publication time remain unmeasured.']}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['status', 'generation_seconds_sum', 'queue_seconds', 'conclusion']}))
