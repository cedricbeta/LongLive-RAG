"""Summarize measured costs and observations after both bounded candidates finish."""
import hashlib
import json
from pathlib import Path

F = Path(__file__).resolve().parent
R = F.parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(F / 'plan.json')
obs = read(F / 'developer_observations.json')
assert read(F / 'finalize_queue_status.json')['phase'] == 'completed'
runs = []
for name in ['native', 'euler']:
    folder = F / 'sampler_pair' / name
    audit = read(folder / 'audit.json')
    sampler = read(folder / 'sampler_audit.json')
    telemetry = read(R / ('native_noah_hair_' + name + '_v1_telemetry') / 'summary.json')
    assert audit['status']['status'] == 'completed'
    assert audit['measured_engineering_checks_passed'] and sampler['status'] == 'passed'
    runs.append({'mode': name, 'status': audit['status'],
        'video_sha256': audit['video_sha256'], 'latent_sha256': audit['latent_sha256'],
        'generation_wall_seconds': audit['status']['wall_seconds'],
        'private_sampler_seconds_nested_in_generation': audit['repair_stage_seconds_nested_in_generation'],
        'private_sampler_forwards': audit['additional_denoiser_calls'],
        'audit_wall_seconds': audit['audit_wall_seconds'],
        'device_energy_bounds': telemetry['device_energy_bounds'],
        'observed_foreign_compute_pids': telemetry['observed_foreign_compute_pids_inside'],
        'changed_edit_latents': len(audit['changed_edited_latents']),
        'changed_suffix_latents': len(audit['changed_suffix_latents']),
        'outside_mask_latents_exact': audit['outside_mask_latents_exact'],
        'nonempty_prefix_tested': audit['nonempty_prefix_tested'],
        'history_content_hashes_audited': audit['history_content_hashes_audited'],
        'resident_content_hashes_audited': audit['resident_content_hashes_audited'],
        'retrieval_queries': audit['retrieval_queries']})
review = read(F / 'frame_review/review.json')
costs = {
    'parent_generation_seconds_previously_incurred': plan['parent_generation_cost_previously_incurred_seconds'],
    'man_mask_preflight_seconds': read(F / 'clipseg_preflight/result.json')['wall_seconds'],
    'person_mask_preflight_seconds': read(F / 'person_preflight/result.json')['wall_seconds'],
    'full_mask_seconds': read(F / 'hair_mask/result.json')['wall_seconds'],
    'pair_preparation_seconds': read(F / 'sampler_pair/preparation.json')['wall_seconds'],
    'candidate_generation_sum_seconds_not_elapsed': sum(r['generation_wall_seconds'] for r in runs),
    'generation_interval_elapsed_seconds': max(r['status']['started_unix'] + r['generation_wall_seconds'] for r in runs)
        - min(r['status']['started_unix'] for r in runs),
    'audits_sum_seconds': sum(r['audit_wall_seconds'] for r in runs),
    'comparison_review_seconds': review['wall_seconds'],
    'limits': ['Candidates overlap on two GPUs; their sum is not elapsed time.',
        'Private sampler time is nested within generation, not additive.',
        'Queue waiting overlaps generation and review, not additional compute.',
        'Device energy includes idle power; sampled utilization is not kernel GPU time.',
        'Parent generation was reused, not free; no new baseline was generated in this study.',
        'Developer inspection, publication and miscellaneous orchestration are unmeasured; not full end-to-end cost.']}
summary = {'phase': 'completed_bounded_exposed_noah_hair_sampler_pair',
    'model': 'official LongLive-RAG 1.3B with disclosed timing/masked-replay/sampler adapters',
    'plan_sha256': sha(F / 'plan.json'), 'new_baseline_attempts': 0, 'repair_attempts': 2,
    'candidate_adopted': False, 'automatic_acceptance_performed': False,
    'human_labels': 0, 'held_out': False, 'quality_claim': False, 'submission_ready': False,
    'selection': plan['selection'], 'summary_zh': obs['summary_zh'],
    'observations_sha256': sha(F / 'developer_observations.json'),
    'runs': runs, 'costs': costs, 'comparison_video': review['comparison_video'],
    'fixed_images': len(review['images']), 'full_source_frames_per_branch': len(review['full_video_frames']),
    'previous_garment_and_Iris_budgets_closed': True, 'this_budget_closed_no_further_retries': True}
(F / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'runs': [{k:r[k] for k in ['mode','generation_wall_seconds','video_sha256']} for r in runs],
                  'costs': costs}, ensure_ascii=False))
