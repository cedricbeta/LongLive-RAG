"""Bind the single completed candidate, observed outcome and measured costs."""
import hashlib
import json
from pathlib import Path

F = Path(__file__).resolve().parent
R = F.parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(F / 'plan.json')
obs = read(F / 'developer_observations.json')
audit = read(F / 'candidate/audit.json')
review = read(F / 'frame_review/review.json')
telemetry = read(R / 'native_kai_silver_hair_v1_telemetry/summary.json')
assert read(F / 'finalize_queue_status.json')['phase'] == 'completed'
assert audit['measured_engineering_checks_passed']
assert audit['strict_prefix_latents'] == 81 and audit['prefix_latents_exact']
assert audit['status']['status'] == 'completed'
summary = {'phase': 'completed_bounded_exposed_kai_silver_hair_probe',
    'model': 'official LongLive-RAG 1.3B with disclosed timing/native masked-replay adapters',
    'plan_sha256': sha(F / 'plan.json'), 'selection': plan['selection'],
    'new_baseline_attempts': 0, 'repair_attempts': 1, 'candidate_adopted': False,
    'automatic_acceptance_performed': False, 'human_labels': 0, 'held_out': False,
    'formal_quality_claim': False, 'submission_ready': False,
    'summary_zh': obs['summary_zh'], 'observations_sha256': sha(F / 'developer_observations.json'),
    'qualitative_local_target_improvement_observed': obs['qualitative_local_target_improvement_observed'],
    'run': {'status': audit['status'], 'video_sha256': audit['video_sha256'],
        'latent_sha256': audit['latent_sha256'], 'device_energy_bounds': telemetry['device_energy_bounds'],
        'observed_foreign_compute_pids': telemetry['observed_foreign_compute_pids_inside'],
        'private_sampler_forwards': audit['additional_denoiser_calls'],
        'retrieval_queries': audit['retrieval_queries']},
    'costs': {'parent_generation_seconds_previously_incurred': plan['parent_generation_cost_previously_incurred_seconds'],
        'man_mask_preflight_seconds': read(F / 'man_preflight/result.json')['wall_seconds'],
        'coat_man_mask_preflight_seconds': read(F / 'coat_man_preflight/result.json')['wall_seconds'],
        'full_mask_seconds': read(F / 'hair_mask/result.json')['wall_seconds'],
        'candidate_preparation_seconds': read(F / 'preparation.json')['wall_seconds'],
        'generation_wall_seconds': audit['status']['wall_seconds'],
        'private_sampler_seconds_nested_in_generation': audit['repair_stage_seconds_nested_in_generation'],
        'audit_seconds': audit['audit_wall_seconds'], 'comparison_review_seconds': review['wall_seconds'],
        'limits': ['Reused baseline generation is not free; no new baseline was generated here.',
            'Private sampling is nested within generation; queue waiting overlaps work.',
            'Device energy includes idle power, not incremental application energy.',
            'Developer inspection and publication are unmeasured; not a complete end-to-end cost.']},
    'latent_checks': {'strict_prefix_latents': 81, 'prefix_latents_exact': True,
        'outside_mask_latents_exact': audit['outside_mask_latents_exact'],
        'changed_edit_latents': len(audit['changed_edited_latents']),
        'changed_suffix_latents': len(audit['changed_suffix_latents']),
        'history_and_resident_content_hashes_measured': False},
    'lossy_video_comparison': {'changed_prefix_frames_before320': [i for i in audit['video_comparison']['changed_frames'] if i < 320],
        'first_changed_frame': min(audit['video_comparison']['changed_frames'], default=None),
        'is_raw_vae_decode_equality_test': False},
    'comparison_video': review['comparison_video'], 'fixed_images': len(review['images']),
    'all_source_frames_per_branch': len(review['full_video_frames']),
    'this_and_previous_study_budgets_closed_no_further_retries': True}
(F / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'costs': summary['costs'], 'latent_checks': summary['latent_checks'],
                  'video_sha256': summary['run']['video_sha256']}, ensure_ascii=False))
