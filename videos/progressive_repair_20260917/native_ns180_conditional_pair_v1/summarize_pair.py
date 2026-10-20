#!/usr/bin/env python3
"""Summarize the closed bounded experiment from actual records; no model calls."""
import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, queue = read(root / 'plan.json'), read(root / 'queue_status.json')
assert queue['status'] == 'completed' and len(queue['arms']) == 2
parent = Path(plan['parent'])
parent_audit = read(parent / 'audit.json')
feedback = root / 'feedback'
status, worker = read(feedback / 'status.json'), read(feedback / 'run/status.json')
assert status['status'] == worker['status'] == 'completed'
selected, = read(feedback / 'run/selected.json')
assert not selected['accepted'] and selected['quarantined']
assert selected['delivered']['attempt'] == 0
assert sha(Path(worker['selected_video'])) == parent_audit['video_sha256']
ledger = [json.loads(line) for line in (feedback / 'run/ledger.jsonl').read_text().splitlines()]
finished, = [row for row in ledger if row['event'] == 'repair_finished']
assert finished['render_attempts'] == 2 and finished['retained_improvements'] == 0
assert finished['termination'] == 'execution_failed'
guard, = [row for row in ledger if row['event'] == 'progressive_current_guard']
assert not guard['decision']['allow']
calls = [read(p) for p in sorted((feedback / 'run/vlm_cache').glob('*/*.json'))]
diagnoses = read(feedback / 'run/diagnosis_calls.json')
assert len(calls) == 22 and len(diagnoses) == 2
segmentation = read(feedback / 'run/segmentation_cost.json')
owned = read(feedback / 'owned_processes.json')
assert owned['live_owned_at_end'] == []
known = {row['pid'] for row in owned['captured']}
telemetry = {}
for gpu in [0, 2]:
    m = read(feedback / f'gpu{gpu}_telemetry/summary.json')
    foreign = sorted(set(m['observed_foreign_compute_pids_inside']) - known)
    telemetry[str(gpu)] = {
        'device_energy_bounds': m['device_energy_bounds'],
        'integrated_sampled_utilization_seconds': m['integrated_sampled_utilization_seconds'],
        'peak_observed_memory_used_bytes': m['peak_observed_memory_used_bytes'],
        'unowned_compute_pids_observed': foreign,
        'exclusive_task_energy': not bool(foreign)}
candidates = []
actions = [row for row in ledger if row['event'] == 'bounded_action_reserved']
assert len(actions) == 2
for ordinal, folder in enumerate(sorted(p for p in (feedback / 'run/backend').glob('candidate_*') if p.is_dir())):
    cs, manifest = read(folder / 'status.json'), read(folder / 'manifest.json')
    row = {'candidate': folder.name, 'status': cs['status'], 'render_supervised_seconds': cs['wall_seconds'],
           'preparation_seconds_nested_in_worker': manifest['preparation_wall_seconds'],
           'program_action': actions[ordinal], 'manifest_action': manifest['action'],
           'manifest_action_fingerprint': manifest['action_fingerprint']}
    if ordinal == 0:
        assert cs['status'] == 'completed' and cs['returncode'] == 0
        audit = read(folder / 'audit.json')
        assert audit['all_engineering_checks_passed'] and audit['outside_mask_latents_exact']
        switches = read(folder / 'diagnostics/switches_0000.json')
        hash_fields = ['history_sha256_before', 'history_sha256_after',
                       'resident_k_sha256_before', 'resident_k_sha256_after',
                       'resident_v_sha256_before', 'resident_v_sha256_after']
        assert all(s.get(k) is None for s in switches for k in hash_fields)
        row.update({'video_sha256': audit['video_sha256'], 'latent_sha256': audit['latent_sha256'],
            'outside_mask_latents_exact': True, 'changed_edited_latents': len(audit['changed_edited_latents']),
            'changed_suffix_latents': len(audit['changed_suffix_latents']),
            'additional_denoiser_calls': audit['additional_denoiser_calls'],
            'repair_stage_seconds_nested_in_render': audit['repair_stage_seconds'],
            'nonempty_prefix_tested': False, 'history_content_hashes_audited': False,
            'resident_content_hashes_audited': False, 'program_retained': False,
            'review': 'reviews/feedback_candidate_0001_shot00_attempt1/review.json'})
    else:
        assert ordinal == 1 and cs['status'] == 'resource_interrupted'
        assert not list((folder / 'videos').glob('*.mp4')) and not (folder / 'audit.json').exists()
        row.update({'unowned_compute_pids_observed': cs['other_compute_pids_observed'],
                    'completed_video': False, 'program_retained': False})
    candidates.append(row)
assert len(candidates) == 2
fixed = read(root / 'fixed_noise/status.json')
assert fixed['status'] == 'failed' and 'GPU 0 is occupied' in fixed['error']
assert read(root / 'fixed_noise/owned_processes.json')['live_owned_at_end'] == []
assert not (root / 'fixed_noise/run').exists()
review = read(root / 'reviews/feedback_candidate_0001_shot00_attempt1/review.json')
observations = read(root / 'developer_observations.json')
assert review['review_frames'] == observations['reviewed_frames']
assert review['comparison_video']['frames_decoded'] == 960
assert review['source_videos']['candidate']['sha256'] == candidates[0]['video_sha256']
summary = {
    'status': 'closed_one_rejected_candidate_then_resource_interruption', 'case': 'ns180',
    'plan_sha256': sha(root / 'plan.json'), 'current_window_seconds': [0, 10],
    'completed_paired_policy_comparison': False, 'all_budgets_closed': True,
    'render_attempts_started': 2, 'completed_candidate_videos': 1,
    'interrupted_render_attempts': 1, 'fixed_noise_render_attempts': 0,
    'automatic_retained_improvements': 0, 'remaining_case_attempts': 0,
    'unused_fixed_noise_candidate_slots_closed': 2, 'total_model_calls': len(calls),
    'text_calls': sum(not c['ordered_image_paths'] for c in calls),
    'visual_calls': sum(bool(c['ordered_image_paths']) for c in calls),
    'model_calls_by_kind': dict(collections.Counter(c['call_kind'] for c in calls)),
    'feedback': {'worker_seconds': worker['wall_seconds'], 'lifecycle_seconds': status['wall_seconds'],
        'startup_seconds': status['server_startup_seconds'], 'selection': selected,
        'model_call_seconds_sum_nested_in_worker': sum(c['wall_seconds'] for c in calls),
        'diagnosis_seconds_nested_in_worker': sum(d['wall_seconds'] for d in diagnoses),
        'segmentation_calls': segmentation, 'segmentation_seconds_nested_in_diagnosis': sum(c['wall_seconds'] for c in segmentation),
        'termination': finished['termination'], 'candidates': candidates, 'device_telemetry': telemetry,
        'current_guard': guard['decision'], 'paired_protection_or_future_checks_reached': False},
    'fixed_noise': {'status': 'failed_before_model_startup', 'lifecycle_seconds': fixed['wall_seconds'],
        'error': fixed['error'], 'model_calls': 0, 'render_attempts': 0},
    'queue_seconds': queue['wall_seconds'],
    'lifecycle_seconds_sum': status['wall_seconds'] + fixed['wall_seconds'],
    'render_supervised_seconds_sum_nested_in_worker': sum(c['render_supervised_seconds'] for c in candidates),
    'comparison_build_seconds_separate_cpu': review['wall_seconds'],
    'selected_original_sha256': parent_audit['video_sha256'],
    'selection_frozen_before_baseline_viewing': True, 'quality_acceptance': False,
    'independent_evaluation': False, 'human_labels': 0, 'held_out': False, 'submission_ready': False,
    'developer_observations': observations,
    'conclusion': 'One real automatic hair-region replay yields incomplete patchy shortening and later hair/prop/artifact changes. The checker rejects it; the original is delivered. The narrower second action is interrupted after another session occupies GPU 0, and fixed noise fails admission. No completed paired comparison or retained improvement.',
    'next_action': 'Close this case without a restart, new target/window, fixed-target probe or checker retuning. Preserve the completed video and all interrupted/admission costs.',
    'limits': ['Shared declared ceilings do not establish C0-normalized equal actual compute.',
        'GPU 0 telemetry includes the unowned job near the end and cannot be attributed entirely to this experiment.',
        'All nested times overlap their enclosing worker/render intervals; do not add them together.',
        'Raw audit equality of absent history/resident hashes is not a byte-level state audit; the edited window has an empty strict prefix.',
        'Original assignment and attribute decisions contain semantic errors; raw checker regressions are not independent ground truth.',
        'Sparse developer views are not human labels, continuous-motion review, identity acceptance or formal quality scores.']}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['status', 'completed_candidate_videos', 'total_model_calls', 'queue_seconds']}))
