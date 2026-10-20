#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, inputs, result, status, responses = [read(root / name) for name in ['plan.json', 'inputs.json', 'result.json', 'status.json', 'responses.json']]
assert status['status'] == result['status'] == 'completed'
assert result['requests'] == plan['maximum_requests'] == len(responses) == 2
assert [row['case'] for row in responses] == plan['cases'] == ['ns010', 'ns070']
assert sha(root / 'worker.py') == inputs['worker_source_sha256']
assert sha(root / 'launch.py') == inputs['launcher_source_sha256']
for version in [1, 2]:
    old = root.parent / f'molmo2_o7b_visual_preflight_v{version}'
    assert sha(old / 'worker.py') == sha(root / 'worker.py')
    assert sha(old / 'plan.json') == sha(root / 'plan.json')
    abort = read(old / 'dependency_abort.json')
    assert abort['requests'] == abort['model_loads'] == 0
for case in inputs['cases']:
    assert [row['frame'] for row in case['frames']] == plan['frame_indices']
    for row in case['frames']:
        assert sha(root / row['path']) == row['sha256']
    original = root.parent / 'native_development_pilot12_batch2_v1' / case['id'] / 'baseline'
    assert case['video_sha256'] == read(original / 'audit.json')['video_sha256']
assert not Path('/proc', str(status['pid'])).exists()
assert not Path('/proc', str(status['supervisor_pid'])).exists()
telemetry = read(root / 'telemetry/summary.json')
assert read(root / 'telemetry/target_terminal_status.json') == status
provision = root.parent / 'molmo2_o7b_provision_v3'
assert read(provision / 'status.json')['status'] == 'completed'
assert sha(provision / 'assets_manifest.json') == inputs['assets_manifest_sha256']
stages = []
for version in [1, 2, 3]:
    folder = root.parent / f'molmo2_o7b_provision_v{version}'
    row = read(folder / 'status.json')
    assert not Path('/proc', str(row['pid'])).exists()
    stages.append({'version': version, **row})
summary = {
    'status': 'completed', 'model': plan['model'], 'revision': plan['revision'],
    'actual_model_loads': 1, 'actual_observation_calls': 2, 'previous_dependency_attempt_model_loads': 0,
    'remaining_call_budget': 0, 'no_change_to_original_question_frames_cases_or_worker': True,
    'supervisor_seconds': status['wall_seconds'], 'model_load_seconds': result['model_load_seconds'],
    'worker_seconds': result['worker_seconds'], 'peak_gpu_memory_allocated_bytes': result['peak_gpu_memory_allocated_bytes'],
    'calls': [{'case': row['case'], 'status': row['status'], 'wall_seconds': row['wall_seconds'],
        'input_tokens': row['input_tokens'], 'output_token_count': len(row['output_tokens']),
        'reached_token_ceiling': row['reached_token_ceiling']} for row in responses],
    'provision_stages': stages, 'provision_stage_seconds_sum': sum(row['wall_seconds'] for row in stages),
    'cpu_preflight_seconds': read(provision / 'cpu_preflight.json')['wall_seconds'],
    'all_shard_diagnostic_seconds': read(root.parent / 'molmo2_o7b_provision_v2/all_shard_digests.json')['wall_seconds'],
    'provision_limits': 'Stage sum excludes the separately listed all-shard diagnostic and CPU preflight; queue/developer wait is unmeasured. Initial three incorrect cached shards are retained; the cause is unknown. Fresh recovery verified all 27 files.',
    'device_energy_bounds': telemetry['device_energy_bounds'],
    'observed_foreign_compute_pids': telemetry['observed_foreign_compute_pids_inside'],
    'developer_review': {
        'all_ten_actual_input_images_viewed': True, 'human_labels': 0, 'independent_accuracy_evaluation': False,
        'ns010': 'One standing woman and green outerwear agree with the supplied views. The reply lists a white cup with objects despite the question about held objects; the cup is on the table in these views. It does not distinguish held versus merely visible objects and repeats same-objects descriptions despite changing hand/notebook configurations.',
        'ns070': 'Three seated people agree with the supplied views. Blue/white/red is coarse: left clothing appears teal and center pale lavender/gray in the images. Object descriptions lack per-person ownership.',
        'temporal_limit': 'No entrances, exits or transfers are established by these first-ten-second image sets. These are easy mostly stationary sequences, not a test of later arrivals, missing participants, object transfer or repair side effects. Sparse frames cannot support the reply\'s exhaustive statement about all time between samples.'
    },
    'semantic_acceptance': False, 'official_metric_runs': 0, 'candidate_selection_calls': 0,
    'new_video_generations': 0, 'automatic_retained_improvements': 0, 'human_labels': 0,
    'model_adopted_as_final_evaluator': False, 'submission_ready': False,
    'conclusion': 'Pinned local BF16 multi-image execution works. Two observations do not calibrate an independent final judge or demonstrate repair efficacy.',
    'inputs_sha256': sha(root / 'inputs.json'), 'responses_sha256': sha(root / 'responses.json'),
    'plan_sha256': sha(root / 'plan.json'), 'assets_manifest_sha256': inputs['assets_manifest_sha256']
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['actual_model_loads', 'actual_observation_calls', 'supervisor_seconds', 'provision_stage_seconds_sum', 'semantic_acceptance']}))
