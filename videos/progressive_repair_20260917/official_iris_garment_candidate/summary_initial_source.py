#!/usr/bin/env python3
"""Summarize one completed exposed native trial without claiming acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    root = args.root
    sources = {}
    def read(relative):
        path = root / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    candidate = 'official_iris_garment_candidate'
    probes = 'official_iris_candidate_diagnostics'
    audit = read(candidate + '/audit.json')
    manifest = read(candidate + '/manifest.json')
    plan = read(candidate + '/plan.json')
    baseline = read('official_iris_60s_baseline_prepared/audit.json')
    before = read('official_iris_attribute_temporal_fix/result.json')['attribute_assessments']
    after = read(probes + '/candidate_current/result.json')['result']['attribute_assessments']
    later_parent = read(probes + '/parent_later/result.json')['result']['attribute_assessments']
    later_candidate = read(probes + '/candidate_later/result.json')['result']['attribute_assessments']
    guards = {name: read(probes + '/' + name + '/result.json')['review'] for name in ['current_guard', 'later_guard']}
    faces = read('official_iris_face_pair_diagnostic/result.json')
    stages = read(probes + '/status.json')
    assert stages['status'] == 'completed' and audit['all_engineering_checks_passed']
    assert plan['attempt_budget'] == 1 and manifest['automatic_progressive_controller'] is False
    def paired(left, right):
        other = {row['id']: row for row in right}
        assert set(other) == {row['id'] for row in left}
        result = []
        for row in left:
            new = other[row['id']]
            for key in ['part', 'aspect', 'expected']:
                assert row[key] == new[key]
            result.append({key: row[key] for key in ['id', 'part', 'aspect', 'expected']} |
                          {'parent_raw_status': row['status'], 'candidate_raw_status': new['status']})
        return result
    successful_calls = []
    for directory in [root / 'official_iris_baseline_attributes_gpu2', root / probes]:
        for path in directory.rglob('vlm_cache/*.json'):
            item = json.loads(path.read_text())
            successful_calls.append({'path': str(path.relative_to(root)), 'kind': item['call_kind'],
                                     'wall_seconds': item['wall_seconds']})
    mask = read('official_iris_garment_mask/result.json')
    cost = {'baseline_render_wall_seconds': baseline['wall_seconds'],
            'mask_cpu_wall_seconds': mask['wall_seconds'],
            'preparation_asset_verification_wall_seconds': manifest['preparation_wall_seconds'],
            'candidate_render_wall_seconds': audit['status']['wall_seconds'],
            'repair_stage_seconds_included_in_candidate_render': audit['repair_stage_seconds'],
            'engineering_audit_cpu_wall_seconds': audit['audit_wall_seconds'],
            'completed_parent_appearance_wall_seconds': read('official_iris_baseline_attributes_gpu2/result.json')['wall_seconds'],
            'five_post_generation_probes_wall_seconds': stages['wall_seconds'],
            'face_pair_cpu_wall_seconds': faces['wall_seconds'],
            'successful_model_calls': len(successful_calls), 'model_call_records': successful_calls,
            'failed_client': read('official_iris_baseline_attributes/status.json'),
            'initial_preparation_failure': read('official_iris_candidate_preparation_failure.json'),
            'server_lifecycle_records': ['official_iris_attribute_server/launch.json',
                'official_iris_attribute_server/forced_cleanup.json',
                'official_iris_attribute_server_gpu2/watchdog_result.json',
                'official_iris_candidate_judge_server/watchdog_result.json'],
            'limits': ['Stage walls can overlap and are not an end-to-end sum.',
                       'Model call time is included in probe walls; do not add it again.',
                       'Server lifecycle includes model load, idle and teardown, not kernel GPU time.',
                       'Failed resource admission and client time remain in the raw records.']}
    result = {'phase': 'official_native_iris_fixed_candidate_development_diagnostic',
        'status': 'completed_unretained_unresolved', 'held_out': False,
        'automatic_progressive_controller': False, 'candidate_retained': False,
        'retained_improvements': 0, 'independent_final_evaluation': False,
        'fixed_target': plan['target'], 'actual_action_fingerprint': manifest['action_fingerprint'],
        'current_raw_attribute_comparison': paired(before, after),
        'later_raw_attribute_comparison': paired(later_parent, later_candidate),
        'current_frame_indices': [352, 400, 448], 'later_frame_indices': [672, 720, 768],
        'raw_development_guards': guards, 'face_pair_diagnostic': faces['rows'],
        'selected_branch': 'official_iris_60s_baseline_prepared',
        'selected_video_sha256': audit['video_comparison']['parent_sha256'],
        'unresolved_parent_predicate_ids': [row['id'] for row in before if row['status'] != 'pass'],
        'decision_basis': 'Reliable identity/content preservation has not been established; keep the original parent. '
                          'Neither the known-unreliable guard nor uncalibrated face evidence can establish acceptance.',
        'diagnostic_failures': ['Later parent appearance calls again broaden apron to coat despite the aggregation fix.',
            'Both protection calls output identity fail while explaining insufficient facial evidence.',
            'Current protection conflates existing parent clothing error with candidate degradation.',
            'Later protection names hair color as the target although the packet targets garment color.',
            'Independent face detector returns zero detections on all 12 sampled images; this does not prove absent faces.'],
        'engineering': {key: audit[key] for key in ['strict_prefix_latents', 'prefix_latents_exact',
            'outside_mask_latents_exact', 'additional_denoiser_calls', 'retrieval_queries',
            'retrieval_records_identical_to_parent']},
        'changed_edited_latents': len(audit['changed_edited_latents']),
        'changed_suffix_latents': len(audit['changed_suffix_latents']),
        'compressed_prefix_frames_different': sum(i < 320 for i in audit['video_comparison']['changed_frames']),
        'raw_vae_prefix_pixels_audited': False, 'cost': cost, 'input_sha256': sources,
        'summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': ['One exposed case, one fixed prompt/mask/noise attempt; no adaptive feedback comparison.',
                  'Reported fail-to-pass is a development judge observation, not independent ground truth.',
                  'One later window of three frames cannot certify complete actions, motion or the suffix.',
                  'No human evaluation, formal score, budget-matched efficacy result or accepted identity reference.']}
    if args.output.exists():
        raise FileExistsError('Use a fresh summary path')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'successful_model_calls': len(successful_calls),
                      'candidate_render_wall_seconds': cost['candidate_render_wall_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
