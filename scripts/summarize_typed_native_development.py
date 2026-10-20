#!/usr/bin/env python3
"""Summarize archived development observations without interpreting them as quality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(root):
    inputs = {}

    def read(relative):
        path = root / relative
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def calls(folder):
        paths = sorted((root / folder / 'run/vlm_cache').rglob('*.json'))
        records = [read(str(path.relative_to(root))) for path in paths]
        assert all(not r['parse_attempts'] for r in records), 'Audit retries separately'
        return {'recorded_model_calls': len(records),
                'call_kinds': dict(Counter(r['call_kind'] for r in records)),
                'recorded_call_wall_seconds': sum(r['wall_seconds'] for r in records),
                'image_calls': sum(bool(r['ordered_image_paths']) for r in records)}

    compiler = []
    for name in ['typed_script_compiler_v4_development',
                 'typed_script_compiler_v5_development',
                 'typed_script_compiler_v5_schema_context']:
        folder = 'results/' + name
        status = read(folder + '/run/status.json')
        cases = read(folder + '/run/results.json')
        plan = read(folder + '/plan.json')
        lifecycle = read(folder + '/server_lifecycle.json')
        assert status['status'] == 'completed' and len(cases) == 3
        assert [r['source_benchmark_index'] for r in cases] == [90, 310, 50]
        assert plan['held_out'] is False and status['new_video_generations'] == 0
        compiler.append({'run': name, 'worker_wall_seconds': status['wall_seconds'],
                         'server_lifecycle_seconds': lifecycle['observed_process_lifecycle_seconds'],
                         'program_ready_cases': [r['source_benchmark_index'] for r in cases if r['status'] == 'program_ready'],
                         'cases': cases, **calls(folder)})

    folder = 'results/native_typed_diagnosis_v5'
    status = read(folder + '/run/status.json')
    manifest = read(folder + '/run/manifest.json')
    selected = read(folder + '/run/selected.json')
    diagnosis = read(folder + '/run/diagnosis_calls.json')
    segmentation = read(folder + '/run/segmentation_cost.json')
    branches = read(folder + '/run/backend/branches.json')
    memory = read(folder + '/run/memory_audit.json')
    lifecycle = read(folder + '/server_lifecycle.json')
    assert status['status'] == 'completed' and status['shots_completed'] == 2
    assert manifest['max_edits_per_shot'] == manifest['lookahead_shots'] == 0
    assert manifest['current_shots_requested'] == [0, 1] and not manifest['full_video_verified']
    assert not memory['identity_references_promoted']
    shots = []
    for record in diagnosis:
        metrics = record['verdict']['metrics']
        constraints = metrics['constraint_assessments']
        shots.append({'shot': record['shot'], 'wall_seconds': record['wall_seconds'],
                      'accept': record['verdict']['accept'], 'reason': record['verdict']['reason'],
                      'constraint_counts': dict(Counter(r['status'] for r in constraints)),
                      'unresolved_constraints': [r for r in constraints if r['status'] != 'satisfied'],
                      'native_regions_offered': len(metrics['native_mask_projection']),
                      'repair_options': len(metrics['repair_options']),
                      'identity_references_promoted': metrics['native_adapter']['identity_references_promoted']})
    native = {'run': 'native_typed_diagnosis_v5', 'worker_wall_seconds': status['wall_seconds'],
              'server_lifecycle_seconds': lifecycle['observed_process_lifecycle_seconds'],
              'shots': shots, 'selected': selected, 'branch_registry': branches,
              'segmentation_cost': segmentation, 'new_video_generations': 0,
              'retained_improvements': 0, 'future_shots_checked': 0,
              'full_video_verified': False, **calls(folder)}
    components = read('results/native_multientity_components_v1/run/result.json')
    component_status = read('results/native_multientity_components_v1/run/status.json')
    assert all(r['all_frame_masks_exact'] and r['frames'] == 160 for r in components['reproduction'])
    tests = read('results/typed_native_contracts/exported_tests.json')
    assert tests['returncode'] == 0
    return {'compiler_runs': compiler, 'native_diagnosis': native,
            'mask_components': {'worker_wall_seconds': component_status['wall_seconds'],
                                'tracking_wall_seconds': components['tracking_wall_seconds'],
                                'clipseg_video_passes': len(components['segmentation_cost']),
                                'reproduction': components['reproduction'],
                                'clara_garment': [r for r in components['rows'] if r['entity'] == 'Clara' and r['part'] == 'garment'],
                                'new_vlm_calls': 0, 'new_video_generations': 0},
            'all_recorded_vlm_calls': sum(r['recorded_model_calls'] for r in compiler) + native['recorded_model_calls'],
            'new_video_generations': 0, 'retained_improvements': 0,
            'actual_human_labels': 0, 'held_out': False, 'quality_claim': False,
            'official_automatic_repair_completed': False,
            'limits': ['Sequential exposed development, not formal compiler accuracy or repair efficacy.',
                       'Program-ready does not establish semantic correctness; Iris remains unresolved.',
                       'Compiler alone receives schema text. Visual checker remains uncalibrated.',
                       'Worker/call/server times overlap; do not add them. Kernel GPU time is not measured.',
                       'Zero new VLM calls for components still includes real SAM2 and CLIPSeg computation.',
                       'Two current shots and zero edits do not exercise the native renderer, paired acceptance or future checks.'],
            'inputs_sha256': inputs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'compiler_calls': [r['recorded_model_calls'] for r in result['compiler_runs']],
                      'native_calls': result['native_diagnosis']['recorded_model_calls'],
                      'new_video_generations': result['new_video_generations']}))
