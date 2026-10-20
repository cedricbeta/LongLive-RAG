#!/usr/bin/env python3
"""Recompute every official scalar and paired difference for three closed videos."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(folder):
    read = lambda p: json.loads(p.read_text())
    plan = read(folder / 'plan.json')
    assert plan['evaluation_ceiling'] == 3
    arms = ['B_original', 'P20_prompt', 'RP_local_cue']
    runs, scores = {}, {}
    for arm in arms:
        status = read(folder / arm / 'status.json')
        audit = read(folder / arm / 'audit.json')
        assert status['status'] == audit['status']
        assert status['status'] in ('completed', 'failed', 'timeout')
        runs[arm] = {'status': status['status'], 'wall_seconds': status['wall_seconds'],
            'video_sha256': plan['inputs'][arm]['sha256'],
            'generation_seconds_previously_incurred': plan['inputs'][arm]['generation_seconds_previously_incurred'],
            'device_energy_bounds': audit['telemetry']['device_energy_bounds'],
            'observed_foreign_compute_pids': audit['telemetry']['observed_foreign_compute_pids_inside']}
        scores[arm] = {}
        if status['status'] != 'completed':
            continue
        raw_path, = (folder / arm / 'outputs').glob('*_eval_results.json')
        raw = read(raw_path)
        assert set(raw) == set(plan['dimensions'])
        for dimension in plan['dimensions']:
            overall, details = raw[dimension][:2]
            assert len(details) == 30
            scalar = statistics.mean(float(row['video_results']) for row in details)
            scalar *= .01 if dimension == 'imaging_quality' else 1
            assert math.isclose(scalar, overall, rel_tol=1e-10, abs_tol=1e-10)
            assert next(row for row in audit['rows'] if row['dimension'] == dimension)['official_score'] == overall
            scores[arm][dimension] = overall
    rows = []
    for dimension in plan['dimensions']:
        row = {'dimension': dimension, **{arm: scores[arm].get(dimension) for arm in arms}}
        for left, right in [('RP_local_cue', 'P20_prompt'), ('P20_prompt', 'B_original'), ('RP_local_cue', 'B_original')]:
            row[left + '_minus_' + right] = (row[left] - row[right]) if row[left] is not None and row[right] is not None else None
        rows.append(row)
    return {'protocol': plan['protocol'], 'status': 'closed', 'remaining_evaluation_budget': 0,
        'new_video_generations': 0, 'new_semantic_checker_calls': 0, 'runs': runs, 'rows': rows,
        'evaluation_supervisor_seconds_sum': sum(row['wall_seconds'] for row in runs.values()),
        'preparation_seconds': read(folder / 'preparation.json')['wall_seconds'],
        'queue_seconds_including_observers_and_audits': read(folder / 'queue_status.json')['wall_seconds'],
        'paired_script_count': 1, 'generation_seeds_per_condition': 1, 'clips_per_video': 30,
        'primary_comparison': 'RP_local_cue minus P20_prompt, all six raw-scale metrics',
        'overall_score': None, 'confidence_interval': None, 'held_out': False, 'human_labels': 0,
        'automatic_acceptance': False, 'entity_or_identity_acceptance': False, 'net_quality_superiority_established': False,
        'source_qualitative_conclusion': read(folder / 'source_summary.json')['decision'],
        'limits': ['One previously exposed script/seed; no variance or generalization estimate.',
            'Thirty correlated clips are not independent samples. Dynamic degree is a motion measurement, not sufficient quality evidence.',
            'These six visual metrics do not establish target hair-color correctness, actor identity, action compliance, or absence of new errors.',
            'No score selects or accepts a video. No composite overall score or noninferiority claim.',
            'Historical generation, segmentation, preparation and publication costs are separate; private sampler time is nested in generation.',
            'Sum of concurrent supervisor durations is not elapsed queue time. NVML device energy includes idle power; sampled utilization is not kernel GPU time.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.folder)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'rows': result['rows'], 'evaluation_seconds': result['evaluation_supervisor_seconds_sum']}))
