#!/usr/bin/env python3
"""Summarize both real metric attempts and the separately measured generation."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    inputs = {}
    def read(name):
        path = root / name
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    base = read('results/official_moviegen_000_120s_v1/audit.json')
    failed = read('results/official_vbench_moviegen000_120s_v1/audit.json')
    success = read('results/official_vbench_moviegen000_120s_v2/audit.json')
    recovery = read('results/official_vbench_moviegen000_120s_v2/recovery_binding.json')
    plans = [read(f'results/official_vbench_moviegen000_120s_v{v}/plan.json') for v in (1, 2)]
    assert failed['status'] == 'failed' and not failed['rows']
    assert success['status'] == 'completed' and len(success['rows']) == 6
    assert base['video_sha256'] == failed['video_sha256'] == success['video_sha256']
    assert recovery['identical_after_removing_run_paths_timestamp_and_av']
    assert [p['package_versions']['av'] for p in plans] == ['17.0.0', '12.0.0']
    evaluation_seconds = failed['supervisor_seconds'] + success['supervisor_seconds']
    return {'phase': 'official_long_custom_input_development_scores',
        'video_sha256': base['video_sha256'], 'video': 'artifacts/official_moviegen_000_120s_v1/videos/rank0-0-0_lora.mp4',
        'official_source_revision': success['official_source_revision'], 'independent_video_count': 1,
        'held_out': False, 'actual_human_labels': 0, 'repair_comparison': False,
        'rows': success['rows'], 'overall_score': None, 'confidence_interval': None,
        'costs': {'generation_supervisor_seconds': base['wall_seconds'],
            'failed_evaluation_supervisor_seconds': failed['supervisor_seconds'],
            'successful_evaluation_supervisor_seconds': success['supervisor_seconds'],
            'all_evaluation_attempt_seconds': evaluation_seconds,
            'separate_generation_and_evaluation_stage_sum_seconds': base['wall_seconds'] + evaluation_seconds,
            'scope': 'Includes cold model loading, preprocessing and scoring within each supervisor. Generation and evaluation are separately measured stages, not a newly timed contiguous pipeline. Environment provisioning, downloads and evidence export are excluded.'},
        'attempts': [{'run': failed['run'], 'status': failed['status'], 'metrics_produced': 0},
                     {'run': success['run'], 'status': success['status'], 'metrics_produced': 6}],
        'interpretation': 'Official visual metrics for one already exposed baseline; not entity-constraint accuracy, instruction compliance, repair efficacy or the full standard-prompt suite.',
        'bookkeeping_limit': 'Upstream reorganized records have clip-derived paths. The six scalars are verified against all 60 raw records; sample size is one original video.',
        'inputs_sha256': inputs}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'metrics': len(result['rows']), 'costs': result['costs']}))
