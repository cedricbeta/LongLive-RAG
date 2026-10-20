#!/usr/bin/env python3
"""Audit terminal official metrics, actual encoded clips and per-video scope."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(folder, telemetry, output):
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(folder.parent))] = digest(path)
        return json.loads(path.read_text())
    status, plan = read(folder / 'status.json'), read(folder / 'plan.json')
    if status['status'] not in ('completed', 'failed', 'timeout'):
        raise ValueError('Wait for the actual terminal state')
    video = folder / 'input/moviegen000-0.mp4'
    assert digest(video) == plan['video_sha256']
    inputs[str(video.relative_to(folder.parent))] = digest(video)
    inputs[str((folder / 'evaluation.log.txt').relative_to(folder.parent))] = digest(folder / 'evaluation.log.txt')
    verified = 0
    for name, expected in plan['source_and_asset_sha256'].items():
        assert digest(Path(name)) == expected, name
        verified += 1
    measured = read(telemetry / 'summary.json')
    tstatus = read(telemetry / 'target_terminal_status.json')
    assert tstatus == status and measured['owned_generator_pid'] == status['pid']
    assert measured['target_status_sha256'] == digest(telemetry / 'target_terminal_status.json')
    samples = telemetry / 'samples.jsonl'
    assert digest(samples) == measured['samples_sha256']
    inputs[str(samples.relative_to(folder.parent))] = digest(samples)
    source = telemetry / 'monitor_source.py'
    tplan = read(telemetry / 'plan.json')
    assert digest(source) == tplan['source_sha256']
    assert digest(folder / 'plan.json') == tplan['target_plan_sha256']
    spec = importlib.util.spec_from_file_location('telemetry_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reconstructed = module.summarize([json.loads(line) for line in samples.read_text().splitlines()], status)
    assert reconstructed == {k: v for k, v in measured.items() if k not in ('samples_sha256', 'target_status_sha256')}
    rows, clips = [], []
    outputs = list((folder / 'outputs').glob('*_eval_results.json'))
    if status['status'] == 'completed':
        assert status['returncode'] == 0 and len(outputs) == 1
        raw = read(outputs[0])
        assert set(raw) == set(plan['dimensions'])
        full_info, = (folder / 'outputs').glob('*_full_info.json')
        metadata = read(full_info)
        assert len(metadata) == 1 and len(metadata[0]['video_list']) == 60
        expected = {f'moviegen000-0_{index:03d}.mp4' for index in range(60)}
        assert {Path(p).name for p in metadata[0]['video_list']} == expected
        for name in sorted(expected):
            path = folder / 'input/split_clip/moviegen000-0' / name
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames',
                '-of', 'json', str(path)], text=True))['streams'][0]
            assert probe == {'width': 832, 'height': 480, 'avg_frame_rate': '16/1',
                             'duration': '2.000000', 'nb_read_frames': '32'}, (name, probe)
            sha = digest(path)
            inputs[str(path.relative_to(folder.parent))] = sha
            clips.append({'path': str(path.relative_to(folder)), 'sha256': sha, 'probe': probe})
        for dimension in plan['dimensions']:
            result = raw[dimension]
            overall, details = result[:2]
            assert len(details) == 60
            values = [float(row['video_results']) for row in details]
            assert all(math.isfinite(v) for v in values) and math.isfinite(overall)
            scale = .01 if dimension == 'imaging_quality' else 1
            recomputed = statistics.mean(values) * scale
            assert math.isclose(overall, recomputed, rel_tol=1e-10, abs_tol=1e-10), dimension
            consistency = dimension in ('subject_consistency', 'background_consistency')
            if consistency:
                assert {Path(row['video_path']).parent.name for row in details} == expected
                assert len({row['clip2clip_score'] for row in details}) == 1
                for row in details:
                    fused = (.5 * row['inclip_score'] + .5 * row['mapped_clip2clip_score']) if row['mapped_clip2clip_score'] != 0 else row['inclip_score']
                    assert math.isclose(fused, row['video_results'], abs_tol=1e-12)
            else:
                assert {Path(row['video_path']).name for row in details} == expected
                assert len(result) == 3 and len(result[2]) == 60
            rows.append({'dimension': dimension, 'official_score': overall,
                         'recomputed_from_all_raw_records': recomputed, 'raw_detail_records': len(details),
                         'independent_video_count': 1, 'confidence_interval': None,
                         'clip_to_clip_score': details[0]['clip2clip_score'] if consistency else None})
        for dimension in ('subject_consistency', 'background_consistency'):
            path = folder / 'input' / (dimension + '_cat_firstframes_videos/moviegen000-0.mp4')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,avg_frame_rate',
                '-of', 'json', str(path)], text=True))['streams'][0]
            assert probe == {'avg_frame_rate': '16/1', 'nb_read_frames': '60'}
            inputs[str(path.relative_to(folder.parent))] = digest(path)
    else:
        assert not outputs, 'Do not silently discard partial output from a failed attempt'
    result = {'phase': 'official_vbench_long_terminal_audit', 'run': folder.name,
        'status': status['status'], 'video_sha256': plan['video_sha256'], 'held_out': False,
        'supervisor_seconds': status['wall_seconds'], 'official_source_revision': plan['revision'],
        'source_and_asset_hashes_verified_after_run': verified, 'rows': rows, 'clips': clips,
        'telemetry': measured, 'telemetry_recomputed_exactly': True, 'independent_video_count': 1,
        'actual_human_labels': 0, 'repair_benefit_evaluation': False, 'overall_score': None,
        'limits': ['One exposed baseline video; no repair arm, held-out generalization, uncertainty interval or human labels.',
            'Six supported visual metrics in official long_custom_input mode; no full 16-dimension suite or instruction-compliance score.',
            'Original code emits per-clip-derived video paths even in its reorganized records. All 60 records are one original video, never 60 independent samples.',
            'All scalar means are checked against every raw record, including official imaging normalization and slow-fast fusion.',
            'Metric scoring and preceding generation are separate measured costs. Device energy includes idle power; utilization is a proxy, not kernel time.'],
        'inputs_sha256': inputs, 'audit_source_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': status['status'], 'metrics': rows, 'clips': len(clips), 'seconds': status['wall_seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--telemetry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.folder.resolve(), args.telemetry.resolve(), args.output.resolve())
