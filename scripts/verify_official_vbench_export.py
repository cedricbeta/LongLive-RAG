#!/usr/bin/env python3
"""Verify portable official metric evidence without models or original paths."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess


def verify(root, name):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    folder = root / 'results' / name
    manifest = read(folder / 'export_manifest.json')
    files = manifest['files']
    for row in files.values():
        assert sha(root / row['repository_path']) == row['sha256'], row['repository_path']
    def exported(path):
        key = str(Path(path).relative_to(Path(manifest['original_runtime_root'])))
        return root / files[key]['repository_path']
    audit, plan, status = (read(folder / n) for n in ['audit.json', 'plan.json', 'status.json'])
    assert sha(root / 'implementation/scripts/audit_official_vbench_long.py') == audit['audit_source_sha256']
    for key, expected in audit['inputs_sha256'].items():
        assert files[key]['sha256'] == expected
    source_manifest = read(root / manifest['source_snapshot_manifest'])
    for path, expected in source_manifest.items():
        assert sha(root / path) == expected, path
    assets = read(root / manifest['model_manifest'])
    model_hashes = {str(Path(assets['cache_root']) / r['cache_relative_path']): r['sha256'] for r in assets['assets']}
    old_runtime = Path(manifest['original_runtime_root'])
    sources = [(old_runtime / 'official_vbench', 'vbench'),
               (Path(assets['cache_root']) / 'torch/hub/facebookresearch_dino_main', 'dino'),
               (Path(assets['cache_root']) / 'facebookresearch_dino_main', 'dino'),
               (Path(assets['cache_root']) / 'torch/hub/facebookresearch_dinov2_main', 'dinov2')]
    actual_source_count, model_count = 0, 0
    for path, expected in plan['source_and_asset_sha256'].items():
        if path in model_hashes:
            assert model_hashes[path] == expected
            model_count += 1
        elif Path(path).is_relative_to(old_runtime / name):
            assert sha(exported(path)) == expected
            actual_source_count += 1
        else:
            prefix, alias = next((prefix, alias) for prefix, alias in sources if Path(path).is_relative_to(prefix))
            target = root / 'reproduction/vbench_long/source' / alias / Path(path).relative_to(prefix)
            assert sha(target) == expected, path
            actual_source_count += 1
    telemetry = root / 'results' / (name + '_telemetry')
    measured = read(telemetry / 'summary.json')
    assert read(telemetry / 'target_terminal_status.json') == status
    assert sha(telemetry / 'target_terminal_status.json') == measured['target_status_sha256']
    assert sha(telemetry / 'samples.jsonl') == measured['samples_sha256']
    tplan = read(telemetry / 'plan.json')
    assert sha(folder / 'plan.json') == tplan['target_plan_sha256']
    assert sha(folder / 'manifest.json') == tplan['target_manifest_sha256']
    assert sha(telemetry / 'monitor_source.py') == tplan['source_sha256']
    spec = importlib.util.spec_from_file_location('telemetry', telemetry / 'monitor_source.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calculated = module.summarize([json.loads(line) for line in (telemetry / 'samples.jsonl').read_text().splitlines()], status)
    assert calculated == {k: v for k, v in measured.items() if k not in ('samples_sha256', 'target_status_sha256')}
    assert measured == audit['telemetry']
    if status['status'] == 'completed':
        raw_path, = (folder / 'outputs').glob('*_eval_results.json')
        raw = read(raw_path)
        assert set(raw) == set(plan['dimensions']) and len(audit['rows']) == 6
        expected_clips = {f'moviegen000-0_{index:03d}.mp4' for index in range(60)}
        metadata_path, = (folder / 'outputs').glob('*_full_info.json')
        metadata = read(metadata_path)
        assert len(metadata) == 1 and len(metadata[0]['video_list']) == 60
        assert {Path(p).name for p in metadata[0]['video_list']} == expected_clips
        for row in audit['rows']:
            overall, details = raw[row['dimension']][:2]
            assert len(details) == row['raw_detail_records'] == 60
            scale = .01 if row['dimension'] == 'imaging_quality' else 1
            assert math.isclose(statistics.mean(float(v['video_results']) for v in details) * scale,
                                overall, rel_tol=1e-10, abs_tol=1e-10)
            assert row['official_score'] == overall and row['independent_video_count'] == 1
            if row['dimension'] in ('subject_consistency', 'background_consistency'):
                assert {Path(v['video_path']).parent.name for v in details} == expected_clips
                assert {v['clip2clip_score'] for v in details} == {row['clip_to_clip_score']}
                for value in details:
                    mapped = value['mapped_clip2clip_score']
                    fused = .5 * value['inclip_score'] + .5 * mapped if mapped != 0 else value['inclip_score']
                    assert math.isclose(fused, value['video_results'], abs_tol=1e-12)
            else:
                assert {Path(v['video_path']).name for v in details} == expected_clips
        assert len(audit['clips']) == 60
        for clip in audit['clips']:
            path = exported(old_runtime / name / clip['path'])
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames',
                '-of', 'json', str(path)], text=True))['streams'][0]
            assert sha(path) == clip['sha256'] and probe == clip['probe']
        for dimension in ('subject_consistency', 'background_consistency'):
            path = exported(old_runtime / name / 'input' /
                (dimension + '_cat_firstframes_videos') / 'moviegen000-0.mp4')
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,avg_frame_rate',
                '-of', 'json', str(path)], text=True))['streams'][0]
            assert probe == {'avg_frame_rate': '16/1', 'nb_read_frames': '60'}
    else:
        assert status['status'] in ('failed', 'timeout') and not audit['rows']
    return {'run': name, 'terminal_status': status['status'], 'files_verified': len(files),
            'frozen_sources_and_inputs_verified': actual_source_count, 'model_hash_bindings_verified': model_count,
            'raw_models_downloaded_or_reloaded': False, 'official_metrics_reexecuted': False,
            'score_aggregates_checked': len(audit['rows']), 'actual_clips_decoded_and_counted': len(audit['clips']),
            'consistency_fusion_checked': status['status'] == 'completed',
            'telemetry_recomputed_exactly': True, 'independent_video_count': 1}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root.resolve(), args.name)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
