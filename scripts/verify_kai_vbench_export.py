#!/usr/bin/env python3
"""Verify the focused portable three-video metric bundle, without model execution."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify(root):
    read = lambda p: json.loads(p.read_text())
    folder = root / 'results/native_kai_vbench_visual_v1'
    manifest = read(folder / 'export_manifest.json')
    files = manifest['files']
    for row in files.values():
        assert sha(root / row['repository_path']) == row['sha256'], row['repository_path']
    old_root = Path(manifest['runtime_root'])
    assets = read(root / 'results/official_vbench_provision_v1/model_assets_manifest.json')
    model_hashes = {str(Path(assets['cache_root']) / row['cache_relative_path']): row['sha256'] for row in assets['assets']}
    old_cache = Path(assets['cache_root'])
    prefixes = [(old_root.parent / 'official_vbench', 'vbench'),
        (old_cache / 'torch/hub/facebookresearch_dino_main', 'dino'),
        (old_cache / 'facebookresearch_dino_main', 'dino'),
        (old_cache / 'torch/hub/facebookresearch_dinov2_main', 'dinov2')]
    clips, source_bindings, model_bindings = 0, 0, 0
    for arm in ('B_original', 'P20_prompt', 'RP_local_cue'):
        audit, plan, status = (read(folder / arm / name) for name in ('audit.json', 'plan.json', 'status.json'))
        assert sha(root / 'implementation/scripts/audit_vbench_custom_video.py') == audit['audit_source_sha256']
        for key, expected in audit['inputs_sha256'].items():
            assert files[key]['sha256'] == expected
        for path, expected in plan['source_and_asset_sha256'].items():
            if path in model_hashes:
                assert model_hashes[path] == expected
                model_bindings += 1
            elif Path(path).is_relative_to(old_root):
                assert files[str(Path(path).relative_to(old_root))]['sha256'] == expected
            else:
                prefix, alias = next((prefix, alias) for prefix, alias in prefixes if Path(path).is_relative_to(prefix))
                assert sha(root / 'reproduction/vbench_long/source' / alias / Path(path).relative_to(prefix)) == expected
                source_bindings += 1
        telemetry = folder / (arm + '_telemetry')
        measured, tplan = read(telemetry / 'summary.json'), read(telemetry / 'plan.json')
        assert read(telemetry / 'target_terminal_status.json') == status
        assert sha(telemetry / 'target_terminal_status.json') == measured['target_status_sha256']
        assert sha(telemetry / 'samples.jsonl') == measured['samples_sha256']
        assert sha(folder / arm / 'plan.json') == tplan['target_plan_sha256']
        assert sha(folder / arm / 'manifest.json') == tplan['target_manifest_sha256']
        assert sha(telemetry / 'monitor_source.py') == tplan['source_sha256']
        recomputed = module('telemetry_' + arm, telemetry / 'monitor_source.py').summarize(
            [json.loads(line) for line in (telemetry / 'samples.jsonl').read_text().splitlines()], status)
        assert recomputed == {key: value for key, value in measured.items() if key not in ('samples_sha256', 'target_status_sha256')}
        assert audit['telemetry'] == measured
        if status['status'] != 'completed':
            assert status['status'] in ('failed', 'timeout') and not audit['rows']
            continue
        expected = {f'moviegen000-0_{i:03d}.mp4' for i in range(30)}
        raw_path, = (folder / arm / 'outputs').glob('*_eval_results.json')
        full_path, = (folder / arm / 'outputs').glob('*_full_info.json')
        metadata, raw = read(full_path), read(raw_path)
        assert len(metadata) == 1 and len(metadata[0]['video_list']) == 30
        assert {Path(p).name for p in metadata[0]['video_list']} == expected
        for dimension in plan['dimensions']:
            overall, details = raw[dimension][:2]
            assert len(details) == 30 and math.isfinite(overall)
            if dimension in ('subject_consistency', 'background_consistency'):
                assert {Path(row['video_path']).parent.name for row in details} == expected
                for row in details:
                    mapped = row['mapped_clip2clip_score']
                    fused = .5 * row['inclip_score'] + .5 * mapped if mapped != 0 else row['inclip_score']
                    assert math.isclose(fused, row['video_results'], abs_tol=1e-12)
            else:
                assert {Path(row['video_path']).name for row in details} == expected
        assert len(audit['clips']) == 30
        for clip in audit['clips']:
            path = root / files[arm + '/' + clip['path']]['repository_path']
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames', '-of', 'json', str(path)], text=True))['streams'][0]
            assert sha(path) == clip['sha256'] and probe == clip['probe']
            clips += 1
        for dimension in ('subject_consistency', 'background_consistency'):
            key = arm + '/input/' + dimension + '_cat_firstframes_videos/moviegen000-0.mp4'
            path = root / files[key]['repository_path']
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames,avg_frame_rate', '-of', 'json', str(path)], text=True))['streams'][0]
            assert probe == {'avg_frame_rate': '16/1', 'nb_read_frames': '30'}
    summary = module('summary', root / 'implementation/scripts/summarize_kai_vbench.py').summarize(folder)
    assert summary == read(folder / 'summary.json')
    return {'status': 'passed', 'manifest_files': len(files), 'clips_decoded': clips,
        'source_bindings_checked': source_bindings, 'model_hash_bindings_checked': model_bindings,
        'actual_models_reloaded': False, 'score_aggregates_and_differences_recomputed': True,
        'telemetry_recomputed': True, 'independent_script_count': 1, 'quality_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
