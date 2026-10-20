#!/usr/bin/env python3
"""Validate one real exact-duration engineering run; no visual-quality score."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    status = json.loads((folder / 'status.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert status['status'] == 'completed'
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    assert all(sha(Path(manifest['cwd']) / name) == digest
               for name, digest in manifest['source_sha256'].items())
    timing = json.loads((folder / 'diagnostics/timing_0000.json').read_text())
    switches = json.loads((folder / 'diagnostics/switches_0000.json').read_text())
    expected = manifest['expected_output']
    videos = list((folder / 'videos').glob('*.mp4'))
    assert len(videos) == 1
    video = videos[0]
    ffprobe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_read_frames',
        '-of', 'json', str(video)], text=True))['streams'][0]
    assert int(ffprobe['nb_read_frames']) == expected['frames'] == timing['export_frames']
    assert ffprobe['avg_frame_rate'] == ffprobe['r_frame_rate'] == '16/1'
    assert float(ffprobe['duration']) == expected['seconds']
    assert [s['start_latent_frame'] for s in switches] == expected['switch_latent_frames']
    assert [s['export_prefix_frames'] for s in switches] == expected['switch_export_frames']
    states = []
    for s in switches:
        row = {'start_latent_frame': s['start_latent_frame'],
               'start_export_frame': s['export_prefix_frames'],
               'wall_seconds_including_audit': s['wall_seconds_including_optional_audit'],
               'history_preserved': s['history_content_audited'] and s['history_sha256_before'] == s['history_sha256_after'],
               'cursors_preserved': s['cache_positions_before'] == s['cache_positions_after'],
               'pool_counts_preserved': s['retrieval_pool_counts_before'] == s['retrieval_pool_counts_after'],
               'cross_attention_initialized': all(s['cross_attention_initialized']),
               'changed_resident_k_layers': sum(a != b for a,b in zip(s['resident_k_sha256_before'], s['resident_k_sha256_after'])),
               'changed_resident_v_layers': sum(a != b for a,b in zip(s['resident_v_sha256_before'], s['resident_v_sha256_after']))}
        assert all(row[k] for k in ['history_preserved', 'cursors_preserved', 'pool_counts_preserved', 'cross_attention_initialized'])
        assert row['changed_resident_k_layers'] == row['changed_resident_v_layers'] == 30
        states.append(row)
    latent_path = folder / 'diagnostics/latents_0000.pt'
    latent = torch.load(latent_path, weights_only=True, map_location='cpu')
    assert latent.shape[1] == timing['latent_frames'] and torch.isfinite(latent).all()
    memory_path = next((folder / 'videos').glob('*_memory_log.json'))
    memory = json.loads(memory_path.read_text())
    assert memory and all(r['compression_method'] == 'ae' for r in memory)
    # An inspectable sample at the middle of each authored segment; no labels.
    cap = cv2.VideoCapture(str(video))
    samples = folder / 'sample_frames'
    samples.mkdir(exist_ok=True)
    try:
        for segment in timing['segments']:
            frame = segment['start_export_frame'] + segment['export_frames'] // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            assert ok
            cv2.imwrite(str(samples / f'frame_{frame:04d}.png'), image)
    finally:
        cap.release()
    return {'phase': 'official_exact_60s_engineering', 'engineering_checks_passed': True,
            'held_out': False, 'correction_enabled': False, 'quality_claim': False,
            'source_benchmark_index': manifest['source_benchmark_index'],
            'ffprobe': ffprobe, 'latent_shape': list(latent.shape), 'latent_dtype': str(latent.dtype),
            'latent_sha256': sha(latent_path), 'video_sha256': sha(video),
            'timing': timing, 'switch_checks': states, 'retrieval_query_count': len(memory),
            'last_retrieval_query_frame': memory[-1]['query_frame'],
            'wall_seconds': status['wall_seconds'],
            'allocated_gpu_seconds': status['wall_seconds'],
            'gpu_cost_definition': 'One reserved GPU times wall time, not active kernel time',
            'total_prompt_refresh_seconds_including_audit': sum(s['wall_seconds_including_audit'] for s in states),
            'input_sha256': {str(p.relative_to(folder)): sha(p) for p in [
                folder / 'manifest.json', folder / 'status.json', folder / 'config.yaml',
                folder / 'diagnostics/timing_0000.json', folder / 'diagnostics/switches_0000.json', memory_path]},
            'limits': ['One exposed development script; not held-out benchmark performance.',
                       'Export time and state consistency do not establish prompt compliance or visual quality.',
                       'Timing adaptation changes block boundaries, VAE caching and the initial frame crop.',
                       'All paired formal baselines must share the same frozen timing adaptation.',
                       'Audit hashes of the growing history add cost; recorded wall time includes them.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    result = audit(args.folder.resolve())
    (args.folder / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['engineering_checks_passed', 'ffprobe', 'wall_seconds', 'retrieval_query_count', 'total_prompt_refresh_seconds_including_audit']}))
