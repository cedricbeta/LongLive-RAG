#!/usr/bin/env python3
"""Check native baseline inputs/duration/cursors and expose all frozen views."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
from PIL import Image, ImageDraw
import torch
import yaml


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def audit(folder):
    began = time.monotonic()
    torch.set_num_threads(4)
    read = lambda p: json.loads(p.read_text())
    assert not (folder / 'audit.json').exists()
    status, manifest, plan = [read(folder / n) for n in ('status.json', 'manifest.json', 'plan.json')]
    assert status['status'] == 'completed'
    assert sha(folder / 'plan.json') == manifest['source_case_plan_sha256']
    assert sha(folder / 'config.yaml') == manifest['config_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert sha(Path(manifest['cwd']) / name) == digest
    case = next(c for c in plan['cases'] if c['index'] == manifest['source_benchmark_index'])
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    assert config['timed_segments'] == [{'seconds': 10, 'prompts': [p]} for p in case['prompts']]
    assert config['audit_prompt_switch'] is False and 'repair_request' not in config
    timing = read(folder / 'diagnostics/timing_0000.json')
    assert timing['export_frames'] == 960 and timing['latent_frames'] == 241
    switches = read(folder / 'diagnostics/switches_0000.json')
    assert len(switches) == 5
    for shot, row in enumerate(switches, 1):
        assert row['start_latent_frame'] == 40 * shot + 1 and row['export_prefix_frames'] == 160 * shot
        assert row['cache_positions_before'] == row['cache_positions_after']
        assert row['retrieval_pool_counts_before'] == row['retrieval_pool_counts_after']
        assert all(row['cross_attention_initialized']) and not row['history_content_audited']
        for field in ('history_sha256_before', 'history_sha256_after', 'resident_k_sha256_before',
                      'resident_k_sha256_after', 'resident_v_sha256_before', 'resident_v_sha256_after'):
            assert row[field] is None
    latent_path = folder / 'diagnostics/latents_0000.pt'
    latents = torch.load(latent_path, weights_only=True, map_location='cpu')
    assert tuple(latents.shape) == (1, 241, 16, 60, 104) and latents.dtype == torch.bfloat16
    assert torch.isfinite(latents).all()
    video, = list((folder / 'videos').glob('*.mp4'))
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened() and capture.get(cv2.CAP_PROP_FPS) == 16
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 960
    frames = folder / 'sample_frames'
    frames.mkdir()
    rows, index = [], 0
    montage = Image.new('RGB', (3 * 416, 6 * 267), 'white')
    draw = ImageDraw.Draw(montage)
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape == (480, 832, 3)
            if index in plan['review_frames']:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                dest = frames / f'frame_{index:04d}.png'
                im = Image.fromarray(rgb)
                im.save(dest)
                position = len(rows)
                x, y = position % 3 * 416, position // 3 * 267
                montage.paste(im.resize((416, 240)), (x, y + 27))
                draw.text((x + 8, y + 7), f'{case["id"]} baseline | shot {position // 3 + 1} | {index / 16:g}s', fill='black')
                rows.append({'frame': index, 'seconds': index / 16, 'path': str(dest.relative_to(folder)), 'sha256': sha(dest)})
            index += 1
    finally:
        capture.release()
    assert index == 960 and len(rows) == 18
    montage.save(frames / 'all_fixed_views.png')
    memory = read(next((folder / 'videos').glob('*memory_log.json')))
    assert len(memory) == 78 and all(r['compression_method'] == 'ae' for r in memory)
    record = {'phase': 'exposed_native_baseline_before_bounded_target_probe',
        'engineering_checks_passed': True, 'history_content_hashes_audited': False,
        'resident_content_hashes_audited': False, 'prompt_switch_cursor_and_pool_lengths_preserved': True,
        'source_benchmark_index': case['index'], 'frames_decoded': index, 'fps': 16, 'seconds': 60,
        'latent_shape': list(latents.shape), 'latent_dtype': str(latents.dtype),
        'latent_sha256': sha(latent_path), 'video_sha256': sha(video), 'video_path': str(video.relative_to(folder)),
        'review_frames': rows, 'retrieval_query_count': len(memory), 'timing': timing,
        'generation_wall_seconds': status['wall_seconds'], 'audit_wall_seconds': time.monotonic() - began,
        'source_sha256': sha(Path(__file__)), 'manifest_sha256': sha(folder / 'manifest.json'),
        'held_out': False, 'correction_enabled': False, 'quality_claim': False, 'human_labels': 0,
        'limits': ['Native refresh/cursor guards executed; history and resident tensor byte hashes deliberately not measured.',
                   'This is the original baseline. A visual defect and a new frozen automatic mask are required before any repair.',
                   'Fixed stills and numeric checks do not establish identity/action correctness.']}
    (folder / 'audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('source_benchmark_index', 'frames_decoded', 'generation_wall_seconds', 'audit_wall_seconds')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    audit(p.parse_args().folder.resolve())
