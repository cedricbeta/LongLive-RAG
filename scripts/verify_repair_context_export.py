#!/usr/bin/env python3
"""Verify this bounded context experiment's published evidence without models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
from PIL import Image
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def verify(root):
    read = lambda p: json.loads(p.read_text())
    folder = root / 'results/native_iris_repair_context_v1'
    art = root / 'artifacts/native_iris_repair_context_v1'
    export, plan, summary = [read(folder / n) for n in ('export_manifest.json', 'plan.json', 'summary.json')]
    for name, digest in export['files_sha256'].items():
        assert sha(folder / name) == digest, name
    for name, digest in export['artifacts_sha256'].items():
        assert sha(art / name) == digest, name
    assert sha(root / 'reproduction/native_expression_control/automatic_mask.tensor') == plan['mask_file_sha256']
    for name, digest in plan['native_source_sha256'].items():
        assert sha(folder / 'native_source' / name) == digest, name
    for name, digest in plan['adapter_source_sha256'].items():
        assert sha(folder / 'source' / name) == digest, name
    contracts = root / 'results/repair_context_contracts_v2'
    unit = read(contracts / 'unit_tests.json')
    assert unit['returncode'] == 0 and unit['passed'] == 16
    assert sha(contracts / 'unit_tests.txt') == unit['transcript_sha256']
    for name, digest in unit['source_sha256'].items():
        assert sha(root / name) == digest, name
    kernel = read(contracts / 'kernel.json')
    assert len(kernel['cases']) == 4 and kernel['new_video_generations'] == 0
    assert sha(root / 'implementation/scripts/probe_repair_context_kernel.py') == kernel['source_sha256']
    assert sha(root / 'adapters/official_longlive_rag/repair_context.py') == kernel['context_source_sha256']
    assert read(root / 'results/repair_context_contracts_v1/failure.json')['status'] == 'execution_failed'
    for case, intercepts in [('off', 0), ('on', 1680)]:
        path = folder / case
        manifest, audit, status = [read(path / n) for n in ('manifest.json', 'context_audit.json', 'status.json')]
        assert sha(path / 'config.yaml') == manifest['config_sha256']
        assert hashlib.sha256(json.dumps(manifest['action'], sort_keys=True).encode()).hexdigest() == manifest['action_fingerprint']
        assert status['status'] == 'completed' and audit['private_forwards'] == 56
        assert audit['intercepted_self_attention_calls'] == intercepts and not audit['candidate_retained']
        assert audit['context_only_private_calls'] and audit['context_callable_restored_each_step']
        if case == 'off':
            assert audit['off_conformance_passed'] and audit['equal_reference_latent_indices'] == list(range(241))
        telemetry = root / f'results/native_iris_repair_context_{case}_v1_telemetry'
        energy = read(telemetry / 'summary.json')
        assert sha(telemetry / 'samples.jsonl') == energy['samples_sha256']
        assert sha(path / 'status.json') == energy['target_status_sha256']
        row = next(r for r in summary['cases'] if r['case'] == case)
        assert row['wall_seconds'] == status['wall_seconds'] and row['energy_bounds'] == energy['device_energy_bounds']
    assert summary['two_render_seconds'] == sum(r['wall_seconds'] for r in summary['cases'])
    assert not summary['candidate_retained'] and summary['new_vlm_calls'] == summary['human_labels'] == 0
    observation = read(folder / 'developer_observations.json')
    assert not observation['human_validated'] and observation['actual_viewed_frames'] == plan['review_frames']
    review = read(art / 'review.json')
    assert len(review['images']) == 66 and review['review_frames'] == plan['review_frames']
    for row in review['images']:
        assert sha(art / row['path']) == row['sha256']
        pixels = np.asarray(Image.open(art / row['path']).convert('RGB'))
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row['rgb_sha256']
    assert sha(art / 'current_block_candidate.mp4') == review['source_videos']['current_block']['sha256']
    capture = cv2.VideoCapture(str(art / 'current_block_candidate.mp4'))
    assert capture.get(cv2.CAP_PROP_FPS) == 16
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == review['full_video_frames'][index]['rgb_sha256']['current_block']
            index += 1
    finally:
        capture.release()
    assert index == 960
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(art / 'three_way_comparison.mp4')], text=True))['streams'][0]
    assert int(metadata['nb_read_frames']) == 960 and metadata['r_frame_rate'] == '16/1'
    assert float(metadata['duration']) == 60 and [metadata['width'], metadata['height']] == [1248, 270]
    return {'published_pngs_checked': 66, 'actual_on_video_frames_redecoded_and_hashed': index,
        'comparison_frames': 960, 'source_and_action_bindings_match': True, 'both_fail_and_pass_kernel_probes_retained': True,
        'raw_latents_or_weights_recomputed': False, 'new_model_calls': 0, 'quality_judgment_performed': False,
        'scope': 'Published artifacts/source/record consistency; runtime native audits checked excluded raw tensors. Semantic observations remain developer inspection.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
