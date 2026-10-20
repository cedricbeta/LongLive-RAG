#!/usr/bin/env python3
"""Verify published duration-pilot files and recompute its measured telemetry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import cv2


def verify(root, name):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text())
    folder = root / 'results' / name
    export = read(folder / 'export_manifest.json')
    for path, expected in export['files_sha256'].items():
        assert sha(root / path) == expected, path
    plan = read(folder / 'plan.json')
    for path, expected in plan['source_sha256'].items():
        canonical = root / path if path.startswith('adapters/') else root / 'implementation' / path
        assert sha(canonical) == sha(folder / 'source' / path) == expected, path
    for path, expected in plan['runtime_files_sha256'].items():
        assert sha(folder / path) == expected, path
    assert sha(folder / 'queue_source.py') == plan['queue_source_sha256']
    status, audit = read(folder / 'status.json'), read(folder / 'audit.json')
    assert status['status'] == 'completed' and status['returncode'] == 0
    assert audit['engineering_checks_passed'] and audit['single_prompt_unchanged']
    assert not audit['switch_checks'] and len(audit['timing']['schedule']) == 1
    assert audit['latent_shape'] == [1, 481, 16, 60, 104]
    for path, expected in audit['input_sha256'].items():
        assert sha(folder / path) == expected, path
    assert sha(root / 'implementation/scripts/audit_native_single_prompt.py') == audit['audit_source_sha256']
    video, = (root / 'artifacts' / name / 'videos').glob('*.mp4')
    assert sha(video) == audit['video_sha256']
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_read_frames',
        '-of', 'json', str(video)], text=True)
    assert json.loads(probe)['streams'][0] == audit['ffprobe']
    assert audit['ffprobe']['nb_read_frames'] == '1920' and audit['ffprobe']['duration'] == '120.000000'
    capture = cv2.VideoCapture(str(video))
    try:
        for row in audit['sample_frames']:
            assert sha(folder / row['path']) == row['sha256']
            capture.set(cv2.CAP_PROP_POS_FRAMES, row['frame'])
            ok, frame = capture.read()
            assert ok
            ok, encoded = cv2.imencode('.png', frame)
            assert ok and hashlib.sha256(encoded.tobytes()).hexdigest() == row['sha256']
    finally:
        capture.release()
    telemetry = root / 'results' / (name + '_telemetry')
    tplan, measured = read(telemetry / 'plan.json'), read(telemetry / 'summary.json')
    assert sha(telemetry / 'monitor_source.py') == tplan['source_sha256']
    assert sha(root / 'implementation/scripts/monitor_native_gpu.py') == tplan['source_sha256']
    assert sha(folder / 'manifest.json') == tplan['target_manifest_sha256']
    assert sha(folder / 'plan.json') == tplan['target_plan_sha256']
    assert sha(telemetry / 'samples.jsonl') == measured['samples_sha256']
    assert sha(folder / 'status.json') == sha(telemetry / 'target_terminal_status.json') == measured['target_status_sha256']
    assert read(telemetry / 'device.json')['uuid'] == status['gpu_uuid']
    module_path = root / 'implementation/scripts/monitor_native_gpu.py'
    spec = importlib.util.spec_from_file_location('native_telemetry', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reconstructed = module.summarize([json.loads(line) for line in
        (telemetry / 'samples.jsonl').read_text().splitlines()], status)
    reconstructed.update(samples_sha256=sha(telemetry / 'samples.jsonl'),
                         target_status_sha256=sha(folder / 'status.json'))
    assert reconstructed == measured
    return {'name': name, 'export_files_verified': len(export['files_sha256']),
        'frozen_sources_verified': len(plan['source_sha256']),
        'actual_decoded_frames': 1920, 'sample_frames_reproduced': len(audit['sample_frames']),
        'telemetry_summary_recomputed_exactly': True, 'raw_latents_exported': False,
        'new_model_calls': 0, 'new_generated_videos': 0,
        'scope': 'File, timing, frame and telemetry audit; finite-latent check is bound to the recorded runtime audit, not independently repeated without raw latents.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='official_moviegen_000_120s_v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root.resolve(), args.name)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
