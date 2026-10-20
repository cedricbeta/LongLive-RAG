#!/usr/bin/env python3
"""Prepare or execute one bounded, frozen official VBench-Long development run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REVISION = 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
DIMENSIONS = ['subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(args):
    folder, checkout, provision, cache = (p.resolve() for p in
        (args.folder, args.checkout, args.provision, args.cache))
    if args.gpu not in (0, 1, 2):
        raise ValueError('GPU 3 is reserved')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == REVISION
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip()
    preflight = read(provision / 'model_cpu_preflight.json')
    assert preflight['all_passed'] and preflight['actual_video_quality_evaluations'] == 0
    audit = read(args.source_audit)
    assert audit['engineering_checks_passed'] and audit['single_prompt_unchanged']
    assert sha(args.source_video) == audit['video_sha256']
    assert audit['ffprobe']['nb_read_frames'] == '1920'
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'input').mkdir()
    # A stable identifier avoids upstream filename parsing on underscores/long prompts.
    video = folder / 'input/moviegen000-0.mp4'
    shutil.copyfile(args.source_video, video)
    shutil.copyfile(args.source_audit, folder / 'source_video_audit.json')
    shutil.copyfile(Path(__file__), folder / 'supervisor.py')
    checks = {str(video): sha(video), str(folder / 'source_video_audit.json'): sha(folder / 'source_video_audit.json'),
              str(folder / 'supervisor.py'): sha(Path(__file__))}
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=checkout).decode().split('\0')
    for path in paths:
        if path and (checkout / path).is_file():
            checks[str(checkout / path)] = sha(checkout / path)
    assets = read(provision / 'model_assets_manifest.json')
    for row in assets['assets']:
        path = cache / row['cache_relative_path']
        assert sha(path) == row['sha256'], path
        checks[str(path)] = row['sha256']
    old_cache = Path(assets['cache_root'])
    for row in read(provision / 'cached_hub_source_audit.json'):
        assert not row['missing'] and not row['different']
        location = cache / Path(row['cache_path']).relative_to(old_cache)
        for name, digest in row['files_sha256'].items():
            assert sha(location / name) == digest, name
            checks[str(location / name)] = digest
    packages = ['torch', 'torchvision', 'transformers', 'timm', 'pyiqa', 'dreamsim', 'peft',
                'open-clip-torch', 'scenedetect', 'moviepy', 'decord', 'av', 'numpy', 'Pillow']
    command = [str(args.python.absolute()), '-u', str(checkout / 'vbench2_beta_long/eval_long.py'),
        '--videos_path', str(folder / 'input'), '--output_path', str(folder / 'outputs'),
        '--dimension', *DIMENSIONS, '--mode', 'long_custom_input', '--dev_flag',
        '--num_of_samples_per_prompt', '1', '--imaging_quality_preprocessing_mode', 'longer']
    plan = {'protocol': 'official_vbench_long_single_video_development_v1', 'frozen_unix': time.time(),
        'checkout': str(checkout), 'revision': REVISION, 'command': command,
        'environment': {'CUDA_VISIBLE_DEVICES': str(args.gpu), 'PYTHONPATH': str(checkout),
                        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'},
        'gpu': args.gpu, 'timeout_seconds': args.timeout, 'dimensions': DIMENSIONS,
        'video_sha256': audit['video_sha256'], 'source_video': str(args.source_video.absolute()),
        'prompt': audit['timing']['schedule']['0'][0], 'held_out': False,
        'source_and_asset_sha256': checks,
        'package_versions': {n: importlib.metadata.version(n) for n in packages},
        'expected_preprocessing': {'source_fps': 16, 'clip_seconds': 2, 'clip_frames': 32,
                                   'clip_count': 60, 'semantic_splitting': False,
                                   'note': 'Official splitter overwrites its fps=8 argument with source fps.'},
        'aggregation': 'Original six per-dimension outputs. No invented overall score or confidence interval from one video. All clip records preserved; clips are not independent videos.',
        'failure_policy': 'One attempt; timeout or model error retained. No automatic retry or alternate checkpoint. No result is claimed until terminal outputs and actual inputs are audited.',
        'quality_scope': 'Visual metrics of one previously exposed baseline. No correction benefit, identity accuracy, instruction compliance or human-evaluation claim.',
        'cost_scope': 'Supervisor includes input/source/asset checks, original CLI startup, preprocessing, model loading and scoring. Preparation and earlier generation reported separately.'}
    write(folder / 'manifest.json', {'gpu': args.gpu, 'phase': plan['protocol'], 'video_sha256': audit['video_sha256']})
    write(folder / 'plan.json', plan)
    print(json.dumps({'folder': str(folder), 'dimensions': DIMENSIONS, 'checks': len(checks)}))


def admission(gpu):
    info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    uuid, free, used, utilization = info[0], *map(int, info[1:])
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    pids = [int(line.split(', ')[1]) for line in processes.splitlines() if line.split(', ')[0] == uuid]
    if pids or free < 75000 or used > 1000 or utilization > 5:
        raise RuntimeError(f'GPU {gpu} is occupied: free={free}, used={used}, utilization={utilization}, PIDs={pids}')
    return {'gpu_uuid': uuid, 'admission_free_mib': free, 'admission_compute_pids': pids}


def execute(folder):
    plan = read(folder / 'plan.json')
    if (folder / 'status.json').exists():
        raise FileExistsError('A recorded attempt is never restarted')
    assert plan['gpu'] in (0, 1, 2)
    started = time.monotonic()
    status = {'status': 'preflight', 'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'gpu': plan['gpu']}
    write(folder / 'status.json', status)
    child = None
    try:
        status.update(admission(plan['gpu']))
        for name, expected in plan['source_and_asset_sha256'].items():
            assert sha(Path(name)) == expected, name
        for name, version in plan['package_versions'].items():
            assert importlib.metadata.version(name) == version, name
        status.update(admission(plan['gpu']))
        with (folder / 'evaluation.log.txt').open('x') as log:
            child = subprocess.Popen(plan['command'], cwd=plan['checkout'],
                env={**os.environ, **plan['environment']}, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            status.update(status='running', pid=child.pid)
            write(folder / 'status.json', status)
            code = child.wait(timeout=max(1, plan['timeout_seconds'] - (time.monotonic() - started)))
        status.update(status='completed' if code == 0 else 'failed', returncode=code)
        assert sha(folder / 'input/moviegen000-0.mp4') == plan['video_sha256']
    except subprocess.TimeoutExpired:
        status.update(status='timeout', error='Frozen evaluation wall-time limit reached')
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        status.update(wall_seconds=time.monotonic() - started, finished_unix=time.time())
        write(folder / 'status.json', status)
    print(json.dumps(status))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--checkout', type=Path)
    parser.add_argument('--provision', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--source-video', type=Path)
    parser.add_argument('--source-audit', type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    else:
        execute(args.folder.resolve())
