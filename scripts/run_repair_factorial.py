#!/usr/bin/env python3
"""Frozen development-only 2x2 mask/prompt experiment; no candidate selection.

Prepare all four cells before inspecting outputs. Each cell replays the entire
archived scene with identical generator code/seed, changing only mask and prompt.
No failed cell is silently rerun. GPU wall allocation is a cost upper bound,
not a claim to measured kernel time or matched total pipeline compute.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def verify(output, manifest):
    for rel, digest in manifest['files'].items():
        if sha(output / rel) != digest:
            raise RuntimeError(f'Frozen input changed: {rel}')


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'videos/feedback_repair_20260914'
    recent = ROOT / 'videos/feedback_repair_v2_20260916'
    # Read immutable archived source, never the current defaults.
    source = prior / 'source_feedback'
    shutil.copytree(source, out / 'source', symlinks=True)
    driver = out / 'frozen_driver.py'
    shutil.copy2(__file__, driver)
    run = prior / 'runs/feedback' / args.scene / args.scene
    base = yaml.safe_load((run / 'configs/edit_shot00_attempt1.yaml').read_text())
    if len(base['inference']['edit_renoise_plans']) != 1:
        raise ValueError('Expected exactly one first-shot archived plan')
    actions = next(r['actions'] for r in json.loads((recent / 'analysis/first_shot_actions.json').read_text())
                   if r['scene'] == args.scene)
    for version in ('v1', 'v2'):
        action = actions[version]
        if action['sigma'] != 1.0 or action['seed'] != 4243:
            raise ValueError('Unexpected archived sigma/seed')
        shutil.copy2(action['mask_path'], out / f'{version}_mask.pt')
    shutil.copytree(run / 'data', out / 'data')
    cases = []
    for mask in ('v1', 'v2'):
        for prompt in ('v1', 'v2'):
            cell = f'mask_{mask}_prompt_{prompt}'
            config = deepcopy(base)
            config['data']['data_path'] = str(out / 'data')
            config['output_folder'] = config['inference']['output_folder'] = str(out / cell / 'renders')
            config['inference']['filename_prefix'] = cell
            plan = config['inference']['edit_renoise_plans'][0]
            plan.update(mask_path=str(out / f'{mask}_mask.pt'), prompt=actions[prompt]['prompt'],
                        entry_sigma=1.0, noise_seed=4243)
            path = out / 'configs' / f'{cell}.yaml'
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            cases.append({'id': cell, 'mask': mask, 'prompt': prompt, 'config': str(path.relative_to(out))})
    # Source files are small; model directories remain shared prerequisites.
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
             if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}
    manifest = {'schema_version': 1, 'phase': 'development_diagnostic', 'quality_claim': False,
                'scene': args.scene, 'created_unix': time.time(), 'cells': cases, 'files': files,
                'source_origin': str(source.relative_to(ROOT)), 'sigma': 1.0, 'noise_seed': 4243,
                'runtime_python': args.python, 'timeout_seconds': args.timeout,
                'evaluation': 'All cells retained; frozen historical judge and explicit visual review required.',
                'limitations': ['One exposed script and one seed; no held-out effectiveness inference.',
                                'Full-scene 5B replay, not an official 1.3B baseline or streaming experiment.',
                                'GPU allocated wall seconds include initialization; kernel time is not measured.',
                                'Weight hashes and evaluation costs must be added before a reproducibility claim.']}
    write(out / 'manifest.json', manifest)
    verify(out, manifest)
    print(json.dumps({'prepared': str(out), 'cells': len(cases), 'frozen_files': len(files)}), flush=True)


def run(args):
    out = args.output
    manifest = json.loads((out / 'manifest.json').read_text())
    verify(out, manifest)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError('GPU lanes must be distinct')

    def lane(gpu, cells):
        # Use the same lock namespace as earlier render drivers.
        with Path(f'/tmp/longlive_feedback_render_gpu{gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for cell in cells:
                directory = out / cell['id']
                directory.mkdir(exist_ok=True)
                status_path = directory / 'status.json'
                if status_path.exists():
                    raise RuntimeError(f'Cell already attempted; inspect its process/result: {cell["id"]}')
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free < 78000:
                    raise RuntimeError(f'GPU {gpu} is occupied ({free} MiB free)')
                verify(out, manifest)
                command = [manifest['runtime_python'], 'inference.py', '--config_path', str(out / cell['config'])]
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
                       'PYTORCH_ALLOC_CONF': 'expandable_segments:True'}
                started = time.monotonic()
                status = {**cell, 'status': 'running', 'gpu': gpu, 'started_unix': time.time(),
                          'command': command, 'source': 'source', 'automatic_retries': 0}
                with (directory / 'render.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=out / 'source', env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    status['pid'] = child.pid
                    write(status_path, status)
                    print(f'{cell["id"]}: started PID {child.pid} on GPU {gpu}', flush=True)
                    try:
                        code = child.wait(timeout=manifest['timeout_seconds'])
                        status['status'] = 'completed' if code == 0 else 'execution_failed'
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        code = child.returncode
                        status['status'] = 'timeout'
                status.update(returncode=code, wall_seconds=time.monotonic() - started)
                status['gpu_allocated_seconds'] = status['wall_seconds']
                try:
                    verify(out, manifest)
                except Exception as error:
                    status.update(status='invalid_inputs', error=str(error))
                status['outputs'] = [{'path': str(p.relative_to(out)), 'sha256': sha(p)}
                                     for p in sorted(directory.rglob('*.mp4'))]
                if status['status'] == 'completed' and not status['outputs']:
                    status.update(status='execution_failed', error='No MP4 output')
                write(status_path, status)
                print(f'{cell["id"]}: {status["status"]} ({status["wall_seconds"]:.1f}s)', flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(lane, gpu, manifest['cells'][i::len(args.gpus)])
                   for i, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', default='ns310_at_the_flower_shop')
    parser.add_argument('--python', default='/home/chendong/miniconda3/bin/python')
    parser.add_argument('--timeout', type=int, default=2400)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    args.output = args.output.resolve()
    {'prepare': prepare, 'run': run}[args.stage](args)


if __name__ == '__main__':
    main()
