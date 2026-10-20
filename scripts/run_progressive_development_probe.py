#!/usr/bin/env python3
"""Freeze and run a bounded, exposed Iris engineering probe. No automatic retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_feedback_repair_comparison import snapshot_source, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
SCENE = 'ns090_iris_stands_at_the'


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = snapshot_source(output / 'source')
    prior = ROOT / 'videos/feedback_repair_20260914'
    shutil.copytree(prior / 'frozen/data' / SCENE, output / 'data' / SCENE)
    shutil.copy2(prior / 'frozen/entity_specs.json', output / 'entity_specs.json')
    shutil.copy2(prior / 'runs/feedback' / SCENE / SCENE / 'configs/baseline.yaml', output / 'base.yaml')
    files = {f'source/{p}': h for p, h in hashes.items()}
    for path in [output / 'base.yaml', output / 'entity_specs.json', *sorted((output / 'data').rglob('*'))]:
        if path.is_file():
            files[str(path.relative_to(output))] = digest(path)
    write_json(output / 'manifest.json', {
        'phase': 'development_engineering_probe', 'scene': SCENE, 'created_unix': time.time(),
        'quality_claim': False, 'held_out': False, 'independent_judge': False,
        'files': files, 'runtime_python': str(ROOT / 'venv/bin/python'),
        'renderer_python': '/home/chendong/miniconda3/bin/python',
        'max_render_attempts': 2, 'process_shots_0based': [0], 'lookahead_max_shots': 2,
        'repair_wall_budget_seconds': 2400, 'case_timeout_seconds': 3600,
        'baseline_policy': 'Generate anew and retain raw latents; archived video equality is separately checked.',
        'limitations': ['Exposed 5B native LongLive example, not official RAG or IAMFlow.',
                       'Fixed part recipe, not the proposed adaptive action policy.',
                       'Offline full replay; exact KV snapshots and streaming are not implemented.',
                       'Three sampled frames cannot establish whole-shot identity/action correctness.']})


def run(output, base_url, render_gpu, mask_gpu):
    manifest = json.loads((output / 'manifest.json').read_text())
    for rel, sha in manifest['files'].items():
        if digest(output / rel) != sha:
            raise RuntimeError(f'Frozen file changed: {rel}')
    if (output / 'status.json').exists():
        raise RuntimeError('Probe already attempted; inspect it instead of silently rerunning')
    locks = []
    for gpu in sorted({render_gpu, mask_gpu}):
        handle = open(f'/tmp/longlive_feedback_render_gpu{gpu}.lock', 'a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 78000:
            raise RuntimeError(f'GPU {gpu} is occupied: {free} MiB free')
    source = output / 'source'
    command = [manifest['runtime_python'], '-u', str(source / 'scripts/run_agent_edit_demo.py'),
        '--base_url', base_url, '--render_gpu', str(render_gpu), '--clipseg_device', f'cuda:{mask_gpu}',
        '--base_config', str(output / 'base.yaml'), '--data_root', str(output / 'data'),
        '--entity_specs', str(output / 'entity_specs.json'), '--output_root', str(output / 'run'),
        '--scenes', SCENE, '--allow_missing_global', '--judge', 'entity',
        '--repair_policy', 'progressive', '--comparer', 'attributes', '--attribute_masks',
        '--role_localizer', 'indexed', '--judge_chunks', 'sparse', '--place_absent', 'never',
        '--entry_sigma', '1.0', '--max_edits_per_shot', str(manifest['max_render_attempts']),
        '--commit_lookahead_shots', str(manifest['lookahead_max_shots']), '--process_shots', '1',
        '--max_repair_seconds', str(manifest['repair_wall_budget_seconds'])]
    status = {'status': 'running', 'command': command, 'started_unix': time.time(),
              'render_gpu': render_gpu, 'mask_gpu': mask_gpu, 'automatic_retries': 0}
    started = time.monotonic()
    with (output / 'run.log').open('x') as log:
        child = subprocess.Popen(command, cwd=source, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status['pid'] = child.pid
        write_json(output / 'status.json', status)
        try:
            code = child.wait(timeout=manifest['case_timeout_seconds'])
            status.update(returncode=code, status='completed' if code == 0 else 'execution_failed')
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            status.update(status='timeout', returncode=child.returncode)
    status['wall_seconds'] = time.monotonic() - started
    status['reserved_gpu_wall_seconds'] = status['wall_seconds'] * len(locks)
    status['cost_note'] = 'Reservation wall time includes idle intervals; shared judge cost is in VLM transcripts, not included here.'
    write_json(output / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base_url')
    parser.add_argument('--render_gpu', type=int, default=1)
    parser.add_argument('--mask_gpu', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output.resolve())
    else:
        if not args.base_url:
            parser.error('--base_url is required for run')
        run(args.output.resolve(), args.base_url, args.render_gpu, args.mask_gpu)
