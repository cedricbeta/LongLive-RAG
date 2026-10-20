#!/usr/bin/env python3
"""Freeze and run target-blind, different-family observations of exposed videos."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

PROMPT = ('Describe only what is visibly present in this image. Count the visible people. '
          'Describe their clothing types, colors, sleeves, layers and visible fastenings or straps, '
          'and then their hair color and texture. State details that cannot be determined. '
          'Do not infer identities, intentions, earlier images or hidden details. '
          'Return only a JSON object with exactly four fields: people_count (a nonnegative integer), '
          'clothing (a string), hair (a string), uncertainty (a string). '
          'When no person is visible, use people_count 0 and say that clothing and hair are not visible. '
          'Keep the descriptions together under 120 words.')
PACKAGES = ['torch', 'torchvision', 'transformers', 'accelerate', 'Pillow', 'numpy', 'opencv-python']


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(args):
    import cv2
    from PIL import Image
    root, folder = args.root.resolve(), args.folder.resolve()
    assert args.gpu in (0, 1, 2) and args.timeout > 0
    model = read(args.model_manifest)
    assert model['revision'] == '784410650d12be9bc086118fdefa32d2c3bced86'
    for name, row in model['files'].items():
        assert sha(args.model_path / name) == row['sha256'], name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'inputs').mkdir()
    shutil.copy2(args.model_manifest, folder / 'model_manifest.json')
    shutil.copy2(Path(__file__), folder / 'runner.py')
    (folder / 'model_source').mkdir()
    for name in model['files']:
        if name.endswith('.py') or name == 'chat_template.jinja':
            shutil.copy2(args.model_path / name, folder / 'model_source' / name)
    cases, videos = [], []
    indices = [320, 352, 400, 448, 479, 672, 720, 768]
    video_paths = {
        'original': 'artifacts/official_iris_60s_baseline_prepared/videos/rank0-0-0_lora.mp4',
        'shared_first': 'artifacts/native_iris_names_v6_window/candidate.mp4',
        'feedback_second': 'artifacts/native_iris_mixed_evidence_v1/candidate.mp4',
        'fixed_second': 'artifacts/native_iris_fixed_noise_v1/candidate.mp4'}
    for branch, relative in video_paths.items():
        path = root / relative
        videos.append({'branch': branch, 'repository_path': relative, 'sha256': sha(path)})
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 16
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 960
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok and frame.shape == (480, 832, 3)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            path = folder / 'inputs' / f'{branch}_{index}.png'
            image.save(path)
            cases.append({'branch': branch, 'frame': index, 'seconds': index / 16,
                          'image': str(path.relative_to(folder)), 'image_sha256': sha(path),
                          'scope': 'current' if index < 480 else 'future', 'control': None})
        cap.release()
    path = folder / 'inputs/gray.png'
    Image.new('RGB', (832, 480), (127, 127, 127)).save(path)
    cases.append({'branch': None, 'frame': None, 'seconds': None, 'image': 'inputs/gray.png',
                  'image_sha256': sha(path), 'scope': 'synthetic', 'control': 'no_person'})
    for branch in ('original', 'fixed_second'):
        base = next(r for r in cases if r['branch'] == branch and r['frame'] == 400)
        cases.append({**base, 'control': 'same_image_repeat'})
    random.Random(17344).shuffle(cases)
    for i, row in enumerate(cases):
        row['id'] = f'image_{i:03d}'
    plan = {'protocol': 'native_iris_target_blind_observations_v1', 'frozen_unix': time.time(),
        'hypothesis': 'Unconditioned observations may expose garment-category or temporal changes obscured by target-conditioned pass/fail labels. This is a development diagnostic, not calibrated ground truth.',
        'model': model['repo'], 'model_revision': model['revision'],
        'model_path': str(args.model_path.absolute()), 'python': str(args.python.absolute()),
        'package_versions': {n: importlib.metadata.version(n) for n in PACKAGES},
        'prompt': PROMPT, 'cases': cases, 'videos': videos, 'gpu': args.gpu,
        'timeout_seconds': args.timeout, 'max_calls': 35,
        'generation': {'max_new_tokens': 512, 'do_sample': False}, 'seed_per_call': 0,
        'dtype': 'bfloat16', 'attention': 'sdpa', 'grammar_constrained_decoding': False,
        'source_sha256': sha(folder / 'runner.py'),
        'model_manifest_sha256': sha(folder / 'model_manifest.json'),
        'held_out': False, 'actual_human_labels': 0, 'integrated_into_acceptance': False,
        'failure_policy': 'One run; zero parsing/model retries. All replies/errors retained. Timeout terminates only owned child group. No candidate selection or historical verdict change.',
        'scope': 'Four archived video branches from one exposed Iris script, eight fixed full frames each, one uniform-gray control and two exact-image repetitions. Model receives only the common prompt and one image, no branch/script/target/planner/checker metadata. Repetitions assess determinism, not accuracy.'}
    assert len(cases) == plan['max_calls']
    write(folder / 'plan.json', plan)
    write(folder / 'manifest.json', {'gpu': args.gpu, 'protocol': plan['protocol']})
    print(json.dumps({'folder': str(folder), 'cases': len(cases), 'videos': len(videos)}))


def admission(gpu):
    assert gpu in (0, 1, 2)
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    uuid, free, used, util = row[0], *map(int, row[1:])
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True)
    pids = [int(line.split(', ')[1]) for line in processes.splitlines() if line.split(', ')[0] == uuid]
    if pids or free < 75000 or used > 1000 or util > 5:
        raise RuntimeError(f'Occupied GPU {gpu}: free={free}, used={used}, util={util}, PIDs={pids}')
    return {'gpu_uuid': uuid, 'admission_free_mib': free, 'admission_compute_pids': pids}


def supervise(folder):
    plan = read(folder / 'plan.json')
    if (folder / 'status.json').exists():
        raise FileExistsError('Recorded attempts are never restarted')
    started = time.monotonic()
    status = {'status': 'preflight', 'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'gpu': plan['gpu']}
    write(folder / 'status.json', status)
    child = None
    try:
        status.update(admission(plan['gpu']))
        assert sha(folder / 'runner.py') == plan['source_sha256']
        assert sha(folder / 'model_manifest.json') == plan['model_manifest_sha256']
        for name, row in read(folder / 'model_manifest.json')['files'].items():
            assert sha(Path(plan['model_path']) / name) == row['sha256'], name
        for name, version in plan['package_versions'].items():
            assert importlib.metadata.version(name) == version, name
        for row in plan['cases']:
            assert sha(folder / row['image']) == row['image_sha256']
        status.update(admission(plan['gpu']))
        with (folder / 'worker.log.txt').open('x') as log:
            child = subprocess.Popen([plan['python'], '-u', str(folder / 'runner.py'), '--worker', '--folder', str(folder)],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(plan['gpu']), 'OMP_NUM_THREADS': '4',
                     'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status='running', pid=child.pid)
            write(folder / 'status.json', status)
            code = child.wait(timeout=max(1, plan['timeout_seconds'] - (time.monotonic() - started)))
        status.update(status='completed' if code == 0 else 'failed', returncode=code)
    except subprocess.TimeoutExpired:
        status.update(status='timeout', error='Frozen supervisor deadline reached')
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


def worker(folder):
    started = time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    plan = read(folder / 'plan.json')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(plan['model_path'], trust_remote_code=True,
                                              local_files_only=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(plan['model_path'], trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    model.eval()
    torch.cuda.synchronize()
    loaded = time.monotonic()
    (folder / 'transcripts').mkdir()
    rows = []
    for case in plan['cases']:
        tick = time.monotonic()
        row = {'id': case['id'], 'image_sha256': case['image_sha256'], 'prompt': plan['prompt'],
               'generation': plan['generation'], 'seed': plan['seed_per_call']}
        try:
            with Image.open(folder / case['image']) as im:
                image = im.convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': plan['prompt']},
                                                   {'type': 'image', 'image': image}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors='pt', return_dict=True)
            row['input_tensors'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            torch.manual_seed(plan['seed_per_call'])
            row['model_call_started'] = True
            with torch.inference_mode():
                ids = model.generate(**inputs, **plan['generation'])
            tokens = ids[0, inputs['input_ids'].size(1):]
            row['output_token_ids'] = tokens.tolist()
            row['reply'] = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            torch.cuda.synchronize()
            try:
                parsed = json.loads(row['reply'])
                assert set(parsed) == {'people_count', 'clothing', 'hair', 'uncertainty'}
                assert type(parsed['people_count']) is int and parsed['people_count'] >= 0
                assert all(isinstance(parsed[k], str) for k in ('clothing', 'hair', 'uncertainty'))
                row.update(status='completed', parsed=parsed)
            except Exception as error:
                row.update(status='parse_failed', error=f'{type(error).__name__}: {error}')
        except Exception as error:
            row.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
        row['wall_seconds'] = time.monotonic() - tick
        write(folder / 'transcripts' / (case['id'] + '.json'), row)
        rows.append(row)
        write(folder / 'partial_results.json', rows)
        print(json.dumps({'id': case['id'], 'status': row['status'], 'seconds': row['wall_seconds']}), flush=True)
    write(folder / 'result.json', {'rows': rows, 'model_load_seconds': loaded - started,
        'worker_seconds': time.monotonic() - started,
        'actual_calls': sum(r.get('model_call_started', False) for r in rows),
        'cases_attempted': len(rows),
        'completed': sum(r['status'] == 'completed' for r in rows),
        'actual_human_labels': 0, 'new_generated_videos': 0, 'adopted_into_acceptance': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--root', type=Path)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--model-manifest', type=Path)
    p.add_argument('--python', type=Path)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--timeout', type=int, default=900)
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.worker:
        worker(args.folder.resolve())
    else:
        supervise(args.folder.resolve())
