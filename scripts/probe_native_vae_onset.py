#!/usr/bin/env python3
"""Frozen causal decode interventions on existing parent/candidate latents."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare(args):
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    folder, checkout = args.folder.resolve(), args.checkout.resolve()
    assert args.gpu in (0, 1, 2)
    folder.mkdir(parents=True, exist_ok=False)
    source_files = ('utils/wan_wrapper.py', 'wan/modules/vae.py', 'timing_adapter.py')
    files = {'parent': args.parent.resolve() / 'diagnostics/latents_0000.pt',
             'candidate': args.candidate.resolve() / 'diagnostics/latents_0000.pt',
             'mask': args.mask.resolve(), 'vae_weight': checkout / 'wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'}
    inputs = {key: {'path': str(path), 'sha256': sha(path)} for key, path in files.items()}
    reference = read(args.candidate / 'manifest.json')
    audit = read(args.candidate / 'audit.json')
    assert inputs['parent']['sha256'] == reference['action']['parent_latents_sha256']
    assert inputs['candidate']['sha256'] == audit['latent_sha256']
    assert inputs['vae_weight']['sha256'] == next(r['sha256'] for r in reference['paired_assets'] if r['path'].endswith('Wan2.1_VAE.pth'))
    assert reference['action']['latent_range'] == [81, 121]
    parent, candidate = [torch.load(files[n], map_location='cpu', weights_only=True) for n in ('parent', 'candidate')]
    mask = torch.load(files['mask'], map_location='cpu', weights_only=True)
    assert tuple(parent.shape) == tuple(candidate.shape) == (1, 241, 16, 60, 104)
    assert parent.dtype == candidate.dtype == torch.bfloat16
    assert torch.equal(parent[:, :81], candidate[:, :81])
    assert mask['parent_latents_sha256'] == inputs['parent']['sha256']
    assert tensor_sha(mask['mask'].to(torch.uint8)) == reference['action']['mask_tensor_sha256']
    rows = []
    for index in range(81, 121):
        delta = candidate[:, index].float() - parent[:, index].float()
        region = mask['mask'][index - 81].bool()
        assert not delta[..., ~region].any()
        rows.append({'latent': index, 'export_frames': [4 * (index - 1), 4 * index - 1],
            'mask_cells': int(region.sum()), 'masked_mae': delta[..., region].abs().mean().item(),
            'masked_rms': delta[..., region].square().mean().sqrt().item(), 'max_abs': delta.abs().max().item()})
    write(folder / 'latent_preflight.json', {'rows': rows, 'prefix_81_exact': True,
        'outside_mask_exact': True, 'parent_shape': list(parent.shape), 'dtype': str(parent.dtype)})
    shutil.copy2(Path(__file__), folder / 'runner.py')
    shutil.copy2(Path(__file__).with_name('probe_native_observations.py'), folder / 'admission_source.py')
    for name in source_files:
        target = folder / 'native_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkout / name, target)
    for name in ('manifest.json', 'audit.json'):
        shutil.copy2(args.candidate / name, folder / ('candidate_' + name))
    shutil.copy2(args.parent / 'diagnostics/timing_0000.json', folder / 'timing.json')
    plan = {'protocol': 'native_vae_onset_block_factorial_v1', 'frozen_unix': time.time(),
        'hypothesis': 'Measure when the first edited native block influences decoded pixels and how much later edited latents contribute. Separate raw causal decoding from lossy MP4 changes; do not infer garment accuracy from numerical deltas.',
        'gpu': args.gpu, 'python': str(args.python.absolute()), 'checkout': str(checkout),
        'timeout_seconds': 900, 'input_files': inputs,
        'native_source_sha256': {name: sha(checkout / name) for name in source_files},
        'source_sha256': sha(folder / 'runner.py'), 'admission_source_sha256': sha(folder / 'admission_source.py'),
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')},
        'latent_prefix_length': 120, 'decode_chunk_size': 60, 'export_frames': 476,
        'early_block': [81, 84], 'later_block': [84, 120],
        'cases': [{'id': 'parent', 'early_candidate': False, 'later_candidate': False},
                  {'id': 'early_only', 'early_candidate': True, 'later_candidate': False},
                  {'id': 'later_only', 'early_candidate': False, 'later_candidate': True},
                  {'id': 'candidate', 'early_candidate': True, 'later_candidate': True},
                  {'id': 'parent_repeat', 'early_candidate': False, 'later_candidate': False}],
        'saved_frames': [316, 319, 320, 321, 322, 323, 324, 327, 331, 332, 335, 339, 351, 352, 367, 383],
        'raw_window': [312, 384], 'seed_per_decode': 0, 'dtype': 'bfloat16',
        'acceptance_tests': 'Record actual first changed raw frames, exact equalities before interventions, unchanged parent repetition, all per-frame deltas and full mask statistics. Do not force expectations into results.',
        'failure_policy': 'One frozen run, five VAE decodes, no diffusion or VLM calls. Execution failure/900s deadline stops remaining cases; retain all partial data. No retries or candidate adoption.',
        'scope': '2x2 intervention on existing native latent blocks from one exposed development script. Hybrids are off-distribution decoder probes, not repaired candidates or measured semantic improvements.'}
    write(folder / 'plan.json', plan)
    write(folder / 'manifest.json', {'gpu': args.gpu, 'protocol': plan['protocol']})
    write(folder / 'preparation.json', {'wall_seconds': time.monotonic() - started, 'plan_sha256': sha(folder / 'plan.json')})
    print(json.dumps({'folder': str(folder), 'plan_sha256': sha(folder / 'plan.json'), 'cases': 5}))


def worker(folder):
    import torch
    import numpy as np
    from PIL import Image
    plan = read(folder / 'plan.json')
    checkout = Path(plan['checkout'])
    sys.path.insert(0, str(checkout))
    os.chdir(checkout)
    from utils.wan_wrapper import WanVAEWrapper
    from timing_adapter import decode_continuous
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    parent, candidate = [torch.load(plan['input_files'][name]['path'], map_location='cpu', weights_only=True)[:, :120] for name in ('parent', 'candidate')]
    mask = torch.load(plan['input_files']['mask']['path'], map_location='cpu', weights_only=True)['mask']
    vae = WanVAEWrapper().to(device='cuda:0', dtype=torch.bfloat16).eval()
    torch.cuda.synchronize()
    loaded = time.monotonic()
    (folder / 'frames').mkdir()
    (folder / 'raw_local_only').mkdir()
    rows, outputs = [], {}
    baseline = None
    for case in plan['cases']:
        tick = time.monotonic()
        latent = parent.clone()
        if case['early_candidate']:
            latent[:, 81:84] = candidate[:, 81:84]
        if case['later_candidate']:
            latent[:, 84:120] = candidate[:, 84:120]
        torch.manual_seed(plan['seed_per_decode'])
        raw = decode_continuous(vae, latent.to('cuda'), chunk_size=plan['decode_chunk_size'])
        pixels = (raw[:, 1:] * .5 + .5).clamp(0, 1)[0].contiguous()
        assert list(pixels.shape) == [476, 3, 480, 832]
        del raw, latent
        torch.cuda.synchronize()
        if baseline is None:
            baseline = pixels.clone()
        delta_rows = []
        for index in range(len(pixels)):
            difference = pixels[index] - baseline[index]
            row = {'frame': index, 'max_abs': difference.abs().max().item(),
                'mae': difference.abs().mean().item(), 'rms': difference.square().mean().sqrt().item(),
                'changed_values': int(difference.count_nonzero()),
                'raw_frame_sha256': tensor_sha(pixels[index])}
            if index >= 320:
                region = mask[index // 4 + 1 - 81].bool().repeat_interleave(8, 0).repeat_interleave(8, 1)
                row['mask_projected_mae'] = difference[:, region].abs().mean().item()
                row['mask_projected_rms'] = difference[:, region].square().mean().sqrt().item()
            delta_rows.append(row)
        images = {}
        for frame in plan['saved_frames']:
            path = folder / 'frames' / f"{case['id']}_{frame}.png"
            arr = (pixels[frame] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(path)
            images[str(frame)] = sha(path)
        window = pixels[plan['raw_window'][0]:plan['raw_window'][1]].clone()
        outputs[case['id']] = window
        local = folder / 'raw_local_only' / (case['id'] + '.pt')
        torch.save(window, local)
        record = {'id': case['id'], 'case': case, 'decoded_shape': list(pixels.shape),
            'raw_video_sha256': tensor_sha(pixels), 'raw_prefix_320_sha256': tensor_sha(pixels[:320]),
            'first_changed_frame': next((r['frame'] for r in delta_rows if r['changed_values']), None),
            'frames': delta_rows, 'saved_png_sha256': images,
            'local_raw_window_file_sha256': sha(local), 'raw_window_tensor_sha256': tensor_sha(window),
            'decode_and_analysis_seconds': time.monotonic() - tick}
        rows.append(record)
        write(folder / 'partial_results.json', rows)
        print(json.dumps({k: record[k] for k in ('id', 'first_changed_frame', 'decode_and_analysis_seconds')}), flush=True)
        del pixels
    comparisons = {}
    for left, right in [('early_only', 'candidate'), ('parent', 'later_only'), ('parent', 'parent_repeat')]:
        comparisons[left + '_vs_' + right] = [{'frame': plan['raw_window'][0] + i,
            'exact': torch.equal(a, b), 'mae': (a - b).abs().mean().item(), 'max_abs': (a - b).abs().max().item()}
            for i, (a, b) in enumerate(zip(outputs[left], outputs[right]))]
    for path, expected in plan['native_source_sha256'].items():
        assert sha(checkout / path) == expected
    write(folder / 'result.json', {'protocol': plan['protocol'], 'rows': rows, 'pairwise_raw_window': comparisons,
        'worker_seconds': time.monotonic() - started, 'loading_seconds_nested': loaded - started,
        'actual_vae_decodes': len(rows), 'actual_cached_decode_calls': len(rows) * 2,
        'new_diffusion_calls': 0, 'new_vlm_calls': 0, 'candidate_adopted': False, 'human_labels': 0,
        'scope': plan['scope']})


def supervise(folder):
    plan = read(folder / 'plan.json')
    assert not (folder / 'status.json').exists(), 'Never restart an existing attempt'
    spec = importlib.util.spec_from_file_location('onset_admission', folder / 'admission_source.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    started = time.monotonic()
    status = {'status': 'preflight', 'supervisor_pid': os.getpid(), 'gpu': plan['gpu'], 'started_unix': time.time()}
    write(folder / 'status.json', status)
    child = None
    try:
        status.update(admission.admission(plan['gpu']))
        assert sha(folder / 'runner.py') == plan['source_sha256']
        assert sha(folder / 'admission_source.py') == plan['admission_source_sha256']
        for item in plan['input_files'].values():
            assert sha(Path(item['path'])) == item['sha256'], item['path']
        for path, expected in plan['native_source_sha256'].items():
            assert sha(Path(plan['checkout']) / path) == expected
        for name, version in plan['package_versions'].items():
            assert importlib.metadata.version(name) == version
        status.update(admission.admission(plan['gpu']))
        with (folder / 'worker.log.txt').open('x') as log:
            child = subprocess.Popen([plan['python'], '-u', str(folder / 'runner.py'), '--worker', '--folder', str(folder)],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(plan['gpu']), 'CUBLAS_WORKSPACE_CONFIG': ':16:8',
                     'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1'},
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
    print(json.dumps(status), flush=True)
    if status['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', action='store_true')
    for name in ('parent', 'candidate', 'mask', 'checkout', 'python'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--gpu', type=int, default=1)
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.worker:
        worker(args.folder.resolve())
    else:
        supervise(args.folder.resolve())
