#!/usr/bin/env python3
"""Prepare one reviewable native-RAG candidate from a completed parent branch.

This does not judge or retain a candidate. It binds the actual parent, binary
mask, prompt and seed and verifies the paired model assets before rendering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from omegaconf import OmegaConf
import torch
import yaml

from repair_adapter import MaskedParentRepair, file_hash, tensor_hash


def prepare(args):
    started = time.monotonic()
    parent, checkout, output = args.parent.resolve(), args.checkout.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh candidate directory')
    if json.loads((parent / 'status.json').read_text())['status'] != 'completed':
        raise ValueError('Parent generation has not completed')
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    adapter = json.loads((checkout / 'repair_adapter_manifest.json').read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if revision != adapter['official_revision'] or parent_manifest['official_revision'] != revision:
        raise ValueError('Parent and candidate must use the pinned official revision')
    if not all(file_hash(checkout / f) == h for f, h in adapter['adapted_sha256'].items()):
        raise ValueError('Installed candidate adapter changed')
    assets = json.loads(args.assets_manifest.read_text())['files']
    required = {'checkpoints/longlive_base.pt', 'checkpoints/longlive_lora.pt', 'checkpoints/ae_latent_mem.pt',
                'wan_models/Wan2.1-T2V-1.3B/config.json', 'wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
                'wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors',
                'wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth',
                *{'wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/' + name for name in
                  ['special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json']}}
    if not required <= {item.get('path', item.get('file')) for item in assets}:
        raise ValueError('Asset provenance must include base/LoRA/AE, initialization, text encoder, VAE and tokenizer')
    checked = []
    for item in assets:
        name = item.get('path', item.get('file'))
        a, b = Path(parent_manifest['cwd']) / name, checkout / name
        actual = file_hash(b)
        if actual != item['sha256'] or (a.resolve() != b.resolve() and file_hash(a) != actual):
            raise ValueError('Paired asset differs from its frozen provenance: ' + name)
        checked.append({'path': name, 'sha256': actual, 'paired_content_equal': True})
    video, = list((parent / 'videos').glob('*.mp4'))
    latent_path = parent / 'diagnostics/latents_0000.pt'
    timing_path = parent / 'diagnostics/timing_0000.json'
    timing = json.loads(timing_path.read_text())
    config = yaml.safe_load((parent / 'config.yaml').read_text())
    config.pop('repair_request', None)  # Never execute an ancestor's edit again.
    payload = torch.load(args.mask, map_location='cpu', weights_only=True)
    prompts = [args.prompt.read_text().strip()]
    if not prompts[0]:
        raise ValueError('Empty repair prompt')
    request = {'schema_version': 1, 'entry_sigma': 1.0, 'parent_branch_id': args.parent_branch,
               'latent_range': payload['latent_range'], 'noise_seed': args.noise_seed, 'prompts': prompts}
    for name, path in [('parent_latents', latent_path), ('parent_video', video), ('parent_timing', timing_path),
                       ('parent_config', parent / 'config.yaml'), ('mask', args.mask.resolve())]:
        request[name + '_path'] = str(path)
        request[name + '_sha256'] = file_hash(path)
    config.update(data_path=str(output / 'initial_prompt.txt'), output_folder=str(output / 'videos'),
                  adapter_diagnostics_dir=str(output / 'diagnostics'), repair_request=request)
    # Check actual shape, dtype, provenance, timing, binary masks and paired
    # generation settings before reserving a GPU or launching any model.
    latent = torch.load(latent_path, map_location='cpu', weights_only=True)
    MaskedParentRepair(request, torch.empty_like(latent), timing, OmegaConf.create(config))
    action = {'parent_latents_sha256': request['parent_latents_sha256'], 'latent_range': request['latent_range'],
              'mask_tensor_sha256': tensor_hash(payload['mask'].to(torch.uint8)),
              'mask_shape': list(payload['mask'].shape), 'prompt_sha256': hashlib.sha256(prompts[0].encode()).hexdigest(),
              'noise_seed': args.noise_seed, 'entry_sigma': 1.0}
    fingerprint = hashlib.sha256(json.dumps(action, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    initial = timing['schedule']['0'][0]
    (output / 'initial_prompt.txt').write_text(initial + '\n')
    (output / 'repair_prompt.txt').write_text(prompts[0] + '\n')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {'phase': 'official_rag_native_repair_candidate', 'gpu': args.gpu, 'cwd': str(checkout),
        'command': [sys.executable, '-u', 'inference.py', '--config_path', str(output / 'config.yaml')],
        'timeout_seconds': args.timeout, 'official_revision': revision,
        'config_sha256': file_hash(output / 'config.yaml'), 'source_sha256': adapter['adapted_sha256'],
        'parent_manifest_sha256': file_hash(parent / 'manifest.json'),
        'source_benchmark_index': parent_manifest.get('source_benchmark_index'),
        'expected_output': parent_manifest['expected_output'],
        'paired_assets': checked, 'action': action, 'action_fingerprint': fingerprint,
        'preparation_wall_seconds': time.monotonic() - started,
        'held_out': parent_manifest.get('held_out', False), 'repair_enabled': True,
        'quality_claim': False, 'candidate_retained': False,
        'limits': ['Render preparation does not validate localization, target improvement, identity or actions.',
                   'Parent reference is a selected rendered branch, not automatically trusted identity evidence.',
                   'A controller must account for duplicate actions, cost, acceptance and later-shot checks.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__).with_name('run_prepared.py'), output / 'supervisor.py')
    return {'command': [sys.executable, str(output / 'supervisor.py'), str(output)], 'action_fingerprint': fingerprint}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-branch', required=True)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assets-manifest', type=Path, required=True)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--noise-seed', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    args = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(prepare(args), indent=2))
