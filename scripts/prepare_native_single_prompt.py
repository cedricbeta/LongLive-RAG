#!/usr/bin/env python3
"""Freeze one exact-duration, unchanged official MovieGenBench prompt pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare(args):
    started = time.monotonic()
    template, checkout, output = args.template.resolve(), args.checkout.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new attempt directory; existing attempts are never resumed silently')
    if args.gpu not in (0, 1, 2):
        raise ValueError('GPU 3 is reserved')
    if args.seconds not in (60, 120):
        raise ValueError('This development protocol freezes only 60- or 120-second durations')
    original = json.loads((template / 'manifest.json').read_text())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    assert revision == original['official_revision']
    for name, digest in original['source_sha256'].items():
        assert sha(checkout / name) == digest, name
    assets = json.loads(args.assets_manifest.read_text())['files']
    required = {
        'checkpoints/longlive_base.pt', 'checkpoints/longlive_lora.pt',
        'checkpoints/ae_latent_mem.pt', 'checkpoints/moviegenbench_128_refined.txt',
        'wan_models/Wan2.1-T2V-1.3B/config.json',
        'wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
        'wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors',
        'wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth',
        *{'wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/' + name for name in
          ('special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json')},
    }
    assert required <= {item.get('path', item.get('file')) for item in assets}
    checked = []
    for item in assets:
        name = item.get('path', item.get('file'))
        digest = sha(checkout / name)
        assert digest == item['sha256'], name
        checked.append({'path': name, 'sha256': digest})
    corpus = checkout / 'checkpoints/moviegenbench_128_refined.txt'
    prompts = corpus.read_text().splitlines()
    assert len(prompts) == 128 and 0 <= args.index < len(prompts)
    prompt = prompts[args.index]
    config = yaml.safe_load((template / 'config.yaml').read_text())
    config.pop('prompt_schedule', None)
    config.pop('repair_request', None)
    output.mkdir(parents=True)
    (output / 'initial_prompt.txt').write_text(prompt + '\n')
    config.update(data_path=str(output / 'initial_prompt.txt'), output_folder=str(output / 'videos'),
                  adapter_diagnostics_dir=str(output / 'diagnostics'),
                  timed_segments=[{'seconds': args.seconds, 'prompts': [prompt]}],
                  num_output_frames=4 * args.seconds + 1, seed=args.seed)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(template / 'supervisor.py', output / 'supervisor.py')
    manifest = {
        'phase': 'official_moviegen_single_prompt_development', 'gpu': args.gpu,
        'command': [str(args.generator_python.absolute()), '-u', 'inference.py',
                    '--config_path', str(output / 'config.yaml')],
        'cwd': str(checkout), 'timeout_seconds': args.timeout,
        'config_sha256': sha(output / 'config.yaml'), 'source_sha256': original['source_sha256'],
        'supervisor_sha256': sha(output / 'supervisor.py'), 'official_revision': revision,
        'source_benchmark': 'official_moviegenbench_128_refined', 'source_benchmark_index': args.index,
        'prompt_corpus_sha256': sha(corpus), 'prompt_sha256': sha(output / 'initial_prompt.txt'),
        'expected_output': {'frames': args.seconds * 16, 'fps': 16, 'seconds': args.seconds,
                            'switch_export_frames': [], 'switch_latent_frames': []},
        'held_out': False, 'correction_enabled': False, 'quality_claim': False,
        'assets_verified': checked, 'preparation_wall_seconds': time.monotonic() - started,
        'preparation_source_sha256': sha(Path(__file__)),
        'template_manifest_sha256': sha(template / 'manifest.json'),
        'template_config_sha256': sha(template / 'config.yaml'),
        'changes_from_official_default': [
            'Continuous VAE feature cache across decode chunks',
            'Drop one initial decoded warmup frame',
            'Final short generation block if needed for exact duration'],
        'limits': ['One exposed development prompt, not a formal evaluation.',
                   'One unchanged single prompt throughout; no synthetic storyline or prompt switch.',
                   'This uses the disclosed exact-duration adapter, not the unmodified official export path.',
                   'Generation success does not establish visual quality, entity consistency or repair efficacy.'],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'prepared': str(output), 'assets_verified': len(checked),
            'expected_output': manifest['expected_output'],
            'next_command': [str(args.generator_python.absolute()), str(output / 'supervisor.py'), str(output)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('template', 'checkout', 'output', 'assets-manifest', 'generator-python'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--seconds', type=int, choices=(60, 120), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=2400)
    print(json.dumps(prepare(parser.parse_args()), indent=2))
