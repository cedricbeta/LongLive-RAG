#!/usr/bin/env python3
"""Bind one predeclared native/Euler pair to an admitted actual parent."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    began = time.monotonic()
    root, checkout, adapter = [p.resolve() for p in (args.root, args.checkout, args.adapter)]
    assert args.gpu_native in (0, 1, 2) and args.gpu_euler in (0, 1, 2)
    read = lambda p: json.loads(p.read_text())
    plan = read(root / 'plan.json')
    assert plan['protocol'] == 'two_new_exposed_baselines_before_sampler_pair_v1'
    row = next(c for c in plan['cases'] if c['id'] == args.case)
    triage = read(root / 'developer_triage.json')
    assert next(c for c in triage['cases'] if c['id'] == args.case)['decision'] == 'render_sampler_pair'
    case, parent = root / args.case, root / args.case / 'baseline'
    assert sha(root / 'plan.json') == read(parent / 'manifest.json')['source_case_plan_sha256']
    assert read(parent / 'status.json')['status'] == 'completed'
    mask = case / 'garment_mask/mask.pt'
    assert read(case / 'garment_mask/status.json')['status'] == 'completed'
    assert read(case / 'garment_mask/result.json')['nonempty']
    output = case / 'sampler_pair'
    output.mkdir(exist_ok=False)
    (output / 'prompt.txt').write_text(row['prompts'][0] + ' ' + plan['repair_sentences'][args.case] + '\n')
    write(output / 'assets_manifest.json', {'files': plan['paired_assets']})
    record = {'protocol': 'native_four_step_sampler_pair_v1', 'frozen_unix': time.time(),
        'case': args.case, 'source_case_plan_sha256': sha(root / 'plan.json'),
        'triage_sha256': sha(root / 'developer_triage.json'), 'parent_manifest_sha256': sha(parent / 'manifest.json'),
        'mask_sha256': sha(mask), 'prompt_sha256': sha(output / 'prompt.txt'),
        'preparation_source_sha256': sha(Path(__file__)),
        'installed_renderer_sha256': read(checkout / 'repair_adapter_manifest.json')['adapted_sha256'],
        'attempts_maximum': 2, 'review_frames': plan['review_frames'], 'noise_seed': 4243, 'entry_sigma': 1.,
        'modes': ['native_x0_renoise_v1', 'fixed_noise_euler_v1'],
        'scope': 'Joint comparison of update equation and private-noise refresh. Same native four timesteps and assets; not an isolated noise-only or integration-only effect.',
        'failure_policy': 'One candidate per mode; no retry, added seed, revised prompt/mask or replacement case.',
        'automatic_acceptance': False, 'human_labels': 0, 'held_out': False, 'quality_claim': False}
    write(output / 'plan.json', record)
    command = [sys.executable, str(adapter / 'prepare_repair.py'), '--parent', str(parent),
        '--parent-branch', 'native_' + args.case + '_seed0', '--checkout', str(checkout),
        '--output', str(output / 'native'), '--assets-manifest', str(output / 'assets_manifest.json'),
        '--mask', str(mask), '--prompt', str(output / 'prompt.txt'), '--noise-seed', '4243',
        '--gpu', str(args.gpu_native), '--timeout', '1800']
    with (output / 'prepare.log.txt').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Duplicate only validated, unlaunched preparation, not generated output.
    assert not (output / 'native/status.json').exists()
    shutil.copytree(output / 'native', output / 'euler')
    base = read(output / 'native/manifest.json')
    assert base['action']['latent_range'] == [0, 41]
    actions = []
    for name, mode, gpu in [('native', 'native_x0_renoise_v1', args.gpu_native),
                            ('euler', 'fixed_noise_euler_v1', args.gpu_euler)]:
        folder = output / name
        config = yaml.safe_load((folder / 'config.yaml').read_text())
        config['repair_request']['repair_sampler'] = mode
        config.update(data_path=str(folder / 'initial_prompt.txt'), output_folder=str(folder / 'videos'),
                      adapter_diagnostics_dir=str(folder / 'diagnostics'))
        (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = dict(base)
        manifest.update(gpu=gpu, config_sha256=sha(folder / 'config.yaml'),
            command=[sys.executable, '-u', 'inference.py', '--config_path', str(folder / 'config.yaml')],
            phase='fixed_target_native_sampler_pair', sampler_pair_plan_sha256=sha(output / 'plan.json'),
            native_base_action_fingerprint=base['action_fingerprint'])
        manifest['action'] = {**base['action'], 'repair_sampler': mode,
                              'renderer_protocol': 'native_four_step_sampler_pair_v1'}
        manifest['action_fingerprint'] = hashlib.sha256(json.dumps(manifest['action'], sort_keys=True).encode()).hexdigest()
        write(folder / 'manifest.json', manifest)
        shutil.copy2(output / 'plan.json', folder / 'plan.json')
        actions.append({'mode': mode, 'action_fingerprint': manifest['action_fingerprint'],
                        'manifest_sha256': sha(folder / 'manifest.json'), 'config_sha256': sha(folder / 'config.yaml')})
    write(output / 'preparation.json', {'status': 'prepared_unlaunched', 'actions': actions,
        'paired_assets_hashed_once': len(base['paired_assets']), 'wall_seconds': time.monotonic() - began,
        'native_input_validation_command': command})
    print(json.dumps({'status': 'prepared_unlaunched', 'case': args.case, 'actions': actions}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'checkout', 'adapter'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--case', choices=['ns000', 'ns280'], required=True)
    p.add_argument('--gpu-native', type=int, default=0)
    p.add_argument('--gpu-euler', type=int, default=2)
    prepare(p.parse_args())
