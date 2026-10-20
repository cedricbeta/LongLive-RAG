#!/usr/bin/env python3
"""Freeze OFF/ON native repair diagnostics from one existing candidate action."""
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare(args):
    started = time.monotonic()
    if args.gpu not in (0, 1, 2):
        raise ValueError('GPU 3 is reserved')
    folder, reference, checkout, adapter = [p.resolve() for p in
        (args.folder, args.reference, args.checkout, args.adapter)]
    old_manifest = json.loads((reference / 'manifest.json').read_text())
    old = yaml.safe_load((reference / 'config.yaml').read_text())['repair_request']
    assert old['entry_sigma'] == 1.0 and old['noise_seed'] == 4244 and old['latent_range'] == [81, 121]
    assert old_manifest['action_fingerprint'] == '13e81e51970795f9717e240f84180f436584ff379df4ef59f3ba82e79d2b6db6'
    assert json.loads((reference / 'status.json').read_text())['status'] == 'completed'
    folder.mkdir(parents=True, exist_ok=False)
    parent = Path(old['parent_latents_path']).parents[1]
    (folder / 'prompt.txt').write_text(old['prompts'][0] + '\n')
    shutil.copy2(old['mask_path'], folder / 'automatic_mask.pt')
    write(folder / 'assets_manifest.json', {'files': old_manifest['paired_assets']})
    shutil.copy2(Path(__file__), folder / 'prepare_source.py')
    for name in ('manifest.json', 'audit.json', 'config.yaml'):
        shutil.copy2(reference / name, folder / ('reference_' + name))
    for name in ('repair_context.py', 'install_repair_context.py', 'prepare_repair.py', 'run_prepared.py',
                 'repair_adapter.py', 'audit_candidate.py', 'audit_controls.py', 'install_repair.py', 'install.py'):
        (folder / 'source').mkdir(exist_ok=True)
        shutil.copy2(adapter / name, folder / 'source' / name)
    installed = json.loads((checkout / 'repair_adapter_manifest.json').read_text())
    for name in installed['adapted_sha256']:
        target = folder / 'native_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkout / name, target)
    shutil.copy2(checkout / 'repair_adapter_manifest.json', folder / 'installed_adapter_manifest.json')
    plan = {'protocol': 'native_private_context_control_v1', 'frozen_unix': time.time(),
        'hypothesis': 'Historical garment context may resist localized recoloring/type repair. Removing all historical self-attention keys only during private repair tests this combined-context mechanism, not RAG retrieval alone.',
        'reference': str(reference), 'reference_manifest_sha256': sha(reference / 'manifest.json'),
        'reference_video_sha256': sha(next((reference / 'videos').glob('*.mp4'))),
        'reference_latents_sha256': sha(reference / 'diagnostics/latents_0000.pt'),
        'parent': str(parent), 'gpu': args.gpu, 'checkout': str(checkout),
        'native_source_sha256': installed['adapted_sha256'],
        'preparation_source_sha256': sha(Path(__file__)),
        'adapter_source_sha256': {p.name: sha(p) for p in (folder / 'source').iterdir()},
        'mask_file_sha256': sha(folder / 'automatic_mask.pt'), 'prompt_file_sha256': sha(folder / 'prompt.txt'),
        'cases': [{'id': 'off', 'mode': 'full_context_v1'}, {'id': 'on', 'mode': 'current_block_only_v1'}],
        'attempts_maximum': 2, 'per_render_timeout_seconds': 1800,
        'on_admission': 'Run ON only after OFF completes and all 241 latent tensors, all decoded MP4 bytes, retrieval records and recorded cache states equal the archived natural-expression reference (ignore wall times and new context trace). If this fails, retain OFF failure and do not run ON.',
        'unchanged': ['original parent', 'automatic mask', 'repair prompt', 'seed 4244', 'sigma 1.0',
            '12 model assets', 'authored prompts/timing', 'private sampler', 'native ordinary denoising',
            'current-block RoPE coordinates', 'native clean recache', 'suffix retrieval and generation'],
        'changed_on': 'Only private repair self-attention sees current-block K/V; historical sink, retrieved and local-past keys are omitted together. Cache construction/metadata remain native.',
        'review_frames': [316, 319, 320, 321, 322, 323, 324, 327, 331, 332, 335, 339, 351, 352, 367, 383, 400, 448, 479, 672, 720, 768],
        'engineering_checks': ['prefix latents exact', 'outside-mask latents exact', 'native RNG preserved',
            'authored crossattention restored', 'cache positions/pool lengths preserved',
            '56 private forwards; ON exactly 30 self-attention calls each; mode absent from ordinary calls'],
        'failure_policy': 'No restart, new seed, revised mask or additional generation after viewing outputs. Preserve every attempt. No automatic adoption or semantic pass inferred from numeric deltas.',
        'automatic_retention': False, 'planner_action_added': False, 'held_out': False,
        'independent_human_labels': 0, 'quality_claim': False,
        'limits': ['One development-exposed script and fixed action; not a fairness, generalization or method-benefit experiment.',
            'Full offline replay; not streaming.', 'Visual inspection is developer inspection, not independent human evaluation.']}
    write(folder / 'plan.json', plan)
    records = []
    for case in plan['cases']:
        output = folder / case['id']
        command = [sys.executable, str(adapter / 'prepare_repair.py'), '--parent', str(parent),
            '--parent-branch', old['parent_branch_id'], '--checkout', str(checkout), '--output', str(output),
            '--assets-manifest', str(folder / 'assets_manifest.json'), '--mask', str(folder / 'automatic_mask.pt'),
            '--prompt', str(folder / 'prompt.txt'), '--noise-seed', '4244', '--gpu', str(args.gpu), '--timeout', '1800']
        with (folder / (case['id'] + '_prepare.log.txt')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['action'] == old_manifest['action']
        assert manifest['paired_assets'] == old_manifest['paired_assets']
        config = yaml.safe_load((output / 'config.yaml').read_text())
        config['repair_request']['repair_context_mode'] = case['mode']
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest['config_sha256'] = sha(output / 'config.yaml')
        manifest['native_base_action_fingerprint'] = manifest['action_fingerprint']
        manifest['action'] = {**manifest['action'], 'repair_context_mode': case['mode'],
                              'renderer_protocol': 'native_private_context_control_v1'}
        manifest['action_fingerprint'] = hashlib.sha256(json.dumps(manifest['action'], sort_keys=True).encode()).hexdigest()
        manifest['phase'] = 'standalone_native_private_context_diagnostic'
        manifest['context_control_plan_sha256'] = sha(folder / 'plan.json')
        write(output / 'manifest.json', manifest)
        shutil.copy2(folder / 'plan.json', output / 'plan.json')
        records.append({'id': case['id'], 'prepare_command': command,
                        'native_base_action_fingerprint': manifest['native_base_action_fingerprint'],
                        'action_fingerprint': manifest['action_fingerprint'],
                        'manifest_sha256': sha(output / 'manifest.json'), 'config_sha256': manifest['config_sha256']})
    write(folder / 'preparation.json', {'wall_seconds': time.monotonic() - started,
        'plan_sha256': sha(folder / 'plan.json'), 'cases': records, 'status': 'prepared_unlaunched'})
    print(json.dumps({'folder': str(folder), 'plan_sha256': sha(folder / 'plan.json'), 'status': 'prepared_unlaunched'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'reference', 'checkout', 'adapter'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    prepare(p.parse_args())
