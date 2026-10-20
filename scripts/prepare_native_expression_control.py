#!/usr/bin/env python3
"""Prepare one post-failure wording control using an archived automatic action."""
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    assert args.gpu in (0, 1, 2)
    reference = args.reference.resolve()
    compiled = json.loads(args.compiled.read_text())
    entity, = [row for row in compiled['entities'] if row['name'] == args.entity]
    rows = [r for r in entity['appearance'] if r['part'] == 'garment'
            and r['source']['shot'] <= args.shot <= r['last_active_shot']]
    color, = [r for r in rows if r['aspect'] == 'color']
    kind, = [r for r in rows if r['aspect'] == 'type']
    phrase = color['expected'] + ' ' + kind['expected']
    # This case uses a literal contiguous authored phrase, not a new inferred description.
    for row in (color, kind):
        assert row['source']['quote'] in compiled['prompts'][row['source']['shot']]
        assert row['expected'] in row['source']['quote']
    assert phrase in compiled['prompts'][args.shot]
    assert phrase[0].lower() not in 'aeiou', 'This bounded control only implements the consonant article case'
    config = yaml.safe_load((reference / 'config.yaml').read_text())
    old = config['repair_request']
    assert old['entry_sigma'] == 1.0 and old['noise_seed'] == 4244
    assert old['latent_range'] == [81, 121] and args.shot == 2
    assert old['prompts'][0].startswith(compiled['prompts'][args.shot])
    assert old['prompts'][0] == (reference / 'repair_prompt.txt').read_text().strip()
    sentence = f'{args.entity} wears a {phrase}.'
    prompt = old['prompts'][0] + ' ' + sentence
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.compiled, output / 'compiled_script.json')
    shutil.copy2(args.observation_plan, output / 'observation_reference_plan.json')
    shutil.copy2(Path(old['mask_path']), output / 'automatic_mask.pt')
    shutil.copy2(Path(__file__), output / 'prepare_source.py')
    (output / 'prompt.txt').write_text(prompt + '\n')
    parent = Path(old['parent_latents_path']).parents[1]
    assets = {'files': json.loads((reference / 'manifest.json').read_text())['paired_assets']}
    (output / 'assets_manifest.json').write_text(json.dumps(assets, indent=2) + '\n')
    adapter = args.adapter.resolve()
    command = [sys.executable, str(adapter / 'prepare_repair.py'), '--parent', str(parent),
        '--parent-branch', old['parent_branch_id'], '--checkout', str(args.checkout.resolve()),
        '--output', str(output / 'candidate'), '--assets-manifest', str(output / 'assets_manifest.json'),
        '--mask', str(output / 'automatic_mask.pt'), '--prompt', str(output / 'prompt.txt'),
        '--noise-seed', '4244', '--gpu', str(args.gpu), '--timeout', '1800']
    plan = {'protocol': 'native_iris_natural_expression_control_v1', 'frozen_unix': time.time(),
        'hypothesis': 'A natural authored color/type sentence may condition the renderer differently from the grammatical-type labels and earlier separate garment-is clauses. No benefit is assumed.',
        'reference': str(reference), 'reference_manifest_sha256': sha(reference / 'manifest.json'),
        'reference_config_sha256': sha(reference / 'config.yaml'),
        'compiled_script_sha256': sha(output / 'compiled_script.json'),
        'observation_reference_plan_sha256': sha(output / 'observation_reference_plan.json'),
        'automatic_mask_file_sha256': sha(output / 'automatic_mask.pt'),
        'old_prompt': old['prompts'][0], 'new_prompt': prompt, 'appended_sentence': sentence,
        'prompt_source_rows': [color, kind], 'parent': str(parent), 'prepare_command': command,
        'prepare_source_sha256': sha(Path(__file__)),
        'adapter_source_sha256': {n: sha(adapter / n) for n in
            ('prepare_repair.py', 'run_prepared.py', 'repair_adapter.py', 'audit_candidate.py')},
        'comparison': 'One new standalone replay versus both archived seed-4244 second candidates. Same original parent, automatic mask, sigma, seed, interval, checkpoints and replay adapter; only the repair prompt changes. This is a post-failure exposed development control, outside the completed two-arm budget comparison.',
        'generation_attempts': 1, 'render_timeout_seconds': 1800,
        'review_frames': [320, 352, 400, 448, 479, 672, 720, 768],
        'observation_protocol': 'Same target-blind Molmo2-O-7B prompt, snapshot, greedy settings and full-frame times as native_iris_molmo_observations_v1; source/configuration must be bound before reviewing the new output.',
        'new_hidden_attributes_or_manual_masks': False, 'held_out': False,
        'automatic_retention': False, 'core_policy_changed': False,
        'failure_policy': 'One native generation; do not restart or choose a new seed after failure. Record all outcomes and costs. No historical guard/selection change and no candidate adoption from these sparse observations.'}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    with (output / 'prepare.log.txt').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    new_manifest = json.loads((output / 'candidate/manifest.json').read_text())
    old_manifest = json.loads((reference / 'manifest.json').read_text())
    differing = [k for k in old_manifest['action'] if old_manifest['action'][k] != new_manifest['action'][k]]
    assert differing == ['prompt_sha256'], differing
    assert old_manifest['paired_assets'] == new_manifest['paired_assets']
    assert old_manifest['source_sha256'] == new_manifest['source_sha256']
    new_request = yaml.safe_load((output / 'candidate/config.yaml').read_text())['repair_request']
    for key, value in old.items():
        if key not in ('mask_path', 'prompts'):
            assert new_request[key] == value, key
    binding = {'only_action_difference': differing, 'same_parent_mask_sigma_seed_and_interval': True,
        'same_assets_and_native_source': True, 'old_action': old_manifest['action'],
        'new_action': new_manifest['action'], 'plan_sha256': sha(output / 'plan.json')}
    (output / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    shutil.copy2(output / 'plan.json', output / 'candidate/plan.json')
    print(json.dumps({'output': str(output), 'new_sentence': sentence, 'binding': binding}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'compiled', 'adapter', 'checkout', 'output', 'observation-plan'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--entity', required=True)
    p.add_argument('--shot', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    prepare(p.parse_args())
