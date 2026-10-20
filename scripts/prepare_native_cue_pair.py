#!/usr/bin/env python3
"""Prepare one identical local repair with versus without a literal suffix cue."""
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
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    started = time.monotonic()
    root, checkout, adapter = [p.resolve() for p in (args.root, args.checkout, args.adapter)]
    sys.path.insert(0, str(adapter))
    from continuation_cue import compile_cue, digest
    read = lambda p: json.loads(p.read_text())
    plan = read(root / 'plan.json')
    assert plan['protocol'] == 'new_ns235_local_repair_by_suffix_cue_factorial_v1'
    assert read(root / 'repair_admission.json')['decision'] == 'render_local_pair'
    assert read(root / 'baseline_pair_check.json')['prefix_latents_exact']
    assert sha(adapter / 'continuation_cue.py') == sha(checkout / 'continuation_cue.py') == plan['cue_source_sha256']
    assert read(root / 'hair_mask/status.json')['status'] == 'completed'
    assert read(root / 'hair_mask/result.json')['nonempty']
    parent = root / 'B_original'
    assert read(parent / 'status.json')['status'] == 'completed'
    assert not (root / 'R_local').exists() and not (root / 'RP_local_cue').exists()
    timing = read(parent / 'diagnostics/timing_0000.json')
    schedule = {int(k): v for k, v in timing['schedule'].items()}
    effective, cue_record = compile_cue(schedule, plan['continuation_cue'], earliest_latent=121)
    assert cue_record == plan['cue_record']
    (root / 'repair_prompt.txt').write_text(plan['repair_prompt'] + '\n')
    command = [sys.executable, str(adapter / 'prepare_repair.py'), '--parent', str(parent),
        '--parent-branch', 'native_ns235_seed0', '--checkout', str(checkout),
        '--output', str(root / 'R_local'), '--assets-manifest', str(root / 'assets_manifest.json'),
        '--mask', str(root / 'hair_mask/mask.pt'), '--prompt', str(root / 'repair_prompt.txt'),
        '--noise-seed', '4245', '--gpu', '0', '--timeout', '1800']
    with (root / 'prepare_local_pair.log.txt').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert not (root / 'R_local/status.json').exists()
    shutil.copytree(root / 'R_local', root / 'RP_local_cue')
    base = read(root / 'R_local/manifest.json')
    assert base['action']['latent_range'] == [81, 121]
    actions = []
    for name, gpu in [('R_local', 0), ('RP_local_cue', 2)]:
        folder = root / name
        cue = plan['continuation_cue'] if name == 'RP_local_cue' else None
        actual_schedule = effective if cue else schedule
        config = yaml.safe_load((folder / 'config.yaml').read_text())
        config['repair_request']['continuation_cue'] = cue
        config.update(data_path=str(folder / 'initial_prompt.txt'), output_folder=str(folder / 'videos'),
                      adapter_diagnostics_dir=str(folder / 'diagnostics'))
        (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        action = {**base['action'], 'continuation_cue': cue,
                  'effective_schedule_sha256': digest(actual_schedule),
                  'renderer_protocol': 'literal_hair_color_continuation_v1'}
        manifest = {**base, 'phase': 'fixed_target_local_repair_by_suffix_cue_factorial', 'gpu': gpu,
            'command': [sys.executable, '-u', 'inference.py', '--config_path', str(folder / 'config.yaml')],
            'config_sha256': sha(folder / 'config.yaml'), 'action': action,
            'action_fingerprint': digest(action), 'native_base_action_fingerprint': base['action_fingerprint'],
            'study_plan_sha256': sha(root / 'plan.json'),
            'expected_continuation_cue_record': cue_record if cue else {'enabled': False, 'changes': []},
            'timing_record_semantics': 'Authored schedule remains bound to B; actual changed text is separately recorded and must be checked against prompt-switch hashes.'}
        write(folder / 'manifest.json', manifest)
        shutil.copy2(root / 'plan.json', folder / 'plan.json')
        actions.append({'arm': name, 'manifest_sha256': sha(folder / 'manifest.json'),
                        'action_fingerprint': manifest['action_fingerprint']})
    write(root / 'local_pair_preparation.json', {'status': 'prepared_unlaunched', 'actions': actions,
        'wall_seconds': time.monotonic() - started, 'parent_validation_command': command,
        'source_sha256': sha(Path(__file__))})
    print(json.dumps({'status': 'prepared_unlaunched', 'actions': actions}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'checkout', 'adapter']:
        parser.add_argument('--' + key, type=Path, required=True)
    prepare(parser.parse_args())
