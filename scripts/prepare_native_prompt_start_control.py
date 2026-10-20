#!/usr/bin/env python3
"""Prepare one same-boundary ordinary prompt control for the exposed Kai case."""
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
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    began = time.monotonic()
    root, reference, supervisor = [p.resolve() for p in (args.root, args.reference_study, args.supervisor)]
    assert not root.exists()
    read = lambda p: json.loads(p.read_text())
    prior = read(reference / 'plan.json')
    assert prior['protocol'] == 'archived_ns280_local_repair_by_suffix_cue_v1'
    assert read(reference / 'pair_check.json')['status'] == 'passed'
    assert read(reference / 'summary.json')['remaining_generation_budget'] == 0
    refs = {'B_original': (reference / 'B_original').resolve(),
            'P30_prompt': reference / 'P_cue', 'RP_local_cue': reference / 'RP_local_cue'}
    manifests = {name: read(path / 'manifest.json') for name, path in refs.items()}
    configs = {name: yaml.safe_load((path / 'config.yaml').read_text()) for name, path in refs.items()}
    for name, path in refs.items():
        assert read(path / 'status.json')['status'] == 'completed'
        assert sha(path / 'config.yaml') == manifests[name]['config_sha256']
        assert manifests[name]['source_benchmark_index'] == 280
    manifest = manifests['B_original']
    checkout = Path(manifest['cwd'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == manifest['official_revision']
    for name, digest in manifest['source_sha256'].items():
        assert sha(checkout / name) == digest
    for item in manifest['paired_assets']:
        assert sha(checkout / item['path']) == item['sha256']
    authored = configs['B_original']['timed_segments']
    assert len(authored) == 6 and all(s['seconds'] == 10 for s in authored)
    assert 'Kai, an older man with silver hair' in authored[0]['prompts'][0]
    assert authored[2]['prompts'][0].startswith('Kai and Ava stand at the classroom front table,')
    sentence = 'Kai has silver hair.'
    actual = [{'seconds': 10, 'prompts': [s['prompts'][0] + (' ' + sentence if i >= 2 else '')]}
              for i, s in enumerate(authored)]
    assert actual[:2] == authored[:2]
    assert actual[3:] == configs['P30_prompt']['timed_segments'][3:]
    assert actual[2]['prompts'] == configs['RP_local_cue']['repair_request']['prompts']
    root.mkdir(parents=True)
    plan = {'protocol': 'ns280_same_boundary_whole_prompt_control_v1', 'frozen_unix': time.time(),
        'source_benchmark_index': 280, 'seed': 0, 'seconds': 60, 'fps': 16, 'resolution': [832, 480],
        'selection': 'Post-outcome exposed control motivated by earlier RP versus P30 timing asymmetry. Not a new heldout case or replacement for a failed admission.',
        'question': 'Does ordinary whole-shot conditioning with the same silver sentence at20s match or exceed local repair plus the same later reminders?',
        'arms': {'B_original': 'Archived original baseline.', 'P30_prompt': 'Archived cue-only from30s.',
            'P20_prompt': 'One new ordinary native run; cue begins at20s and continues at30/40/50s.',
            'RP_local_cue': 'Archived local20–30s repair with the same cue at30/40/50s.'},
        'new_generation_ceiling': 1, 'timeout_seconds': 1800, 'new_arm': 'P20_prompt',
        'archived_references': {name: {'path': str(path), 'manifest_sha256': sha(path / 'manifest.json'),
            'video_sha256': sha(next((path / 'videos').glob('*.mp4'))),
            'latent_sha256': sha(path / 'diagnostics/latents_0000.pt'),
            'generation_seconds_previously_incurred': read(path / 'status.json')['wall_seconds']}
            for name, path in refs.items()},
        'reference_study_plan_sha256': sha(reference / 'plan.json'),
        'authored_segments': authored, 'effective_segments': actual, 'sentence': sentence,
        'cue_start_export_frame': 320, 'cue_start_latent_frame': 81,
        'actual_change_vs_P30': 'Only third-shot ordinary prompt gets the same silver sentence. Later prompts, seed, weights, sampler, timing and all other generation settings remain equal.',
        'pairing_rule': 'Actual first81latents must match B, P30 and RP. All5executed switch text hashes must match the frozen condition. Mismatch invalidates paired attribution; no retry.',
        'predeclared_review_frames': [f for shot in range(6) for f in (160 * shot, 160 * shot + 80, 160 * shot + 159)],
        'review_frames': [f for shot in range(6) for f in (160 * shot, 160 * shot + 80, 160 * shot + 159)],
        'interpretation': 'If P20 also corrects sampled25–30s hair and retains it later, the earlier local timing benefit does not establish local-method superiority. Compare side effects and actual cost separately; no scalar quality win from hair alone.',
        'static_control_scope': 'Direct predeclared prompt appendix with exact authored actor/attribute quotes, not an automated detector/planner. The narrow continuation helper is neither extended nor used for the plural third-shot subject.',
        'paired_assets': manifest['paired_assets'], 'generation_source_sha256': manifest['source_sha256'],
        'new_mask_calls': 0, 'new_semantic_checker_calls': 0, 'additional_private_denoiser_calls': 0,
        'failure_policy': 'One new generation, no new seed/prompt/mask/sampler/case or retry. Keep all outcomes and old closed studies unchanged.',
        'limits': ['One exposed script/seed, selected after earlier results; not a success rate or heldout evidence.',
            'A comparison of methods changes whole-shot text conditioning versus masked private repair; not an isolated mask-only ablation.',
            'No adaptive feedback, fixed-retry/best-of-N superiority, independent quality acceptance or human labels.',
            'Optional history/resident tensor byte hashes and raw-VAE prefix equality remain unmeasured.'],
        'automatic_acceptance': False, 'human_labels': 0, 'held_out': False, 'submission_ready': False,
        'preparer_sha256': sha(Path(__file__))}
    write(root / 'plan.json', plan)
    shutil.copy2(Path(__file__), root / 'prepare_source.py')
    for name, path in refs.items():
        (root / name).symlink_to(path, target_is_directory=True)
    out = root / 'P20_prompt'
    out.mkdir()
    config = dict(configs['B_original'])
    assert 'repair_request' not in config
    config.update(data_path=str(out / 'initial_prompt.txt'), output_folder=str(out / 'videos'),
                  adapter_diagnostics_dir=str(out / 'diagnostics'), timed_segments=actual)
    (out / 'initial_prompt.txt').write_text(actual[0]['prompts'][0] + '\n')
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    child = {**manifest, 'phase': 'same_boundary_whole_prompt_control', 'gpu': 0,
        'command': [sys.executable, '-u', 'inference.py', '--config_path', str(out / 'config.yaml')],
        'timeout_seconds': 1800, 'config_sha256': sha(out / 'config.yaml'),
        'study_plan_sha256': sha(root / 'plan.json'), 'quality_claim': False, 'correction_enabled': False,
        'purpose': plan['question'], 'cue_start_latent_frame': 81}
    write(out / 'manifest.json', child)
    shutil.copy2(root / 'plan.json', out / 'plan.json')
    shutil.copy2(supervisor, out / 'supervisor.py')
    write(root / 'preparation.json', {'status': 'prepared_unlaunched', 'wall_seconds': time.monotonic() - began,
        'assets_verified': len(manifest['paired_assets']), 'supervisor_source_sha256': sha(supervisor)})
    print(json.dumps({'status': 'prepared_unlaunched', 'seconds': time.monotonic() - began}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'reference-study', 'supervisor']:
        parser.add_argument('--' + key, type=Path, required=True)
    prepare(parser.parse_args())
