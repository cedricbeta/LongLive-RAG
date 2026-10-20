#!/usr/bin/env python3
"""Freeze two new exposed baselines for a bounded repair-sampler comparison."""
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
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare(args):
    began = time.monotonic()
    folder, checkout, reference = [p.resolve() for p in (args.folder, args.checkout, args.reference)]
    assert not folder.exists()
    original = json.loads((reference / 'manifest.json').read_text())
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == original['official_revision']
    for name, digest in original['source_sha256'].items():
        assert sha(checkout / name) == digest, name
    assets = json.loads(args.assets_from.read_text())['paired_assets']
    assert len(assets) == 12
    for row in assets:
        assert sha(checkout / row['path']) == row['sha256'], row['path']
    benchmark = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    # Exact public scripts, with no appearance sentence added to any baseline.
    cases = [dict(id='ns000', index=0, gpu=0, entity='Noah', part='garment', target='dark green work shirt',
                  source_quote='Noah, a tall man with a shaved head and a dark green work shirt'),
             dict(id='ns280', index=280, gpu=2, entity='Kai', part='garment', target='brown trench coat',
                  source_quote='Kai, an older man with silver hair and a brown trench coat')]
    folder.mkdir(parents=True)
    shutil.copy2(Path(__file__), folder / 'prepare_source.py')
    for row in cases:
        prompts = benchmark[row['index']]['prompts']
        assert len(prompts) == 6 and row['source_quote'] in prompts[0]
        row['prompts'] = prompts
    plan = {'protocol': 'two_new_exposed_baselines_before_sampler_pair_v1', 'frozen_unix': time.time(),
        'purpose': 'Test whether fixed-noise Euler repair differs from native x0/re-noise on two new official1.3B garment cases. First establish whether either frozen target actually conflicts. Neither success nor a baseline defect is assumed.',
        'source_sha256': sha(Path(__file__)), 'benchmark_sha256': sha(args.benchmark),
        'reference_manifest_sha256': sha(reference / 'manifest.json'),
        'cases': cases, 'seed': 0, 'seconds': 60, 'resolution': [832, 480], 'fps': 16,
        'baseline_attempts_maximum': 2, 'per_case_timeout_seconds': 1800,
        'subsequent_repair_attempts_maximum_per_case': 2,
        'subsequent_repair_rule': 'Only after a clear conflict of the frozen authored garment color or type at first-shot fixed views; unknown or already-correct means no repair. Exactly one native x0/re-noise candidate and one fixed-noise Euler candidate, sharing actual parent, mask, full authored prompt plus literal garment sentence, sigma1, seed4243, same four native timesteps, and unchanged assets. No seed retry, extra sampler, replacement case or post-result mask/prompt tuning. This is fixed-target mechanism testing, not automatic acceptance.',
        'mask_recipe': {'entity_phrase': 'man', 'part_phrase': 'clothing', 'exclude_phrase': 'face', 'part_context': 'entity_box', 'threshold': .35, 'segment': 0},
        'repair_sentences': {'ns000': 'Noah wears a dark green work shirt.', 'ns280': 'Kai wears a brown trench coat.'},
        'comparison': 'Only private repair sampling changes. Baseline/suffix generation and native retrieval stay unchanged. Native uses private fresh noise each step; Euler keeps the initial noise for background anchoring and uses native flow output with sigma differences. Both keep all authored context; no historical-context interception.',
        'review_frames': [frame for shot in range(6) for frame in (shot * 160, shot * 160 + 80, shot * 160 + 159)],
        'generation_source_sha256': original['source_sha256'], 'paired_assets': assets,
        'audit_prompt_switch': False,
        'audit_scope': 'Native prompt refresh and cursor/pool-length guards remain enabled. Optional history/resident byte hashing is disabled in both baseline and any child; do not claim content hashes were checked in these runs.',
        'held_out': False, 'historically_exposed_scripts': True, 'automatic_retention': False,
        'human_labels': 0, 'core_policy_changed': False,
        'failure_policy': 'No restart, extra seed, replacement case or extra repair beyond the two frozen sampler arms after seeing the results. No repair when target is already correct or uncertain. Do not use local5B outcomes as evidence of native1.3B repair success.'}
    write(folder / 'plan.json', plan)
    for row in cases:
        out = folder / row['id'] / 'baseline'
        out.mkdir(parents=True)
        config = yaml.safe_load((reference / 'config.yaml').read_text())
        assert 'repair_request' not in config and config['seed'] == 0
        config.update(data_path=str(out / 'initial_prompt.txt'), output_folder=str(out / 'videos'),
                      adapter_diagnostics_dir=str(out / 'diagnostics'), audit_prompt_switch=False,
                      timed_segments=[{'seconds': 10, 'prompts': [p]} for p in row['prompts']])
        (out / 'initial_prompt.txt').write_text(row['prompts'][0] + '\n')
        (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = {**original, 'phase': 'exposed_native_baseline_before_bounded_target_probe',
            'gpu': row['gpu'], 'cwd': str(checkout),
            'command': [sys.executable, '-u', 'inference.py', '--config_path', str(out / 'config.yaml')],
            'config_sha256': sha(out / 'config.yaml'), 'source_benchmark_index': row['index'],
            'source_case_plan_sha256': sha(folder / 'plan.json'), 'paired_assets': assets,
            'purpose': plan['purpose'], 'audit_prompt_switch': False,
            'audit_scope': plan['audit_scope'], 'correction_enabled': False, 'quality_claim': False}
        write(out / 'manifest.json', manifest)
        shutil.copy2(args.supervisor, out / 'supervisor.py')
        shutil.copy2(folder / 'plan.json', out / 'plan.json')
    write(folder / 'preparation.json', {'status': 'prepared_unlaunched',
        'plan_sha256': sha(folder / 'plan.json'), 'wall_seconds': time.monotonic() - began,
        'actual_assets_verified': len(assets), 'baseline_attempts_launched': 0})
    print(json.dumps({'folder': str(folder), 'status': 'prepared_unlaunched', 'wall_seconds': time.monotonic() - began}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'checkout', 'reference', 'assets-from', 'benchmark', 'supervisor'):
        p.add_argument('--' + name, type=Path, required=True)
    prepare(p.parse_args())
