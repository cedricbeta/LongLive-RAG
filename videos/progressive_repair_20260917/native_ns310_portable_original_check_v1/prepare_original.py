#!/usr/bin/env python3
"""Prepare one ns310 original with a frozen, conditional two-arm follow-up."""
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
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(a):
    began = time.monotonic()
    root, reference = a.output.resolve(), a.reference.resolve()
    read = lambda p: json.loads(p.read_text())
    write = lambda p, v: p.write_text(json.dumps(v, indent=2) + '\n')
    assert not (root / 'plan.json').exists() and not (root / 'ns310').exists()
    m = read(reference / 'manifest.json')
    assert read(reference / 'status.json')['status'] == 'completed'
    assert read(reference / 'audit.json')['engineering_checks_passed']
    checkout = Path(m['cwd'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == m['official_revision']
    assert sha(reference / 'config.yaml') == m['config_sha256']
    for name, digest in m['source_sha256'].items():
        assert sha(checkout / name) == digest, name
    assert len(m['paired_assets']) == 12
    for row in m['paired_assets']:
        assert sha(checkout / row['path']) == row['sha256'], row['path']
    assert sha(a.benchmark) == '7f04417408d900d015e2e1a8be7387928e314248f338149071f07e763b36dedf'
    prompts = [json.loads(x) for x in a.benchmark.read_text().splitlines() if x.strip()][310]['prompts']
    assert len(prompts) == 6 and 'Clara, a middle-aged woman with auburn hair and a long tan coat' in prompts[0]
    config = yaml.safe_load((reference / 'config.yaml').read_text())
    assert config['seed'] == 0 and 'repair_request' not in config and config['audit_prompt_switch'] is False
    assert shutil.disk_usage(root.parent).free > 6 * 2**30
    root.mkdir(parents=True, exist_ok=True)
    case = {'id': 'ns310', 'index': 310, 'entity': 'Clara', 'part': 'garment', 'prompts': prompts}
    plan = {
        'protocol': 'ns310_conditional_official_original_prompt_local_v1', 'frozen_unix': time.time(),
        'selection': 'Selected before viewing any official ns310 output because the same exposed script has historical local-5B coat-change evidence. That historical gain is not evidence of an official-1.3B defect or benefit.',
        'cases': [case], 'source_benchmark_index': 310, 'benchmark_sha256': sha(a.benchmark),
        'reference_manifest_sha256': sha(reference / 'manifest.json'), 'preparer_sha256': sha(Path(__file__)),
        'target_quote': 'long tan coat', 'target_entity': 'Clara', 'target_segment': 0,
        'literal_sentence': 'Clara wears a long tan coat.', 'target_expression': prompts[0] + ' Clara wears a long tan coat.',
        'authored_prompts': prompts, 'target_export_frames': [0, 160], 'target_latent_range': [0, 41],
        'target_admission_rule': 'Developer inspection of fixed frames 0/80/159 must show a clear authored coat color/type conflict in at least two views and an identifiable target woman. Correct, uncertain, obscured or ambiguous target closes both follow-up slots. An invisible coat hem alone is unknown, not a length error. No later-window substitution.',
        'new_original_attempt_ceiling': 1, 'new_prompt_attempt_ceiling': 1, 'new_local_attempt_ceiling': 1,
        'total_generation_ceiling': 3, 'per_arm_generation_ceiling': 1, 'timeout_seconds': 1800,
        'generation_seed': 0, 'repair_noise_seed': 4243, 'strength': 1.0,
        'arms': {'B_original': 'One unchanged official native original, seed 0.',
                 'P0_prompt': 'Conditional one ordinary first-window literal append, seed 0; all five subsequent prompts unchanged.',
                 'R0_local': 'Conditional one native masked first-window replay, private noise 4243, strength 1.0, same effective repair text; all ordinary prompts unchanged.'},
        'mask_recipe': {'method': 'Existing CLIPSeg-only per-frame projection and temporal union, without manual coordinates; separate from automatic SAM2 person tracking.',
                        'entity_phrase': 'woman', 'part_phrase': 'clothing', 'exclude_phrase': 'face',
                        'part_context': 'entity_box', 'threshold': 0.35, 'segment': 0,
                        'new_mask_attempt_ceiling': 1, 'cpu_timeout_seconds': 600},
        'mask_admission_rule': 'Only if target admission passes. One CPU attempt; inspect fixed latent previews 0/20/40. Empty/failed mask, or clearly selecting the wrong actor, closes both follow-up slots. Preserve and disclose partial coverage. No phrase, threshold or manual-mask retry.',
        'resource_policy': 'Sequential GPU 2 only, subject to fresh idle-device admission and existing supervisor interference checks; preserve all foreign jobs and GPU 3.',
        'failure_policy': 'Close failed/interrupted attempts without restart. No extra seed, target/window replacement, sampler variant, mask sweep, prompt rewrite or checker tuning. Preserve every completed candidate and actual failure cost.',
        'review_frames': [f for s in range(6) for f in [160*s,160*s+80,160*s+159]],
        'seed': 0, 'seconds': 60, 'fps': 16, 'resolution': [832,480],
        'generation_source_sha256': m['source_sha256'], 'paired_assets': m['paired_assets'],
        'held_out': False, 'historically_exposed': True, 'fixed_target': True,
        'automatic_diagnosis_bypassed': True, 'automatic_acceptance': False,
        'human_labels': 0, 'independent_evaluation': False, 'quality_claim': False, 'submission_ready': False,
        'limits': ['Matched target text/time, not equal compute or an isolated mask ablation.',
                   'The prior twelve-script original cohort stays complete and unchanged; this extra historical-script probe is outside that cohort.',
                   'All 60 seconds and all predeclared views must remain available; sparse stills do not certify motion, identity or actions.',
                   'Latest automatic progressive retained-improvement count is not changed by this fixed-target study.',
                   'Optional history/resident content hashes are unmeasured. A first-window edit has no nonempty strict prefix.']}
    write(root / 'plan.json', plan)
    folder = root / 'ns310/baseline'
    folder.mkdir(parents=True)
    config.update(data_path=str(folder / 'initial_prompt.txt'), output_folder=str(folder / 'videos'),
                  adapter_diagnostics_dir=str(folder / 'diagnostics'),
                  timed_segments=[{'seconds':10,'prompts':[s]} for s in prompts])
    (folder / 'initial_prompt.txt').write_text(prompts[0] + '\n')
    (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    child = {**m, 'phase': plan['protocol'], 'gpu': 2, 'source_benchmark_index':310,
             'command':[sys.executable,'-u','inference.py','--config_path',str(folder/'config.yaml')],
             'timeout_seconds':1800, 'config_sha256':sha(folder/'config.yaml'),
             'source_case_plan_sha256':sha(root/'plan.json'), 'purpose':plan['selection']}
    write(folder / 'manifest.json', child)
    shutil.copyfile(root/'plan.json', folder/'plan.json')
    shutil.copyfile(a.supervisor, folder/'supervisor.py')
    write(root/'preparation.json', {'status':'prepared_unlaunched', 'wall_seconds':time.monotonic()-began,
                                   'assets_verified':12, 'original_attempts_launched':0,
                                   'plan_sha256':sha(root/'plan.json')})
    print(json.dumps(read(root/'preparation.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['output','reference','benchmark','supervisor']:
        p.add_argument('--'+key, type=Path, required=True)
    prepare(p.parse_args())
