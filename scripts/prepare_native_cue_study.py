#!/usr/bin/env python3
"""Freeze one new public script and its suffix-only literal-reminder control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ADAPTER = next(p / 'adapters/official_longlive_rag' for p in Path(__file__).resolve().parents
               if (p / 'adapters/official_longlive_rag/continuation_cue.py').is_file())
sys.path.insert(0, str(ADAPTER))
from continuation_cue import compile_cue
from timing_adapter import compile_timing


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    started = time.monotonic()
    root, parent = args.root.resolve(), args.reference.resolve()
    assert not root.exists()
    original = json.loads((parent / 'manifest.json').read_text())
    checkout = Path(original['cwd'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == original['official_revision']
    for path, digest in original['source_sha256'].items():
        assert sha(checkout / path) == digest
    assets = original['paired_assets']
    assert len(assets) == 12
    for item in assets:
        assert sha(checkout / item['path']) == item['sha256']
    rows = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    prompts = rows[235]['prompts']
    assert len(prompts) == 6 and 'Kai, an older man with silver hair and a brown trench coat' in prompts[0]
    segments = [{'seconds': 10, 'prompts': [p]} for p in prompts]
    timing = compile_timing(segments, [prompts[0]])
    cue = {'schema_version': 1, 'entity': 'Kai', 'hair_color': 'silver',
           'evidence_start_latent': 0, 'apply_at': [121, 161, 201]}
    effective, cue_record = compile_cue(timing['schedule'], cue, earliest_latent=121)
    root.mkdir(parents=True)
    shutil.copy2(Path(__file__), root / 'prepare_source.py')
    shutil.copy2(ADAPTER / 'continuation_cue.py', root / 'cue_source.py')
    plan = {'protocol': 'new_ns235_local_repair_by_suffix_cue_factorial_v1',
        'frozen_unix': time.time(), 'source_benchmark_index': 235, 'prompts': prompts,
        'selection': 'Script-only selection before any ns235 rendering: explicit silver-hair definition, Kai explicitly present in all later prompts, and later prompts omit hair color. New studio scene; same exposed actors and shared narrative pattern, not held-out.',
        'seed': 0, 'fps': 16, 'seconds': 60, 'resolution': [832, 480],
        'baseline_generation_ceiling': 2, 'conditional_repair_generation_ceiling': 2,
        'total_generation_ceiling': 4, 'timeout_per_run_seconds': 1800,
        'arms': {'B_original': 'Unchanged official/timed six authored prompts.',
            'P_cue': 'No local repair; only append the exact silver-hair sentence to authored shots4–6.',
            'R_local': 'Conditional native local hair repair in shot3; original later prompts.',
            'RP_local_cue': 'Exactly the same local repair as R plus the same later cues as P.'},
        'paired_prefix_rule': 'B and P must have exactly matching first121latentframes; R and RP must share the actual B parent/mask/private prompt/seed and match through repaired latent120. Any mismatch stops causal attribution.',
        'repair_admission': 'Only if B has clearly non-silver Kai hair at one or more predeclared third-shot views320/400/479 and an automatic mask reasonably covers Kai hair at all3without broad face/Ava overlap. Already correct, unknown, failed or unsuitable localization means no R/RP; preserve B/P. No replacement case.',
        'repair_segment': 2, 'repair_latent_range': [81, 121], 'entry_sigma': 1.0,
        'private_noise_seed': 4245, 'repair_prompt': prompts[2] + ' Kai has silver hair.',
        'mask_recipe': {'entity_phrase': 'man in a brown coat', 'part_phrase': 'hair',
            'exclude_phrase': 'face', 'threshold': .35, 'part_context': 'entity_box'},
        'mask_preflight_frames': [320, 400, 479], 'continuation_cue': cue,
        'cue_record': cue_record, 'comparison_scope': 'A small local-repair × suffix-reminder factorial, not feedback versus fixed retry or budget-matched best-of-N. Do not credit cue-only benefit to corrective feedback.',
        'review_frames': [f for shot in range(6) for f in [shot*160, shot*160+80, shot*160+159]],
        'paired_assets': assets, 'generation_source_sha256': original['source_sha256'],
        'benchmark_sha256': sha(args.benchmark), 'preparation_source_sha256': sha(Path(__file__)),
        'cue_source_sha256': sha(ADAPTER / 'continuation_cue.py'),
        'held_out': False, 'automatic_acceptance': False, 'human_labels': 0,
        'failure_policy': 'One render per admitted cell only. No extra seed, prompt, mask, case or sampler after seeing outcomes. Prior Kai classroom/Noah/Iris budgets remain closed. Preserve all outcomes and measured costs.',
        'audit_scope': 'Ordinary native cursor/pool-length guards; optional history/resident byte hashing off.',
        'quality_claim': False, 'submission_ready': False}
    write(root / 'plan.json', plan)
    write(root / 'assets_manifest.json', {'files': assets})
    for arm, gpu in [('B_original', 0), ('P_cue', 2)]:
        out = root / arm
        out.mkdir()
        config = yaml.safe_load((parent / 'config.yaml').read_text())
        assert 'repair_request' not in config and config['seed'] == 0
        actual_segments = [{'seconds': 10, 'prompts': effective[t['start_latent_frame']]}
                           for t in timing['segments']] if arm == 'P_cue' else segments
        config.update(data_path=str(out / 'initial_prompt.txt'), output_folder=str(out / 'videos'),
                      adapter_diagnostics_dir=str(out / 'diagnostics'), audit_prompt_switch=False,
                      timed_segments=actual_segments)
        (out / 'initial_prompt.txt').write_text(prompts[0] + '\n')
        (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = {**original, 'phase': 'original_baseline' if arm == 'B_original' else 'suffix_literal_cue_only_control',
            'gpu': gpu, 'cwd': str(checkout), 'source_benchmark_index': 235,
            'command': [sys.executable, '-u', 'inference.py', '--config_path', str(out / 'config.yaml')],
            'config_sha256': sha(out / 'config.yaml'), 'source_case_plan_sha256': sha(root / 'plan.json'),
            'cue_intervention': None if arm == 'B_original' else cue_record,
            'audit_prompt_switch': False, 'held_out': False, 'quality_claim': False,
            'correction_enabled': False, 'purpose': plan['comparison_scope']}
        write(out / 'manifest.json', manifest)
        shutil.copy2(ADAPTER / 'run_prepared.py', out / 'supervisor.py')
        shutil.copy2(root / 'plan.json', out / 'plan.json')
    write(root / 'preparation.json', {'status': 'prepared_unlaunched',
        'wall_seconds': time.monotonic() - started, 'actual_assets_verified': len(assets)})
    print(json.dumps({'status': 'prepared_unlaunched', 'root': str(root), 'wall_seconds': time.monotonic() - started}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'reference', 'benchmark']:
        parser.add_argument('--' + key, type=Path, required=True)
    prepare(parser.parse_args())
