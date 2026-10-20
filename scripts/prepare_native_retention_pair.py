#!/usr/bin/env python3
"""Freeze two new cue arms against an already exposed baseline/local repair."""
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
    began = time.monotonic()
    root, baseline, local, mask, checkout, adapter = [p.resolve() for p in
        (args.root, args.baseline, args.local_repair, args.mask, args.checkout, args.adapter)]
    assert not root.exists()
    sys.path.insert(0, str(adapter))
    from continuation_cue import compile_cue, digest
    read = lambda p: json.loads(p.read_text())
    bm, rm = [read(p / 'manifest.json') for p in (baseline, local)]
    bc, rc = [yaml.safe_load((p / 'config.yaml').read_text()) for p in (baseline, local)]
    assert all(read(p / 'status.json')['status'] == 'completed' for p in (baseline, local))
    assert bm['source_benchmark_index'] == rm['source_benchmark_index'] == 280
    assert read(local / 'audit.json')['outside_mask_latents_exact']
    assert read(local / 'audit.json')['strict_prefix_latents'] == 81
    request = rc['repair_request']
    assert request['latent_range'] == [81, 121] and request['noise_seed'] == 4245
    assert sha(mask) == request['mask_sha256']
    assert request['parent_latents_sha256'] == sha(baseline / 'diagnostics/latents_0000.pt')
    assert request['parent_video_sha256'] == sha(next((baseline / 'videos').glob('*.mp4')))
    assert sha(local / 'config.yaml') == rm['config_sha256']
    assert sha(baseline / 'config.yaml') == bm['config_sha256']
    assert sha(baseline / 'manifest.json') == rm['parent_manifest_sha256']
    for name, frozen in bm['source_sha256'].items():
        assert sha(Path(bm['cwd']) / name) == frozen
    for name, frozen in rm['source_sha256'].items():
        assert sha(Path(rm['cwd']) / name) == frozen
    # The optional installer only inserts future schedule compilation and a log;
    # it must reproduce its transformation of the exact archived native source.
    from install_continuation_cue import patch_pipeline, patch_runner
    assert (checkout / 'pipeline/causal_inference.py').read_text() == patch_pipeline(
        (Path(rm['cwd']) / 'pipeline/causal_inference.py').read_text())
    assert (checkout / 'inference.py').read_text() == patch_runner((Path(rm['cwd']) / 'inference.py').read_text())
    assert sha(checkout / 'repair_adapter.py') == rm['source_sha256']['repair_adapter.py']
    assert sha(checkout / 'continuation_cue.py') == sha(adapter / 'continuation_cue.py')
    timing = read(baseline / 'diagnostics/timing_0000.json')
    schedule = {int(k): v for k, v in timing['schedule'].items()}
    cue = {'schema_version': 1, 'entity': 'Kai', 'hair_color': 'silver',
           'evidence_start_latent': 0, 'apply_at': [121, 161, 201]}
    effective, cue_record = compile_cue(schedule, cue, earliest_latent=121)
    assert request['prompts'] == [schedule[81][0] + ' Kai has silver hair.']
    root.mkdir(parents=True)
    plan = {'protocol': 'archived_ns280_local_repair_by_suffix_cue_v1', 'frozen_unix': time.time(),
        'source_benchmark_index': 280, 'seed': 0, 'seconds': 60, 'fps': 16, 'resolution': [832, 480],
        'selection': 'Post-outcome exposed classroom case: archived B has nonsilver hair and archived R gives local silver color then recurrence. New study tests suffix retention; it does not reopen old budgets or replace the closed ns235 admission.',
        'question': 'Does adding the same later script-derived cue retain the local silver repair, and what does local repair add over cue-only?',
        'arms': {'B_original': 'Archived unchanged six-prompt baseline; reused, not rerendered.',
            'R_local': 'Archived single native local hair repair; reused, not rerendered.',
            'P_cue': 'One new no-repair generation; original shots1–3, exact cue on shots4–6.',
            'RP_local_cue': 'One new replay with identical archived local action plus exactly the P suffix cues.'},
        'new_generation_ceiling': 2, 'per_arm_attempt_ceiling': 1, 'timeout_seconds': 1800,
        'archived_references': {name: {'path': str(path), 'manifest_sha256': sha(path / 'manifest.json'),
            'video_sha256': sha(next((path / 'videos').glob('*.mp4'))),
            'latent_sha256': sha(path / 'diagnostics/latents_0000.pt'),
            'generation_seconds_previously_incurred': read(path / 'status.json')['wall_seconds']}
            for name, path in [('B_original', baseline), ('R_local', local)]},
        'paired_assets': bm['paired_assets'], 'authored_schedule': schedule,
        'repair_action': rm['action'], 'native_base_action_fingerprint': rm['action_fingerprint'],
        'mask_sha256': sha(mask), 'mask_recomputed': False, 'manual_mask': False,
        'repair_prompt': request['prompts'][0], 'continuation_cue': cue, 'cue_record': cue_record,
        'generation_source_sha256': {'P_cue': bm['source_sha256'],
            'RP_local_cue': read(checkout / 'repair_adapter_manifest.json')['adapted_sha256']},
        'integration_validation': 'Installed pipeline/runner exactly equal applying the frozen cue patch to the archived native source; private repair adapter unchanged. Actual RP first121latents must match archived R, B/P first121must match. Any mismatch invalidates paired attribution; no rerun.',
        'review_frames': [f for shot in range(6) for f in (shot * 160, shot * 160 + 80, shot * 160 + 159)],
        'limits': ['One post-outcome exposed script/seed; not heldout or independent selection.',
            'Cue-only begins at30s; does not compare ordinary prompt rewriting inside the20–30s edit interval.',
            'This tests local repair and static suffix conditioning, not adaptive feedback versus fixed retry/best-of-N.',
            'Two archived and two new outputs use the same weights, with the disclosed isolated cue source patch for RP.',
            'Optional history/resident byte hashes and raw-VAE pixel prefix checks are not measured.',
            'Fixed views are developer inspection, not continuous motion review, independent acceptance or human labels.'],
        'failure_policy': 'One attempt per new arm. Preserve all failures; no new mask, prompt, seed, sampler, case, or same-budget retry.',
        'human_labels': 0, 'held_out': False, 'automatic_acceptance': False, 'submission_ready': False,
        'preparer_sha256': sha(Path(__file__))}
    write(root / 'plan.json', plan)
    write(root / 'assets_manifest.json', {'files': bm['paired_assets']})
    shutil.copy2(Path(__file__), root / 'prepare_source.py')
    for name, path in [('B_original', baseline), ('R_local', local)]:
        (root / name).symlink_to(path, target_is_directory=True)
    (root / 'repair_prompt.txt').write_text(request['prompts'][0] + '\n')
    command = [sys.executable, str(adapter / 'prepare_repair.py'), '--parent', str(baseline),
        '--parent-branch', request['parent_branch_id'], '--checkout', str(checkout),
        '--output', str(root / 'RP_local_cue'), '--assets-manifest', str(root / 'assets_manifest.json'),
        '--mask', str(mask), '--prompt', str(root / 'repair_prompt.txt'), '--noise-seed', '4245',
        '--gpu', '2', '--timeout', '1800']
    with (root / 'prepare_local.log.txt').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    rp = root / 'RP_local_cue'
    rmanifest = read(rp / 'manifest.json')
    assert rmanifest['action_fingerprint'] == rm['action_fingerprint']
    config = yaml.safe_load((rp / 'config.yaml').read_text())
    config['repair_request']['continuation_cue'] = cue
    (rp / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    action = {**rmanifest['action'], 'continuation_cue': cue,
              'effective_schedule_sha256': digest(effective), 'renderer_protocol': 'literal_hair_color_continuation_v1'}
    rmanifest.update(phase='archived_local_repair_plus_literal_suffix_cue',
        config_sha256=sha(rp / 'config.yaml'), action=action, action_fingerprint=digest(action),
        native_base_action_fingerprint=rm['action_fingerprint'], expected_continuation_cue_record=cue_record,
        study_plan_sha256=sha(root / 'plan.json'),
        timing_record_semantics='Authored schedule remains bound to B; executed cue text is separate and verified against actual switch hashes.')
    write(rp / 'manifest.json', rmanifest)
    p = root / 'P_cue'
    p.mkdir()
    pconfig = dict(bc)
    pconfig.update(data_path=str(p / 'initial_prompt.txt'), output_folder=str(p / 'videos'),
        adapter_diagnostics_dir=str(p / 'diagnostics'),
        timed_segments=[{'seconds': 10, 'prompts': effective[t['start_latent_frame']]} for t in timing['segments']])
    assert 'repair_request' not in pconfig
    (p / 'initial_prompt.txt').write_text(schedule[0][0] + '\n')
    (p / 'config.yaml').write_text(yaml.safe_dump(pconfig, sort_keys=False))
    pm = {**bm, 'phase': 'ns280_suffix_literal_cue_only_control', 'gpu': 0,
        'command': [sys.executable, '-u', 'inference.py', '--config_path', str(p / 'config.yaml')],
        'config_sha256': sha(p / 'config.yaml'), 'study_plan_sha256': sha(root / 'plan.json'),
        'cue_intervention': cue_record, 'quality_claim': False, 'correction_enabled': False}
    write(p / 'manifest.json', pm)
    shutil.copy2(adapter / 'run_prepared.py', p / 'supervisor.py')
    for out in [p, rp]:
        shutil.copy2(root / 'plan.json', out / 'plan.json')
    write(root / 'preparation.json', {'status': 'prepared_unlaunched', 'wall_seconds': time.monotonic() - began,
        'parent_validation_command': command, 'new_video_generations': 0,
        'canonical_repair_preparer_sha256': sha(adapter / 'prepare_repair.py')})
    print(json.dumps({'status': 'prepared_unlaunched', 'root': str(root), 'seconds': time.monotonic() - began}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'baseline', 'local-repair', 'mask', 'checkout', 'adapter']:
        parser.add_argument('--' + key, type=Path, required=True)
    prepare(parser.parse_args())
