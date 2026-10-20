#!/usr/bin/env python3
"""Prepare this fixed-target pair from a regenerated ns010 parent; do not render."""
import argparse, hashlib, json, shutil, subprocess, time
from pathlib import Path
import yaml

p = argparse.ArgumentParser(description=__doc__)
for name in ['repo', 'parent', 'output', 'repair-checkout', 'python']:
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--mask', type=Path, help='Optional mask regenerated with the frozen program recipe for this parent')
p.add_argument('--archive', type=Path, help='Optional explicit location of the published plan and mask')
a = p.parse_args()
repo, parent, output = a.repo.resolve(), a.parent.resolve(), a.output.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')
started = time.monotonic()
assert not output.exists()
archive = a.archive.resolve() if a.archive else repo / 'results/native_ns010_fixed_target_pair_v1'
plan = read(archive / 'plan.json')
manifest, audit = read(parent / 'manifest.json'), read(parent / 'audit.json')
config = yaml.safe_load((parent / 'config.yaml').read_text())
assert read(parent / 'status.json')['status'] == 'completed' and audit['engineering_checks_passed']
assert manifest['source_benchmark_index'] == 10 and config['seed'] == 0 and 'repair_request' not in config
assert [x['prompts'][0] for x in config['timed_segments']] == plan['authored_prompts']
assert config['timed_segments'][0]['prompts'] == [plan['source_prompt']]
assert all(x['seconds'] == 10 for x in config['timed_segments'])
assert sha(parent / 'config.yaml') == manifest['config_sha256']
for name, digest in manifest['source_sha256'].items():
    assert sha(Path(manifest['cwd']) / name) == digest, name
mask = (a.mask or archive / 'garment_mask/mask.pt').resolve()
output.mkdir()
prompt = output / 'prompt.txt'; prompt.write_text(plan['target_expression'] + '\n')
repair = output / 'R0_local/baseline'
command = [str(a.python.absolute()), str(repo / 'adapters/official_longlive_rag/prepare_repair.py'),
    '--parent', str(parent), '--parent-branch', 'native_ns010_seed0', '--checkout', str(a.repair_checkout.resolve()),
    '--output', str(repair), '--assets-manifest', str(archive / 'assets_manifest.json'), '--mask', str(mask),
    '--prompt', str(prompt), '--noise-seed', '4243', '--gpu', '2', '--timeout', '1800']
with (output / 'prepare_repair.log.txt').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
assert read(repair / 'manifest.json')['action']['latent_range'] == [0, 41]
control = output / 'P0_prompt/baseline'; control.mkdir(parents=True)
config['timed_segments'][0]['prompts'] = [plan['target_expression']]
config.update(data_path=str(control / 'initial_prompt.txt'), output_folder=str(control / 'videos'),
    adapter_diagnostics_dir=str(control / 'diagnostics'))
(control / 'initial_prompt.txt').write_text(plan['target_expression'] + '\n')
(control / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
child = {**manifest, 'phase': 'fixed_target_first_window_prompt_control', 'gpu': 2,
    'command': [str(a.python.absolute()), '-u', 'inference.py', '--config_path', str(control / 'config.yaml')],
    'config_sha256': sha(control / 'config.yaml'), 'timeout_seconds': 1800,
    'fixed_target': True, 'automatic_acceptance': False, 'candidate_retained': False, 'quality_claim': False}
write(control / 'manifest.json', child)
shutil.copyfile(repo / 'adapters/official_longlive_rag/run_prepared.py', control / 'supervisor.py')
for folder in [control, repair]:
    write(folder / 'plan.json', {**plan, 'parent': str(parent), 'published_plan_sha256': sha(archive / 'plan.json'),
        'parent_manifest_sha256': sha(parent / 'manifest.json'), 'parent_video_sha256': audit['video_sha256'],
        'parent_latents_sha256': audit['latent_sha256'],
        'parent_timing_sha256': sha(parent / 'diagnostics/timing_0000.json'),
        'reproduction_parent_manifest_sha256': sha(parent / 'manifest.json'), 'reproduction_mask_sha256': sha(mask)})
    bound = read(folder / 'manifest.json')
    bound.update(fixed_target_probe_plan_sha256=sha(folder / 'plan.json'),
        automatic_diagnosis_bypassed=True, candidate_retained=False, quality_claim=False)
    write(folder / 'manifest.json', bound)
write(output / 'preparation.json', {'status': 'prepared_unlaunched', 'wall_seconds': time.monotonic() - started,
    'published_plan_sha256': sha(archive / 'plan.json'), 'parent_manifest_sha256': sha(parent / 'manifest.json'),
    'mask_sha256': sha(mask), 'command': command, 'source_sha256': sha(Path(__file__)),
    'new_generation_attempts': 0, 'automatic_acceptance': False})
print('Prepared exactly one P0 and one R0 attempt; no GPU generation started.')
