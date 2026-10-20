#!/usr/bin/env python3
"""Prepare the two frozen capacity-probe arms; never start a GPU job here."""
import hashlib, json, shutil, subprocess, time
from pathlib import Path
import yaml

root = Path(__file__).resolve().parent
r, workspace = root.parent, root.parents[2]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')
began = time.monotonic()
plan = read(root / 'plan.json')
binding = read(root / 'source_binding.json')
assert sha(root / 'plan.json') == binding['plan_sha256']
for name, digest in binding['source_sha256'].items():
    assert sha(workspace / name) == digest, name
assert read(root / 'mask_supervision.json')['status'] == 'completed'
assert read(root / 'garment_mask/status.json')['status'] == 'completed'
assert read(root / 'garment_mask/result.json')['nonempty']
automatic = Path(plan['automatic_pair_folder'])
assert read(automatic / 'queue_status.json')['status'] == 'completed'
for arm in ['feedback', 'fixed_noise']:
    assert not read(automatic / arm / 'owned_processes.json')['live_owned_at_end']
    state = read(automatic / arm / 'status.json')
    assert not Path('/proc', str(state['supervisor_pid'])).exists()
parent = Path(plan['parent'])
manifest = read(parent / 'manifest.json')
assert sha(parent / 'manifest.json') == plan['parent_manifest_sha256']
assert sha(root / 'prompt.txt') == hashlib.sha256((plan['target_expression'] + '\n').encode()).hexdigest()
repair = root / 'R0_local/baseline'
repair.parent.mkdir(exist_ok=False)
command = [str(r / 'env_longlive_rag/bin/python'),
    str(workspace / 'adapters/official_longlive_rag/prepare_repair.py'),
    '--parent', str(parent), '--parent-branch', 'native_ns100_seed0',
    '--checkout', str(r / 'official_longlive_rag_repair'), '--output', str(repair),
    '--assets-manifest', str(root / 'assets_manifest.json'), '--mask', str(root / 'garment_mask/mask.pt'),
    '--prompt', str(root / 'prompt.txt'), '--noise-seed', '4243', '--gpu', '2', '--timeout', '1800']
with (root / 'prepare_repair.log.txt').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
child = read(repair / 'manifest.json')
assert child['action']['latent_range'] == [0, 41]
child.update(phase='fixed_target_local_capacity_probe', fixed_target_probe_plan_sha256=sha(root / 'plan.json'),
    automatic_diagnosis_bypassed=True, candidate_retained=False, quality_claim=False)
write(repair / 'manifest.json', child)
shutil.copyfile(root / 'plan.json', repair / 'plan.json')
prompt = root / 'P0_prompt/baseline'
prompt.mkdir(parents=True, exist_ok=False)
config = yaml.safe_load((parent / 'config.yaml').read_text())
assert 'repair_request' not in config and config['seed'] == 0
assert config['timed_segments'][0]['prompts'] == [plan['source_prompt']]
config['timed_segments'][0]['prompts'] = [plan['target_expression']]
config.update(data_path=str(prompt / 'initial_prompt.txt'), output_folder=str(prompt / 'videos'),
    adapter_diagnostics_dir=str(prompt / 'diagnostics'))
(prompt / 'initial_prompt.txt').write_text(plan['target_expression'] + '\n')
(prompt / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
child = {**manifest, 'phase': 'fixed_target_first_window_prompt_control', 'gpu': 0,
    'command': [str(r / 'env_longlive_rag/bin/python'), '-u', 'inference.py', '--config_path', str(prompt / 'config.yaml')],
    'config_sha256': sha(prompt / 'config.yaml'), 'timeout_seconds': 1800,
    'fixed_target_probe_plan_sha256': sha(root / 'plan.json'), 'automatic_diagnosis_bypassed': True,
    'candidate_retained': False, 'quality_claim': False, 'repair_enabled': False}
write(prompt / 'manifest.json', child)
shutil.copyfile(root / 'plan.json', prompt / 'plan.json')
shutil.copyfile(parent / 'supervisor.py', prompt / 'supervisor.py')
repaired_config = yaml.safe_load((repair / 'config.yaml').read_text())
assert repaired_config['repair_request']['prompts'] == config['timed_segments'][0]['prompts']
assert repaired_config['timed_segments'][1:] == config['timed_segments'][1:]
assert repaired_config['timed_segments'][0]['prompts'] == [plan['source_prompt']]
write(root / 'preparation.json', {'status': 'prepared_unlaunched', 'wall_seconds': time.monotonic() - began,
    'preparer_sha256': sha(Path(__file__)), 'repair_preparation_command': command,
    'plan_sha256': sha(root / 'plan.json'), 'mask_sha256': sha(root / 'garment_mask/mask.pt'),
    'same_effective_target_text': True, 'later_authored_prompts_unchanged': True,
    'new_generation_attempts_started': 0})
print(json.dumps(read(root / 'preparation.json')))
