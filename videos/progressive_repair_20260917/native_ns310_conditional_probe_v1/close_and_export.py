#!/usr/bin/env python3
"""Preserve the interrupted original and close every conditional generation slot."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
folder = root / 'ns310/baseline'
status, queue, plan = [read(x) for x in [folder/'status.json',root/'original_queue_status.json',root/'plan.json']]
launch = read(root/'original_launch.json')
assert status['status'] == queue['status'] == 'resource_interrupted'
assert status['returncode'] == -15 and status['other_compute_pids_observed']
assert queue['original_attempts'] == 1 and queue['followup_generations_launched'] == 0
assert queue['audit_returncode'] is None
assert sha(root/'plan.json') == launch['plan_sha256'] == read(folder/'manifest.json')['source_case_plan_sha256']
for pid in [status['pid'], status['supervisor_pid'], launch['observer_pid']]:
    assert not Path('/proc',str(pid)).exists(), pid
assert not list(root.rglob('*.mp4')) and not list(root.rglob('mask.pt'))
assert not (folder/'audit.json').exists()
telemetry = read(root/'ns310_telemetry/summary.json')
assert read(root/'ns310_telemetry/target_terminal_status.json') == status
bindings = read(root/'original_source_binding.json')
for name, digest in bindings.items():
    assert sha(Path(name)) == digest, name
    if Path(name).parent.name == 'scripts':
        dest = root/'source/scripts'/Path(name).name
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists()
        shutil.copyfile(name,dest)
summary = {
    'status':'closed_original_resource_interrupted', 'case':'ns310',
    'plan_sha256':sha(root/'plan.json'), 'closed_unix':time.time(),
    'original_attempts':1, 'completed_originals':0, 'exported_videos':0,
    'new_mask_attempts':0, 'prompt_control_generations':0, 'local_repair_generations':0,
    'remaining_original_budget':0, 'remaining_prompt_budget':0, 'remaining_local_budget':0,
    'remaining_mask_budget':0, 'generation_seconds':status['wall_seconds'], 'queue_seconds':queue['wall_seconds'],
    'preparation_seconds':read(root/'preparation.json')['wall_seconds'],
    'portable_preparation_check_seconds':read(root/'portable_preparation_check.json')['wall_seconds'],
    'observed_foreign_compute_pids_supervisor':status['other_compute_pids_observed'],
    'observed_foreign_compute_pids_telemetry':telemetry['observed_foreign_compute_pids_inside'],
    'device_energy_bounds':telemetry['device_energy_bounds'],
    'energy_attribution':'Whole-device measurements include foreign work near the interruption and idle power; not exclusive task energy.',
    'target_assessment':'unassessed_no_video', 'quality_result':'unmeasured',
    'human_labels':0, 'independent_evaluation':False, 'automatic_acceptance':False,
    'automatic_retained_improvements':0, 'held_out':False, 'submission_ready':False,
    'conclusion':'Foreign processes appeared on GPU 2 after clean admission. The existing supervisor terminated only this generation. No original video was exported, so neither the target-admission rule nor localization was reached. Both conditional follow-up slots are closed without restart or replacement. This is an execution interruption, not a failed semantic repair or evidence about official ns310 quality.',
    'protected_paths':'Historical IDE showcase paths were not modified; old 5B evidence and closed prior official experiments are unchanged.'}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files, paths = {}, []
def bind(dest):
    rel = str(dest.relative_to(target))
    assert rel not in files
    files[rel] = {'sha256':sha(dest),'bytes':dest.stat().st_size}
    paths.append(rel)
for source in sorted(root.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts:
        continue
    assert source.suffix in ['.py','.json','.jsonl','.txt','.log','.yaml'], source
    if source.suffix in ['.json','.jsonl','.txt','.log']:
        assert not re.search(r'\bhf_[A-Za-z0-9]{20,}|X-Amz-(?:Signature|Credential)=|Authorization: Bearer ',source.read_text()),source
    dest = target/'results'/root.name/source.relative_to(root)
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists()
    shutil.copyfile(source,dest)
    bind(dest)
for name in ['prepare.py','README.md']:
    bind(target/'reproduction/native_ns310_conditional_probe'/name)
dest = target/'results'/root.name/'export_manifest.json'
assert not dest.exists()
dest.write_text(json.dumps({'files':files,'omitted_files':[],
    'scope':'One interrupted original, no output video, no mask, no prompt/local repair generation and no quality conclusion.'},indent=2)+'\n')
paths.append(str(dest.relative_to(target)))
(root.parent/'native_ns310_delivery_paths.json').write_text(json.dumps(sorted(paths),indent=2)+'\n')
print(json.dumps({'paths':len(paths),'bytes':sum(x['bytes'] for x in files.values()),'summary':summary}))
