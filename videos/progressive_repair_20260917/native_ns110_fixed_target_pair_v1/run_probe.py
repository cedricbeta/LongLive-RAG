#!/usr/bin/env python3
"""Run the two frozen arms sequentially on GPU 2, each at most once."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
r, workspace = root.parent, root.parents[2]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, d: p.write_text(json.dumps(d, indent=2) + '\n')
plan, binding = read(root / 'plan.json'), read(root / 'launch_binding.json')
assert sha(root / 'plan.json') == binding['plan_sha256']
for name, digest in binding['helper_sha256'].items():
    assert sha(root / name) == digest
for name, digest in read(root / 'source_binding.json')['source_sha256'].items():
    assert sha(workspace / name) == digest
assert read(root / 'preparation.json')['status'] == 'prepared_unlaunched'
assert read(root / 'mask_admission.json')['admitted'] is True
assert not (root / 'launch.json').exists()
started = time.monotonic()
records, outcomes = [], []
native_python = str(r / 'env_longlive_rag/bin/python')
audit_python = str(workspace / 'venv/bin/python')
observer_python = '/home/chendong/miniconda3/envs/sglang/bin/python'
for name in ['P20_prompt', 'R20_local']:
    folder = root / name / 'baseline'
    assert not (folder / 'status.json').exists()
    assert read(folder / 'manifest.json')['gpu'] == 2
    telemetry = root / (name + '_telemetry')
    observer_log = (root / (name + '_observer.log.txt')).open('x')
    observer = subprocess.Popen([observer_python, str(workspace / 'scripts/monitor_native_gpu.py'),
        '--folder', str(folder), '--output', str(telemetry), '--interval', '2', '--max-wait', '2000'],
        stdout=observer_log, stderr=subprocess.STDOUT, start_new_session=True)
    until = time.monotonic() + 15
    while not (telemetry / 'samples.jsonl').exists() or (telemetry / 'samples.jsonl').stat().st_size == 0:
        if observer.poll() is not None or time.monotonic() > until:
            if observer.poll() is None:
                observer.terminate()
            observer.wait(timeout=10)
            observer_log.close()
            raise RuntimeError('Observer readiness failed before this arm started: ' + name)
        time.sleep(.2)
    log = (folder / 'supervisor.log.txt').open('x')
    supervisor = subprocess.Popen([native_python, str(folder / 'supervisor.py'), str(folder)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'arm': name, 'gpu': 2, 'supervisor_pid': supervisor.pid, 'observer_pid': observer.pid}
    records.append(record)
    write(root / 'launch.json', records)
    print(json.dumps(record), flush=True)
    supervisor.wait()
    observer.wait(timeout=30)
    log.close()
    observer_log.close()
    status = read(folder / 'status.json')
    row = {'arm': name, 'generation_status': status['status'], 'generation_seconds': status['wall_seconds']}
    if status['status'] == 'completed':
        command = ([audit_python, str(workspace / 'scripts/audit_native_fast_repair.py'),
                    '--folder', str(folder), '--parent', plan['parent']] if name == 'R20_local' else
                   [audit_python, str(root / 'audit_prompt_control.py'), '--folder', str(folder)])
        with (folder / 'audit.log.txt').open('x') as audit_log:
            row['audit_returncode'] = subprocess.run(command, stdout=audit_log, stderr=subprocess.STDOUT).returncode
    outcomes.append(row)
    write(root / 'queue_status.json', {'status': 'running' if len(outcomes) < 2 else 'completed',
        'arms': outcomes, 'wall_seconds': time.monotonic() - started,
        'new_generation_attempts': len(records), 'remaining_generation_slots': 2 - len(records),
        'automatic_acceptance': False, 'quality_claim': False})
    print(json.dumps(row), flush=True)
