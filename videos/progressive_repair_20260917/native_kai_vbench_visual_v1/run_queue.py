#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
workspace = root.parents[2]
eval_python = root.parent / 'env_vbench_long/bin/python'
observer_python = '/home/chendong/miniconda3/envs/sglang/bin/python'
monitor = workspace / 'scripts/monitor_native_gpu.py'
audit = workspace / 'scripts/audit_vbench_custom_video.py'
started = time.monotonic()
records = []

def write(name, value):
    (root / name).write_text(json.dumps(value, indent=2) + '\n')

def launch(arm):
    folder = root / arm
    telemetry = root / (arm + '_telemetry')
    assert not (folder / 'status.json').exists()
    observer_log = (root / (arm + '_observer.log.txt')).open('x')
    observer = subprocess.Popen([observer_python, str(monitor), '--folder', str(folder),
        '--output', str(telemetry), '--interval', '1', '--max-wait', '1000'],
        stdout=observer_log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 15
    while not (telemetry / 'samples.jsonl').exists() or (telemetry / 'samples.jsonl').stat().st_size == 0:
        if observer.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('Observer was not ready for ' + arm)
        time.sleep(.2)
    log = (folder / 'supervisor.log.txt').open('x')
    worker = subprocess.Popen([str(eval_python), str(folder / 'supervisor.py'), '--folder', str(folder)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    row = {'arm': arm, 'supervisor_pid': worker.pid, 'observer_pid': observer.pid}
    records.append(row)
    write('launch.json', records)
    print(json.dumps(row), flush=True)
    return arm, worker, observer, log, observer_log

def finish(job):
    arm, worker, observer, log, observer_log = job
    worker.wait()
    observer.wait(timeout=20)
    log.close(); observer_log.close()
    status = json.loads((root / arm / 'status.json').read_text())
    records[[r['arm'] for r in records].index(arm)].update(terminal_status=status['status'], wall_seconds=status['wall_seconds'])
    write('launch.json', records)
    print(json.dumps({'arm': arm, 'terminal_status': status['status'], 'seconds': status['wall_seconds']}), flush=True)

jobs = [launch('B_original'), launch('RP_local_cue')]
for job in jobs:
    finish(job)
finish(launch('P20_prompt'))
for arm in ('B_original', 'RP_local_cue', 'P20_prompt'):
    folder = root / arm
    with (folder / 'audit.log.txt').open('x') as log:
        result = subprocess.run([str(eval_python), str(audit), '--folder', str(folder),
            '--telemetry', str(root / (arm + '_telemetry')), '--output', str(folder / 'audit.json')],
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Audit failed for ' + arm)
write('queue_status.json', {'status': 'completed', 'wall_seconds': time.monotonic() - started,
    'records': records, 'remaining_evaluation_budget': 0})
print(json.dumps({'queue_status': 'completed', 'seconds': time.monotonic() - started}), flush=True)
