#!/usr/bin/env python3
import json
from pathlib import Path
import subprocess
import time
root = Path(__file__).resolve().parent
workspace = Path('/home/chendong/video-gen/ICLR27-Longlive-agent/implementation')
python = Path('/home/chendong/video-gen/LongLive-RAG/videos/progressive_repair_20260917/env_longlive_rag/bin/python')
observer_python = '/home/chendong/miniconda3/envs/sglang/bin/python'
audit_python = Path('/home/chendong/video-gen/LongLive-RAG/venv/bin/python')
started = time.monotonic()
records = []
jobs = []
for name in ('ns010', 'ns070'):
    folder = root / name / 'baseline'
    telemetry = root / (name + '_telemetry')
    assert not (folder / 'status.json').exists()
    observer_log = (root / (name + '_observer.log.txt')).open('x')
    observer = subprocess.Popen([observer_python, str(workspace / 'scripts/monitor_native_gpu.py'),
        '--folder', str(folder), '--output', str(telemetry), '--interval', '2', '--max-wait', '2000'],
        stdout=observer_log, stderr=subprocess.STDOUT, start_new_session=True)
    until = time.monotonic() + 15
    while not (telemetry / 'samples.jsonl').exists() or (telemetry / 'samples.jsonl').stat().st_size == 0:
        if observer.poll() is not None or time.monotonic() > until:
            raise RuntimeError('Observer readiness failed: ' + name)
        time.sleep(.2)
    log = (folder / 'supervisor.log.txt').open('x')
    worker = subprocess.Popen([str(python), str(folder / 'supervisor.py'), str(folder)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    row = dict(case=name, supervisor_pid=worker.pid, observer_pid=observer.pid)
    records.append(row)
    (root / 'launch.json').write_text(json.dumps(records, indent=2) + '\n')
    jobs.append((name, worker, observer, log, observer_log))
    print(json.dumps(row), flush=True)
for name, worker, observer, log, observer_log in jobs:
    worker.wait()
    observer.wait(timeout=30)
    log.close(); observer_log.close()
    folder = root / name / 'baseline'
    status = json.loads((folder / 'status.json').read_text())
    print(json.dumps({'case': name, 'status': status['status'], 'seconds': status['wall_seconds']}), flush=True)
    if status['status'] == 'completed':
        with (folder / 'audit.log.txt').open('x') as audit_log:
            code = subprocess.run([str(audit_python), str(workspace / 'scripts/audit_native_fast_case.py'),
                '--folder', str(folder)], stdout=audit_log, stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError('Baseline audit failed: ' + name)
(root / 'queue_status.json').write_text(json.dumps({'status': 'completed',
    'wall_seconds': time.monotonic() - started, 'remaining_second_batch_generation_budget': 0,
    'new_repairs': 0, 'planned_scripts': 12, 'second_batch_attempted_scripts': 2}, indent=2) + '\n')
print('Second batch terminal; no repair or next batch launched.', flush=True)
