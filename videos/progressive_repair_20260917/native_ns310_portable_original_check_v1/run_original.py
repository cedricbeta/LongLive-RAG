#!/usr/bin/env python3
"""Run only the frozen original; no conditional repair is launched here."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
workspace = Path('/home/chendong/video-gen/ICLR27-Longlive-agent/implementation')
folder = root / 'ns310/baseline'
telemetry = root / 'ns310_telemetry'
python = Path('/home/chendong/video-gen/LongLive-RAG/videos/progressive_repair_20260917/env_longlive_rag/bin/python')
observer_python = '/home/chendong/miniconda3/envs/sglang/bin/python'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (folder / 'status.json').exists()
assert not (root / 'original_queue_status.json').exists()
plan = json.loads((root / 'plan.json').read_text())
assert plan['new_original_attempt_ceiling'] == 1
bound = [root/'prepare_original.py', Path(__file__), folder/'supervisor.py',
         workspace/'scripts/monitor_native_gpu.py', workspace/'scripts/audit_native_fast_case.py']
(root/'original_source_binding.json').write_text(json.dumps({str(p):sha(p) for p in bound},indent=2)+'\n')
started = time.monotonic()
with (root/'ns310_observer.log.txt').open('x') as obslog, (folder/'supervisor.log.txt').open('x') as log:
    observer = subprocess.Popen([observer_python,str(workspace/'scripts/monitor_native_gpu.py'),
        '--folder',str(folder),'--output',str(telemetry),'--interval','2','--max-wait','2000'],
        stdout=obslog,stderr=subprocess.STDOUT,start_new_session=True)
    until = time.monotonic()+15
    while not (telemetry/'samples.jsonl').exists() or (telemetry/'samples.jsonl').stat().st_size == 0:
        if observer.poll() is not None or time.monotonic()>until:
            raise RuntimeError('Observer readiness failed')
        time.sleep(.2)
    worker = subprocess.Popen([str(python),str(folder/'supervisor.py'),str(folder)],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch = dict(supervisor_pid=worker.pid,observer_pid=observer.pid,plan_sha256=sha(root/'plan.json'))
    (root/'original_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch),flush=True)
    worker.wait()
    observer.wait(timeout=30)
status = json.loads((folder/'status.json').read_text())
audit_code = None
if status['status']=='completed':
    with (folder/'audit.log.txt').open('x') as log:
        audit_code = subprocess.run(['/home/chendong/video-gen/LongLive-RAG/venv/bin/python',str(workspace/'scripts/audit_native_fast_case.py'),
            '--folder',str(folder)],stdout=log,stderr=subprocess.STDOUT).returncode
record = {'status':status['status'],'wall_seconds':time.monotonic()-started,
          'generation_seconds':status['wall_seconds'],'audit_returncode':audit_code,
          'original_attempts':1,'followup_generations_launched':0}
(root/'original_queue_status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
