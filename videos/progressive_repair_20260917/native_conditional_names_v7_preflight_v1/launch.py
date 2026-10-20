#!/usr/bin/env python3
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parent
workspace = root.parents[2]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(path, value):
    tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)
plan, binding = read(root / 'plan.json'), read(root / 'source_binding.json')
assert sha(root / 'plan.json') == binding['plan_sha256']
assert sha(root / 'worker.py') == binding['worker_sha256'] and sha(Path(__file__)) == binding['launcher_sha256']
for name, digest in binding['source_sha256'].items(): assert sha(workspace / name) == digest, name
assert not (root / 'status.json').exists() and not (root / 'server').exists()
sys.path.insert(0, str(workspace / 'scripts'))
from run_official_vbench_long import admission
admitted = admission(plan['gpu'])
write(root / 'admission.json', admitted)
observer_log = (root / 'observer.log.txt').open('x')
observer = subprocess.Popen(['/home/chendong/miniconda3/envs/sglang/bin/python', str(workspace / 'scripts/monitor_native_gpu.py'),
    '--folder', str(root), '--output', str(root / 'telemetry'), '--interval', '1', '--max-wait', '1000'],
    stdout=observer_log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    until = time.monotonic() + 20
    while not (root / 'telemetry/samples.jsonl').exists() or not (root / 'telemetry/samples.jsonl').stat().st_size:
        assert observer.poll() is None and time.monotonic() < until
        time.sleep(.2)
except BaseException:
    if observer.poll() is None: observer.terminate()
    observer.wait(timeout=10); observer_log.close(); raise
started = time.monotonic()
status = {'status': 'preflight', 'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'gpu': plan['gpu'],
    'gpu_uuid': admitted['gpu_uuid'], 'observer_pid': observer.pid}
write(root / 'status.json', status)
server = worker = None; known = {}; logs = []

def snapshot():
    table = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(p.name)] = {'pid': int(p.name), 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'start_ticks': int(fields[19]), 'state': fields[0]}
        except (FileNotFoundError, ProcessLookupError): pass
    return table

def capture():
    table = snapshot(); current = {p.pid for p in [server, worker] if p is not None}
    while True:
        more = {pid for pid, row in table.items() if row['ppid'] in current} - current
        if not more: break
        current.update(more)
    known.update({pid: table[pid] for pid in current if pid in table})

try:
    server_cmd = [str(workspace / 'venv/bin/python'), str(workspace / 'scripts/launch_qwen3vl_server.py'),
        '--model', plan['model'], '--revision', plan['revision'], '--gpu', str(plan['gpu']), '--out', str(root / 'server'),
        '--mem-fraction-static', str(plan['server_mem_fraction_static']), '--context-length', str(plan['server_context_length']),
        '--max-running-requests', str(plan['server_max_running_requests']), '--startup-timeout', str(plan['server_startup_timeout_seconds'])]
    log = (root / 'server_launcher.log.txt').open('x'); logs.append(log)
    server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, 'HF_HUB_OFFLINE': '1'}, start_new_session=True)
    status.update(status='server_starting', server_launcher_pid=server.pid); write(root / 'status.json', status)
    while not (root / 'server/base_url.txt').exists():
        capture(); assert server.poll() is None, 'Server stopped before readiness'
        if time.monotonic() - started > plan['server_startup_timeout_seconds'] + 15: raise TimeoutError('Server readiness ceiling')
        time.sleep(2)
    status['server_startup_seconds'] = time.monotonic() - started
    write(root / 'model_launch.json', {'base_url': (root / 'server/base_url.txt').read_text().strip(),
        'server_pgid': server.pid, 'gpu_uuid': admitted['gpu_uuid'], 'command': server_cmd})
    log = (root / 'worker.log.txt').open('x'); logs.append(log)
    worker = subprocess.Popen([str(workspace / 'venv/bin/python'), '-u', str(root / 'worker.py')], stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}, start_new_session=True)
    status.update(status='running', pid=worker.pid); write(root / 'status.json', status)
    began = time.monotonic()
    while worker.poll() is None:
        capture()
        if time.monotonic() - began > plan['worker_timeout_seconds']: raise TimeoutError('Worker ceiling')
        time.sleep(2)
    status.update(status='completed' if worker.returncode == 0 else 'failed', returncode=worker.returncode,
        worker_supervised_seconds=time.monotonic() - began)
except BaseException as error:
    status.update(status='failed', error=type(error).__name__ + ': ' + str(error))
finally:
    capture()
    if (root / 'server').exists(): (root / 'server/STOP').touch()
    if server is not None:
        try: server.wait(timeout=25)
        except subprocess.TimeoutExpired: pass
    for sig, wait in [(signal.SIGTERM, 10), (signal.SIGKILL, 1)]:
        table = snapshot(); groups = {row['pgid'] for pid, row in known.items() if pid in table
            and row['start_ticks'] == table[pid]['start_ticks'] and row['pgid'] == table[pid]['pgid'] and table[pid]['state'] != 'Z'}
        assert os.getpgrp() not in groups
        for group in groups:
            try: os.killpg(group, sig)
            except ProcessLookupError: pass
        if groups: time.sleep(wait)
    for proc in [worker, server]:
        if proc is not None: proc.wait(timeout=5)
    table = snapshot(); live = [pid for pid, row in known.items() if pid in table
        and row['start_ticks'] == table[pid]['start_ticks'] and table[pid]['state'] != 'Z']
    write(root / 'owned_processes.json', {'captured': list(known.values()), 'live_owned_at_end': live})
    status.update(wall_seconds=time.monotonic() - started, finished_unix=time.time()); write(root / 'status.json', status)
    observer.wait(timeout=20); observer_log.close()
    for log in logs: log.close()
    assert not live, live
print(json.dumps(status), flush=True)
