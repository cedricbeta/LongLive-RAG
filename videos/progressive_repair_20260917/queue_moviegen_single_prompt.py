"""Queue the frozen duration pilot after both native comparison arms."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/chendong/video-gen/LongLive-RAG')
runtime = root / 'videos/progressive_repair_20260917'
before = runtime / 'native_iris_fixed_noise_v1'
out = runtime / 'official_moviegen_000_120s_v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(state):
    temporary = out / 'queue_status.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(out / 'queue_status.json')


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def released():
    worker = before / 'worker_launch.json'
    if worker.exists():
        status = before / 'run/status.json'
        if not status.exists() or read(status)['status'] == 'running' or alive(read(worker)['pid']):
            return False
    elif read(before / 'queue_status.json')['status'] != 'failed':
        return False
    server = before / 'server_launch.json'
    if server.exists() and alive(read(server)['pid']):
        return False
    uuids = set(subprocess.check_output(['nvidia-smi', '-i', '0,1,2', '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).split())
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    return not any(line.split(',')[0].strip() in uuids for line in apps.splitlines())


state = {'status': 'waiting_fixed_control', 'pid': os.getpid(), 'pgid': os.getpgrp(),
         'started_unix': time.time()}
write(state)
try:
    deadline = time.monotonic() + 10800
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Control/resources did not finish within the queue budget')
        try:
            if released():
                break
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
        time.sleep(15)
    plan = read(out / 'plan.json')
    for name, expected in plan['source_sha256'].items():
        assert sha(root / name) == expected, name
    for name, expected in plan['runtime_files_sha256'].items():
        assert sha(out / name) == expected, name
    assert not (out / 'status.json').exists() and not (out / 'launch.json').exists()
    command = [read(out / 'manifest.json')['command'][0], str(out / 'supervisor.py'), str(out)]
    with (out / 'supervisor.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    launch = {'pid': process.pid, 'pgid': process.pid, 'command': command, 'started_unix': time.time()}
    (out / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    state.update(status='launched', supervisor_pid=process.pid, finished_unix=time.time())
except Exception as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
    raise
finally:
    write(state)
