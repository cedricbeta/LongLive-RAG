"""Start the already frozen comparator only after the preceding jobs exit."""
import hashlib, json, os, subprocess, time
from pathlib import Path

root = Path('/home/chendong/video-gen/LongLive-RAG')
runtime = root / 'videos/progressive_repair_20260917'
before = runtime / 'native_iris_mixed_evidence_v1'
out = runtime / 'native_iris_fixed_noise_v1'
read = lambda p: json.loads(p.read_text())
def write(name, value):
    path = out / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
def alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    return True
def idle():
    ids = set(subprocess.check_output(['nvidia-smi', '-i', '0,1,2', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split())
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    return not any(line.split(',')[0].strip() in ids for line in rows.splitlines())

state = {'status': 'waiting_previous_trial', 'pid': os.getpid(), 'pgid': os.getpgrp(), 'started_unix': time.time()}
write('queue_status.json', state)
server = None
try:
    parent_worker = read(before / 'worker_launch.json')['pid']
    parent_server = read(before / 'server_launch.json')['pid']
    deadline = time.monotonic() + 7200
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Previous trial/resources did not become terminal and idle within queue budget')
        try:
            terminal = read(before / 'run/status.json')['status'] != 'running'
            if terminal and not alive(parent_worker) and not alive(parent_server) and idle():
                break
        except (OSError, ValueError, subprocess.CalledProcessError):
            # A failed observation does not authorize a launch or restart.
            pass
        time.sleep(15)
    plan = read(out / 'plan.json')
    for name, expected in plan['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    assert not (out / 'server_launch.json').exists() and not (out / 'worker_launch.json').exists()
    command = [str(out / 'server') if item == str(before / 'server') else item
               for item in read(before / 'server_launch.json')['command']]
    with (out / 'server_launcher.log').open('x') as log:
        server = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write('server_launch.json', {'pid': server.pid, 'pgid': server.pid, 'command': command,
                                'started_unix': time.time(), 'admission_compute_pids': []})
    state.update(status='waiting_server', server_pid=server.pid)
    write('queue_status.json', state)
    limit = time.monotonic() + 650
    while not (out / 'server/base_url.txt').exists():
        if server.poll() is not None or time.monotonic() >= limit:
            raise RuntimeError('Control server failed to become ready')
        time.sleep(5)
    command = read(before / 'worker_launch.json')['command']
    swaps = {str(before / 'run'): str(out / 'run'), str(before / 'server'): str(out / 'server'),
             str(root / 'scripts/run_native_progressive.py'): str(root / 'scripts/run_native_fixed_noise_control.py')}
    command = [swaps.get(item, item) for item in command]
    command[command.index('--base-url') + 1] = (out / 'server/base_url.txt').read_text().strip()
    command[command.index('--judge-pgid') + 1] = str(server.pid)
    with (out / 'worker.log').open('x') as log:
        worker = subprocess.Popen(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'PYTHONUNBUFFERED': '1'},
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write('worker_launch.json', {'pid': worker.pid, 'pgid': worker.pid, 'command': command,
                                'started_unix': time.time()})
    state.update(status='launched', worker_pid=worker.pid, finished_unix=time.time())
except Exception as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
    if server is not None and server.poll() is None:
        (out / 'server/STOP').touch()
    raise
finally:
    write('queue_status.json', state)
