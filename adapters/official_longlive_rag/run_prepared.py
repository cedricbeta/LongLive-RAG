#!/usr/bin/env python3
"""Bounded single-GPU runner with frozen-source and compute-process admission."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

out = Path(sys.argv[1]).resolve()
manifest = json.loads((out / 'manifest.json').read_text())
if (out / 'status.json').exists():
    raise FileExistsError('Attempt already recorded; do not restart it silently')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
started = time.monotonic()
status = {'status': 'preflight', 'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'gpu': manifest['gpu']}
child = None
try:
    if sha(out / 'config.yaml') != manifest['config_sha256'] or not all(
            sha(Path(manifest['cwd']) / p) == digest for p, digest in manifest['source_sha256'].items()):
        raise ValueError('Frozen source/configuration changed')
    lock = open('/tmp/longlive_feedback_render_gpu' + str(manifest['gpu']) + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(manifest['gpu']), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    def compute_pids():
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        return [int(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(manifest['gpu']), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    observed = compute_pids()
    status.update(gpu_uuid=uuid, admission_free_mib=free, admission_compute_pids=observed)
    if observed or free < 78000:
        status['status'] = 'resource_refused'
    else:
        with (out / 'inference.log').open('x') as log:
            child = subprocess.Popen(manifest['command'], cwd=manifest['cwd'],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(manifest['gpu']), 'PYTHONHASHSEED': '0'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status='running', pid=child.pid)
            (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
            stop_reason = None
            while child.poll() is None:
                if time.monotonic() - started > manifest['timeout_seconds']:
                    stop_reason = 'timeout'
                else:
                    try:
                        other = [pid for pid in compute_pids() if pid != child.pid]
                    except (subprocess.CalledProcessError, OSError, ValueError) as error:
                        # A failed observation is not evidence that the child
                        # exited. Keep its handle and poll again next interval.
                        other = []
                        status['resource_observation_failures'] = status.get('resource_observation_failures', 0) + 1
                        status['last_observation_error'] = f'{type(error).__name__}: {error}'
                    if other:
                        stop_reason = 'resource_interrupted'
                        status['other_compute_pids_observed'] = other
                if stop_reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(5)
            status.update(status=stop_reason or ('completed' if child.returncode == 0 else 'execution_failed'),
                          returncode=child.returncode)
except Exception as error:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        status['returncode'] = child.returncode
    status.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    status['wall_seconds'] = time.monotonic() - started
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
