#!/usr/bin/env python3
"""One sequential pair with frozen sources, two GPUs and owned-process cleanup."""
import hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

root = Path(__file__).resolve().parent
r, workspace = root.parent, root.parents[2]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)

plan = read(root / 'plan.json')
binding = read(root / 'source_binding.json')
dependency = Path(plan['prerequisite'])
dep = read(dependency / 'status.json')
assert dep['status'] in ('completed', 'failed'), 'Preflight must be terminal'
assert not Path('/proc', str(dep['supervisor_pid'])).exists()
assert not read(dependency / 'owned_processes.json')['live_owned_at_end']
preflight = read(dependency / 'result.json')
assert preflight['status'] in ('completed', 'completed_with_case_failures')
assert preflight['actual_http_posts'] <= 4
assert sha(dependency / 'result.json') == binding['compiler_preflight_result_sha256']
compiled = read(dependency / 'cases' / plan['prerequisite_case'] / 'status.json')
assert compiled['status'] == 'compiled' and compiled['readiness_gate'] == 'passed'

assert sha(root / 'plan.json') == binding['plan_sha256']
assert not (root / 'queue_status.json').exists()
batch = r / 'native_development_pilot12_batch3_v1'
assert sha(batch / 'plan.json') == plan['parent_batch_plan_sha256']
waiting = time.monotonic()
while not (batch / 'queue_status.json').exists():
    for name in ['ns200', 'ns300']:
        state = read(batch / name / 'baseline/status.json')
        if state['status'] == 'running':
            assert Path('/proc', str(state['pid'])).exists(), 'Inspect stopped parent; do not restart'
        else:
            assert state['status'] == 'completed', state
    if time.monotonic() - waiting > 1800:
        raise TimeoutError('Parent dependency wait expired; no model launch')
    time.sleep(5)
assert read(batch / 'queue_status.json')['status'] == 'completed'
assert read(root / 'assets_preflight.json')['status'] == 'passed'
parent = Path(plan['parent'])
audit = read(parent / 'audit.json')
assert audit['engineering_checks_passed'] and sha(parent / audit['video_path']) == audit['video_sha256']
sys.path.insert(0, str(workspace / 'scripts'))
from run_official_vbench_long import admission

def proc_table():
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            tail = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(path.name)] = {'pid': int(path.name), 'ppid': int(tail[1]),
                'pgid': int(tail[2]), 'start_ticks': int(tail[19]), 'state': tail[0]}
        except (FileNotFoundError, ProcessLookupError): pass
    return rows

def capture_descendants(roots, known):
    table = proc_table(); current = set(roots)
    while True:
        additions = {pid for pid, row in table.items() if row['ppid'] in current} - current
        if not additions: break
        current.update(additions)
    for pid in current:
        if pid in table: known[pid] = table[pid]

def stop_owned(known):
    # Recheck start time and group before signalling: never follow reused PIDs.
    for sig, delay in [(signal.SIGTERM, 10), (signal.SIGKILL, 1)]:
        table = proc_table()
        live = [row for pid, row in known.items() if pid in table
            and table[pid]['start_ticks'] == row['start_ticks'] and table[pid]['pgid'] == row['pgid']
            and table[pid]['state'] != 'Z']
        groups = {row['pgid'] for row in live}
        assert os.getpgrp() not in groups
        for group in groups:
            try: os.killpg(group, sig)
            except ProcessLookupError: pass
        if groups: time.sleep(delay)

results = []; queue_start = time.monotonic()
for arm in plan['arms']:
    for name, digest in binding['source_sha256'].items():
        assert sha(workspace / name) == digest, name
    folder = root / arm; folder.mkdir(exist_ok=False)
    server_dir = folder / 'server'
    observers = []; logs = []; known = {}; admitted = {}; server = worker = None
    manifest = {'gpu': 2, 'arm': arm, 'pair_plan_sha256': binding['plan_sha256']}
    write(folder / 'manifest.json', manifest)
    observer_setup_started = time.monotonic()
    try:
        for gpu in [0, 2]:
            target = folder / f'gpu{gpu}_target'; target.mkdir()
            write(target / 'manifest.json', {'gpu': gpu, 'purpose': 'Whole-arm device observation including shared judge/segmenter, renderer and idle periods'})
            write(target / 'plan.json', {'pair_plan_sha256': binding['plan_sha256'], 'gpu': gpu, 'arm': arm, 'observation_only': True})
            log = (folder / f'gpu{gpu}_observer.log.txt').open('x'); logs.append(log)
            observer = subprocess.Popen(['/home/chendong/miniconda3/envs/sglang/bin/python',
                str(workspace / 'scripts/monitor_native_gpu.py'), '--folder', str(target),
                '--output', str(folder / f'gpu{gpu}_telemetry'), '--interval', '2', '--max-wait', '3900'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            observers.append(observer)
        ready_until = time.monotonic() + 20
        for gpu in [0, 2]:
            samples = folder / f'gpu{gpu}_telemetry/samples.jsonl'
            while not samples.exists() or samples.stat().st_size == 0:
                assert all(p.poll() is None for p in observers) and time.monotonic() < ready_until
                time.sleep(.2)
    except BaseException as error:
        for observer in observers:
            if observer.poll() is None: observer.terminate()
            observer.wait(timeout=10)
        for log in logs: log.close()
        write(folder / 'observer_setup_failure.json', {'status': 'failed_before_model_launch', 'error': type(error).__name__ + ': ' + str(error), 'wall_seconds': time.monotonic() - observer_setup_started, 'model_loads': 0, 'model_calls': 0, 'observer_pids': [p.pid for p in observers]})
        raise
    write(folder / 'observer_launch.json', {'observer_pids': [p.pid for p in observers], 'setup_seconds': time.monotonic() - observer_setup_started})
    began = time.monotonic()
    status = {'status': 'preflight', 'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'gpu': 2}
    try:
        admitted = {str(gpu): admission(gpu) for gpu in [0, 2]}
        write(folder / 'admission.json', admitted)
        write(folder / 'status.json', status)
        server_cmd = [str(workspace / 'venv/bin/python'), str(workspace / 'scripts/launch_qwen3vl_server.py'),
            '--model', plan['judge_model'], '--revision', plan['judge_revision'], '--gpu', '2', '--out', str(server_dir),
            '--mem-fraction-static', str(plan['server_mem_fraction_static']), '--context-length', str(plan['server_context_length']),
            '--max-running-requests', '1', '--startup-timeout', str(plan['startup_timeout_seconds'])]
        log = (folder / 'server_launcher.log.txt').open('x'); logs.append(log)
        server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'HF_HUB_OFFLINE': '1'}, start_new_session=True)
        status.update(status='server_starting', server_launcher_pid=server.pid)
        write(folder / 'status.json', status)
        started_server = time.monotonic()
        while not (server_dir / 'base_url.txt').exists():
            capture_descendants([server.pid], known)
            assert server.poll() is None, 'Server exited before readiness'
            if time.monotonic() - started_server > plan['startup_timeout_seconds'] + 15:
                raise TimeoutError('Server startup ceiling')
            time.sleep(2)
        status['server_startup_seconds'] = time.monotonic() - started_server
        script = 'run_native_progressive.py' if arm == 'feedback' else 'run_native_fixed_noise_control.py'
        cache = Path.home() / '.cache/huggingface/hub'
        cmd = [str(workspace / 'venv/bin/python'), '-u', str(workspace / 'scripts' / script),
            '--parent', str(parent), '--output', str(folder / 'run'), '--checkout', str(r / 'official_longlive_rag_repair'),
            '--assets-manifest', str(r / 'native_progressive_assets.json'), '--generator-python', str(r / 'env_longlive_rag/bin/python'),
            '--sam2-checkpoint', str(cache / 'models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610/sam2.1_hiera_large.pt'),
            '--clipseg-model', str(cache / 'models--CIDAS--clipseg-rd64-refined/snapshots/999e0328d9e10b484360c477313983f9afdd7050'),
            '--server-dir', str(server_dir), '--base-url', (server_dir / 'base_url.txt').read_text().strip(),
            '--segment-gpu', '2', '--judge-gpu', '2', '--render-gpu', '0', '--judge-pgid', str(server.pid),
            '--share-diagnostic-judge-gpu', '--start-shot', str(plan['current_shots_requested'][0]), '--shot-count', '1', '--max-edits', str(plan['max_edits_per_arm']),
            '--lookahead', str(plan['lookahead_shots']), '--max-seconds', str(plan['max_seconds_per_arm']),
            '--render-timeout', str(plan['render_timeout_seconds']), '--noise-seed', str(plan['noise_seed_base']),
            '--compiler', plan['compiler'], '--compiler-schema-context', '--mixed-evidence-revision',
            '--target-expression', plan['target_expression']]
        log = (folder / 'worker.log.txt').open('x'); logs.append(log)
        worker = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '2', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}, start_new_session=True)
        status.update(status='running', pid=worker.pid)
        write(folder / 'launch.json', {'server_command': server_cmd, 'worker_command': cmd,
            'server_launcher_pid': server.pid, 'worker_pid': worker.pid, 'observer_pids': [p.pid for p in observers]})
        write(folder / 'status.json', status)
        before_worker = time.monotonic()
        while worker.poll() is None:
            capture_descendants([server.pid, worker.pid], known)
            if time.monotonic() - before_worker > plan['outer_worker_timeout_seconds']:
                raise TimeoutError('Outer worker ceiling including one in-flight render')
            time.sleep(2)
        status.update(status='completed' if worker.returncode == 0 else 'failed', returncode=worker.returncode,
            worker_supervised_seconds=time.monotonic() - before_worker)
    except BaseException as error:
        status.update(status='failed', error=type(error).__name__ + ': ' + str(error))
    finally:
        roots = [p.pid for p in [server, worker] if p is not None]
        capture_descendants(roots, known)
        if server_dir.exists(): (server_dir / 'STOP').touch()
        if server is not None:
            try: server.wait(timeout=25)
            except subprocess.TimeoutExpired: pass
        stop_owned(known)
        for process in [worker, server]:
            if process is not None:
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
        status.update(wall_seconds=time.monotonic() - began, finished_unix=time.time())
        write(folder / 'status.json', status)
        table = proc_table()
        live_owned = [pid for pid, row in known.items() if pid in table and table[pid]['start_ticks'] == row['start_ticks'] and table[pid]['state'] != 'Z']
        write(folder / 'owned_processes.json', {'captured': list(known.values()), 'live_owned_at_end': live_owned})
        for gpu in [0, 2]:
            write(folder / f'gpu{gpu}_target/status.json', {**status, 'gpu': gpu,
                'gpu_uuid': admitted.get(str(gpu), {}).get('gpu_uuid')})
        for observer in observers:
            try: observer.wait(timeout=20)
            except subprocess.TimeoutExpired:
                observer.terminate(); observer.wait(timeout=10)
        for log in logs: log.close()
        assert not live_owned, live_owned
    results.append({'arm': arm, **status})
    write(root / 'queue_status.json', {'status': 'running' if len(results) < len(plan['arms']) else 'completed',
        'arms': results, 'wall_seconds': time.monotonic() - queue_start, 'quality_acceptance': False})
    print(json.dumps({'arm': arm, 'status': status['status'], 'wall_seconds': status['wall_seconds']}), flush=True)
