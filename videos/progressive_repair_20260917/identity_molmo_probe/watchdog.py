import json
import os
from pathlib import Path
import signal
import time

root = Path(__file__).resolve().parent
launch = json.loads((root / 'launch.json').read_text())
pid = launch['pid']
deadline = launch['started_unix'] + launch['timeout_seconds']
result = {'pid': pid, 'deadline_unix': deadline, 'watchdog_pid': os.getpid()}
while True:
    status_file = root / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    proc = Path('/proc') / str(pid)
    if status.get('status') in ['completed', 'execution_failed'] or not proc.exists():
        result['termination'] = status.get('status', 'process_exited_without_status')
        break
    if time.time() >= deadline:
        command = (proc / 'cmdline').read_bytes()
        if b'run_identity_contract_probe.py' not in command or os.getpgid(pid) != pid:
            raise RuntimeError('Refusing to signal an unexpected process')
        os.killpg(pid, signal.SIGTERM)
        time.sleep(10)
        if proc.exists():
            os.killpg(pid, signal.SIGKILL)
        status.update(status='timeout', timeout_seconds=launch['timeout_seconds'])
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        result['termination'] = 'timeout'
        break
    time.sleep(5)
result['observed_end_unix'] = time.time()
(root / 'supervisor_result.json').write_text(json.dumps(result, indent=2) + '\n')
