#!/usr/bin/env python3
"""Run the one frozen CPU localization attempt with bounded process cleanup."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(__file__).resolve().parent
workspace = root.parents[2]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, binding = read(root / 'plan.json'), read(root / 'source_binding.json')
assert sha(root / 'plan.json') == binding['plan_sha256']
assert not (root / 'mask_supervision.json').exists() and not (root / 'garment_mask').exists()
for name, h in binding['source_sha256'].items():
    assert sha(workspace / name) == h
recipe = plan['mask_recipe']
command = [str(workspace / 'venv/bin/python'), str(workspace / 'adapters/official_longlive_rag/prepare_mask.py'),
    '--parent', plan['parent'], '--output', str(root / 'garment_mask'),
    '--model', '/home/chendong/.cache/huggingface/hub/models--CIDAS--clipseg-rd64-refined/snapshots/999e0328d9e10b484360c477313983f9afdd7050',
    '--segment', str(recipe['segment']), '--entity-phrase', recipe['entity_phrase'],
    '--part-phrase', recipe['part_phrase'], '--exclude-phrase', recipe['exclude_phrase'],
    '--part-context', recipe['part_context'], '--threshold', str(recipe['threshold'])]
started = time.monotonic()
with (root / 'mask_worker.log.txt').open('x') as log:
    child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'status': 'running', 'supervisor_pid': os.getpid(), 'pid': child.pid,
              'started_unix': time.time(), 'command': command, 'source_sha256': sha(Path(__file__))}
    (root / 'mask_supervision.json').write_text(json.dumps(record, indent=2) + '\n')
    try:
        result = child.wait(timeout=recipe['cpu_timeout_seconds'])
        record.update(status='completed' if result == 0 else 'failed', returncode=result)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        record.update(status='timeout', returncode=child.returncode)
    finally:
        record['wall_seconds'] = time.monotonic() - started
        (root / 'mask_supervision.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
