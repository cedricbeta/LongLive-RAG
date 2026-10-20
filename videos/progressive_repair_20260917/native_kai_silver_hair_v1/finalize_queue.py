"""Observe one already launched candidate, then audit and export all 60 seconds."""
import json
import subprocess
import time
from pathlib import Path

S = Path('/home/chendong/video-gen/LongLive-RAG')
F = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
started = time.monotonic()
state = {'phase': 'waiting_existing_candidate', 'new_generation_launches': 0,
         'started_unix': time.time()}


def write():
    (F / 'finalize_queue_status.json').write_text(json.dumps(state, indent=2) + '\n')


write()
while True:
    p = F / 'candidate/status.json'
    if p.exists():
        status = read(p)
        if status['status'] not in ('running', 'preflight'):
            break
    if time.monotonic() - started > 1900:
        raise TimeoutError('Bounded wait exceeded; no restart')
    time.sleep(5)
state['generation_status'] = status['status']
write()
if status['status'] != 'completed':
    raise RuntimeError('Candidate did not complete; no retry')
with (F / 'audit.log.txt').open('x') as log:
    subprocess.run([str(S / 'venv/bin/python'), str(S / 'scripts/audit_native_fast_repair.py'),
        '--folder', str(F / 'candidate'), '--parent', read(F / 'plan.json')['parent']],
        stdout=log, stderr=subprocess.STDOUT, check=True)
audit = read(F / 'candidate/audit.json')
assert audit['strict_prefix_latents'] == 81 and audit['prefix_latents_exact']
assert audit['edited_range'] == [81, 121] and audit['additional_denoiser_calls'] == 56
state['phase'] = 'exporting_complete_paired_video'
state['nonempty_prefix_latents_exact'] = True
write()
with (F / 'review.log.txt').open('x') as log:
    subprocess.run([str(S / 'venv/bin/python'), str(S / 'scripts/review_native_fixed_target.py'),
        '--root', str(F)], stdout=log, stderr=subprocess.STDOUT, check=True)
state.update(phase='completed', queue_seconds_including_overlap_wait=time.monotonic() - started)
write()
print(json.dumps(state))
