"""Observe the two existing baseline/control jobs; never launch a generator."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

S = Path('/home/chendong/video-gen/LongLive-RAG')
F = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
started = time.monotonic()
state = {'phase': 'waiting_existing_B_P', 'started_unix': time.time(), 'new_generation_launches': 0}
remaining = ['B_original', 'P_cue']


def write():
    (F / 'baseline_queue_status.json').write_text(json.dumps(state, indent=2) + '\n')


write()
while remaining:
    for name in remaining[:]:
        p = F / name / 'status.json'
        if not p.exists() or read(p)['status'] in ('running', 'preflight'):
            continue
        status = read(p)
        state[name] = status['status']
        write()
        if status['status'] != 'completed':
            raise RuntimeError(name + ' did not complete; no retry')
        with (F / (name + '_audit.log.txt')).open('x') as log:
            subprocess.run([str(S / 'venv/bin/python'), str(S / 'scripts/audit_native_cue_case.py'),
                '--folder', str(F / name)], stdout=log, stderr=subprocess.STDOUT, check=True)
        remaining.remove(name)
    if remaining:
        if time.monotonic() - started > 1900:
            raise TimeoutError('Bounded wait exceeded; no restart')
        time.sleep(5)
torch.set_num_threads(4)
a, b = [torch.load(F / n / 'diagnostics/latents_0000.pt', map_location='cpu', weights_only=True)
        for n in ['B_original', 'P_cue']]
assert torch.equal(a[:, :121], b[:, :121]), 'Shared prefix mismatch; stop causal attribution'
configs = [yaml.safe_load((F / n / 'config.yaml').read_text()) for n in ['B_original', 'P_cue']]
assert configs[0]['timed_segments'][:3] == configs[1]['timed_segments'][:3]
for c in configs:
    for k in ['data_path', 'output_folder', 'adapter_diagnostics_dir', 'timed_segments']:
        c.pop(k)
assert configs[0] == configs[1]
memory = [read(next((F / n / 'videos').glob('*memory_log.json'))) for n in ['B_original', 'P_cue']]
record = {'status': 'passed', 'shared_prefix_latents': 121, 'prefix_latents_exact': True,
    'changed_suffix_latents': [i for i in range(121, 241) if not torch.equal(a[:, i], b[:, i])],
    'retrieval_records_through_query121_equal': [x for x in memory[0] if x['query_frame'] <= 121]
        == [x for x in memory[1] if x['query_frame'] <= 121],
    'other_generation_settings_equal': True, 'raw_pixel_equality_test': False,
    'audits_sha256': {n: sha(F / n / 'audit.json') for n in ['B_original', 'P_cue']},
    'quality_claim': False}
(F / 'baseline_pair_check.json').write_text(json.dumps(record, indent=2) + '\n')
state['phase'] = 'exporting_B_P_full_video'
write()
with (F / 'baseline_review.log.txt').open('x') as log:
    subprocess.run([str(S / 'venv/bin/python'), str(S / 'scripts/review_native_cue_study.py'),
        '--root', str(F), '--arms', 'B_original', 'P_cue', '--output', str(F / 'baseline_review')],
        stdout=log, stderr=subprocess.STDOUT, check=True)
state.update(phase='completed', queue_seconds_including_overlap_wait=time.monotonic() - started)
write()
print(json.dumps(state))
