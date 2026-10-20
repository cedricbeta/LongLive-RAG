"""Observe and finish the one existing prompt-control attempt without retry."""
import json
from pathlib import Path
import subprocess
import time

S = Path('/home/chendong/video-gen/LongLive-RAG')
F = Path(__file__).resolve().parent
started = time.monotonic()
state = {'phase': 'observing_existing_P20', 'started_unix': time.time(), 'new_generation_launches': 0}


def write():
    (F / 'finalize_queue_status.json').write_text(json.dumps(state, indent=2) + '\n')


def run(script, args, logname):
    with (F / logname).open('x') as log:
        subprocess.run([str(S / 'venv/bin/python'), str(S / 'scripts' / script), *args],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


try:
    write()
    while True:
        try:
            status = json.loads((F / 'P20_prompt/status.json').read_text())
        except (OSError, ValueError):
            status = {'status': 'observation_unavailable'}
        if status['status'] not in ('preflight', 'running', 'observation_unavailable'):
            break
        if time.monotonic() - started > 1900:
            raise TimeoutError('Observation deadline; inspect same handles without restarting')
        time.sleep(5)
    state['P20_prompt'] = status['status']
    if status['status'] != 'completed':
        raise RuntimeError('P20 terminated without completion; no retry')
    state['phase'] = 'checking_actual_prefix_and_exporting_full_video'
    write()
    run('audit_native_prompt_start_control.py', ['--folder', str(F / 'P20_prompt')], 'audit.log.txt')
    run('review_native_prompt_start_control.py', ['--root', str(F), '--arms',
        'B_original', 'P30_prompt', 'P20_prompt', 'RP_local_cue', '--output', str(F / 'four_way_review')],
        'review.log.txt')
    state.update(phase='completed', queue_seconds_including_overlap_wait=time.monotonic() - started)
    write()
except Exception as error:
    state.update(phase='finalization_failed_no_generation_retry', error=f'{type(error).__name__}: {error}',
                 queue_seconds_including_overlap_wait=time.monotonic() - started)
    write()
    raise
print(json.dumps(state))
