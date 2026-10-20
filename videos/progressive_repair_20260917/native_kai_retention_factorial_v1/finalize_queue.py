"""Finish the two existing runs; never launch or retry a generation."""
import json
from pathlib import Path
import subprocess
import time

S = Path('/home/chendong/video-gen/LongLive-RAG')
F = Path(__file__).resolve().parent
python = str(S / 'venv/bin/python')
read = lambda p: json.loads(p.read_text())
started = time.monotonic()
state = {'phase': 'observing_existing_P_RP', 'started_unix': time.time(), 'new_generation_launches': 0}


def write():
    (F / 'finalize_queue_status.json').write_text(json.dumps(state, indent=2) + '\n')


def run(script, args, log_name):
    with (F / log_name).open('x') as log:
        subprocess.run([python, str(S / 'scripts' / script), *args], stdout=log,
                       stderr=subprocess.STDOUT, check=True)


try:
    remaining = ['P_cue', 'RP_local_cue']
    write()
    while remaining:
        for name in remaining[:]:
            try:
                status = read(F / name / 'status.json')
            except (OSError, ValueError):
                continue
            if status['status'] in ('preflight', 'running'):
                continue
            state[name] = status['status']
            write()
            if status['status'] != 'completed':
                raise RuntimeError(name + ' terminal without completion; no retry')
            if name == 'P_cue':
                run('audit_native_retention_cue.py', ['--folder', str(F / name)], name + '_audit.log.txt')
            else:
                run('audit_native_fast_repair.py', ['--folder', str(F / name), '--parent',
                    str((F / 'B_original').resolve())], name + '_audit.log.txt')
            remaining.remove(name)
        if remaining:
            if time.monotonic() - started > 1900:
                raise TimeoutError('Observation budget elapsed; inspect same handles, no restart')
            time.sleep(5)
    state['phase'] = 'checking_actual_pair_and_exporting_full_video'
    write()
    run('audit_native_retention_pair.py', ['--root', str(F)], 'pair_check.log.txt')
    run('review_native_retention_fourway.py', ['--root', str(F), '--arms',
        'B_original', 'P_cue', 'R_local', 'RP_local_cue', '--output', str(F / 'four_way_review')],
        'four_way_review.log.txt')
    state.update(phase='completed', queue_seconds_including_overlap_wait=time.monotonic() - started)
    write()
except Exception as error:
    state.update(phase='finalization_failed_no_generation_retry', error=f'{type(error).__name__}: {error}',
                 queue_seconds_including_overlap_wait=time.monotonic() - started)
    write()
    raise
print(json.dumps(state))
