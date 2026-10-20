#!/usr/bin/env python3
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
r=root.parent;workspace=r.parents[1]
provision=r/'official_narrastream_planner_provision_v1'
wait_started=time.monotonic()
while True:
    state=json.loads((provision/'status.json').read_text())
    if state['status']=='completed':break
    if state['status']=='failed':raise RuntimeError('Provisioning failed; model preflight not launched')
    assert Path('/proc',str(state['pid'])).exists(),'Provisioner no longer live; do not restart'
    if time.monotonic()-wait_started>1900:raise TimeoutError('Provision wait budget ended; no restart')
    time.sleep(5)
assert json.loads((provision/'cpu_import_config_preflight_v2.json').read_text())['status']=='passed'
metadata=json.loads((provision/'assets_manifest.json').read_text())
pilot=json.loads((r/'native_development_pilot12_v1/plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checkout=r/'official_narrastream_clean_v1'
plan={'protocol':'official_named_planner_local_two_request_preflight_v1','frozen_unix':time.time(),
    'gpu':0,'timeout_seconds':600,'model':metadata['repository'],'revision':metadata['revision'],
    'checkpoint':metadata['cache'],'assets_manifest_sha256':sha(provision/'assets_manifest.json'),
    'official_checkout':str(checkout),'official_revision':'ef23a2a81df5201c8640e9be8b068ebeacd4fd23',
    'extractor_source_sha256':sha(checkout/'narrastream_bench/utils/entity_extraction.py'),
    'worker_source_sha256':sha(root/'worker.py'),'cases':[{'id':c['id'],'shot_index':1,'prompt':c['prompts'][1]}
        for c in pilot['cases'] if c['index'] in (40,130)],
    'max_requests':2,'max_new_tokens':256,'temperature':0,'enable_thinking':False,'json_constrained_decoding':False,
    'failure_policy':'One model load and two fixed calls, no retry or changed prompt. No video scoring or acceptance.'}
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(root/'manifest.json').write_text(json.dumps({'gpu':0,'phase':plan['protocol']},indent=2)+'\n')
sys.path.insert(0,str(workspace/'scripts'))
from run_official_vbench_long import admission
observer_log=(root/'observer.log.txt').open('x')
observer=subprocess.Popen(['/home/chendong/miniconda3/envs/sglang/bin/python',str(workspace/'scripts/monitor_native_gpu.py'),
    '--folder',str(root),'--output',str(r/'official_narrastream_planner_preflight_v1_telemetry'),'--interval','1','--max-wait','700'],
    stdout=observer_log,stderr=subprocess.STDOUT,start_new_session=True)
until=time.monotonic()+15
samples=r/'official_narrastream_planner_preflight_v1_telemetry/samples.jsonl'
while not samples.exists() or not samples.stat().st_size:
    assert observer.poll() is None and time.monotonic()<until
    time.sleep(.2)
started=time.monotonic();status={'status':'preflight','supervisor_pid':os.getpid(),'started_unix':time.time(),'gpu':0}
child=None
try:
    status.update(admission(0))
    with (root/'worker.log.txt').open('x') as log:
        child=subprocess.Popen([str(r/'env_narrastream_planner_v1/bin/python'),'-u',str(root/'worker.py')],
            stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},start_new_session=True)
        status.update(status='running',pid=child.pid)
        (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        code=child.wait(timeout=600)
        status.update(status='completed' if code==0 else 'failed',returncode=code)
except BaseException as error:
    status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time())
    (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
observer.wait(timeout=20);observer_log.close()
print(json.dumps(status),flush=True)
