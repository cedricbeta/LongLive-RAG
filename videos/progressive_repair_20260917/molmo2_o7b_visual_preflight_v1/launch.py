#!/usr/bin/env python3
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import cv2
from PIL import Image

root=Path(__file__).resolve().parent;r=root.parent;workspace=r.parents[1]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(root/'plan.json');batch=r/'native_development_pilot12_batch2_v1';provision=r/'molmo2_o7b_provision_v1'
assert not (root/'status.json').exists()
assert sha(batch/'plan.json')==plan['batch_plan_sha256']
began=time.monotonic()
while True:
    state=read(provision/'status.json')
    cases={c:read(batch/c/'baseline/status.json') for c in plan['cases']}
    failed=state['status']=='failed' or any(s['status'] not in ('running','preflight','completed') for s in cases.values())
    if failed:
        (root/'dependency_abort.json').write_text(json.dumps({'status':'not_launched','model_loads':0,'requests':0,'provision_status':state,'case_statuses':cases},indent=2)+'\n')
        raise RuntimeError('A dependency failed; no model inference launched')
    if state['status']=='completed' and (batch/'queue_status.json').exists() and read(batch/'queue_status.json')['status']=='completed':break
    if state['status']!='completed':assert Path('/proc',str(state['pid'])).exists(),'Inspect missing provisioner, do not restart'
    for s in cases.values():
        if s['status']=='running':assert Path('/proc',str(s['pid'])).exists(),'Inspect missing generation worker, do not restart'
    if time.monotonic()-began>plan['dependency_wait_seconds']:raise TimeoutError('Dependency observation budget; no model launch')
    time.sleep(5)
assert read(provision/'cpu_preflight.json')['status']=='passed'
assets=read(provision/'assets_manifest.json');rows=[]
for case in plan['cases']:
    parent=batch/case/'baseline';video,=(parent/'videos').glob('*.mp4')
    audit=read(parent/'audit.json');assert sha(video)==audit['video_sha256']
    cap=cv2.VideoCapture(str(video));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==16
    frames=[]
    for frame in plan['frame_indices']:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
        dest=root/'frames'/case/f'{frame:04d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(dest)
        frames.append({'path':str(dest.relative_to(root)),'sha256':sha(dest),'frame':frame,'seconds':frame/16})
    cap.release();rows.append({'id':case,'video_sha256':audit['video_sha256'],'frames':frames})
(root/'inputs.json').write_text(json.dumps({'snapshot':assets['snapshot'],'assets_manifest_sha256':sha(provision/'assets_manifest.json'),
    'cases':rows,'worker_source_sha256':sha(root/'worker.py'),'launcher_source_sha256':sha(Path(__file__))},indent=2)+'\n')
sys.path.insert(0,str(workspace/'scripts'));from run_official_vbench_long import admission
observer_log=(root/'observer.log.txt').open('x')
observer=subprocess.Popen(['/home/chendong/miniconda3/envs/sglang/bin/python',str(workspace/'scripts/monitor_native_gpu.py'),
    '--folder',str(root),'--output',str(root/'telemetry'),'--interval','1','--max-wait','700'],stdout=observer_log,stderr=subprocess.STDOUT,start_new_session=True)
until=time.monotonic()+15
while not (root/'telemetry/samples.jsonl').exists() or not (root/'telemetry/samples.jsonl').stat().st_size:
    assert observer.poll() is None and time.monotonic()<until;time.sleep(.2)
started=time.monotonic();status={'status':'preflight','supervisor_pid':os.getpid(),'started_unix':time.time(),'gpu':0};child=None
try:
    status.update(admission(0))
    with (root/'worker.log.txt').open('x') as log:
        child=subprocess.Popen([str(r/'env_longlive_rag/bin/python'),'-u',str(root/'worker.py')],stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'},start_new_session=True)
        status.update(status='running',pid=child.pid);(root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        code=child.wait(timeout=plan['worker_timeout_seconds']);status.update(status='completed' if code==0 else 'failed',returncode=code)
except BaseException as error:status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());(root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
observer.wait(timeout=20);observer_log.close();print(json.dumps(status),flush=True)
