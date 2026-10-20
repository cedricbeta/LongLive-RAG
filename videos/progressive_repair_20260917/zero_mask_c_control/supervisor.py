import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,digest in m['source_files'].items():
    assert sha(Path(m['source_path'])/rel)==digest,rel
assert sha(root/'config.yaml')==m['config_sha256']
assert sha(root/'zero_mask.pt')==m['mask_sha256']
assert not (root/'status.json').exists()
lock=open('/tmp/longlive_feedback_render_gpu1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=78000,free
cmd=['/home/chendong/miniconda3/bin/python','inference.py','--config_path',str(root/'config.yaml')]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONUNBUFFERED':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True','LLV2_DUMP_LATENT_DIR':str(root/'raw_latents')}
started=time.monotonic()
with (root/'render.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=m['source_path'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status={'status':'running','pid':child.pid,'supervisor_pid':os.getpid(),'started_unix':time.time(),'command':cmd,'gpu':1}
    (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        code=child.wait(timeout=m['timeout_seconds'])
        status.update(status='completed' if code==0 else 'execution_failed',returncode=code)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status.update(status='timeout',returncode=child.returncode)
status['wall_seconds']=time.monotonic()-started
(root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
