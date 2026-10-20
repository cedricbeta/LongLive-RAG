import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[1]).resolve();m=json.loads((out/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/'status.json').exists()
assert sha(out/'config.yaml')==m['config_sha256']
assert all(sha(Path(m['cwd'])/n)==h for n,h in m['source_sha256'].items())
lock=open('/tmp/longlive_feedback_render_gpu'+str(m['gpu'])+'.lock','a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
free=int(subprocess.check_output(['nvidia-smi','-i',str(m['gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=78000,free
started=time.monotonic()
with (out/'inference.log').open('x') as log:
 p=subprocess.Popen(m['command'],cwd=m['cwd'],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(m['gpu']),'PYTHONHASHSEED':'0'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status={'status':'running','pid':p.pid,'supervisor_pid':os.getpid(),'started_unix':time.time(),'gpu':m['gpu']};(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 try:code=p.wait(timeout=m['timeout_seconds']);state='completed' if code==0 else 'execution_failed'
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=p.returncode;state='timeout'
 status.update(status=state,returncode=code,wall_seconds=time.monotonic()-started)
 (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
