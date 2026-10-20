import os,sys,subprocess,time,json,signal
from pathlib import Path
out=Path(sys.argv[1]);launch=json.loads((out/'launch.json').read_text());started=time.monotonic()
with (out/'inference.log').open('x') as log:
 p=subprocess.Popen(launch['command'],cwd=launch['cwd'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONHASHSEED':'0'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 status={'status':'running','pid':p.pid,'started_unix':time.time(),'gpu':1};(out/'status.json').write_text(json.dumps(status))
 try:
  code=p.wait(timeout=2400);state='completed' if code==0 else 'execution_failed'
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=p.returncode;state='timeout'
 status.update(status=state,returncode=code,wall_seconds=time.monotonic()-started);status['gpu_allocated_seconds']=status['wall_seconds'];(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
