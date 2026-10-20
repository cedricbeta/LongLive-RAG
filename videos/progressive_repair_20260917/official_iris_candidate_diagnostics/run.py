import json,subprocess,time,sys,os
from pathlib import Path
r=Path(sys.argv[1]);o=r/'official_iris_candidate_diagnostics';server=r/'official_iris_candidate_judge_server';start=time.monotonic();steps=[]
(o/'status.json').write_text(json.dumps({'status':'running','pid':os.getpid(),'started_unix':time.time()},indent=2)+'\n')
try:
 while not (server/'base_url.txt').exists():
  if (server/'EXITED').exists():raise RuntimeError('Judge server exited before readiness')
  if time.monotonic()-start>300:raise TimeoutError('Judge startup deadline')
  time.sleep(3)
 base=(server/'base_url.txt').read_text().strip();py=str(Path('venv/bin/python').absolute())
 cases=[('candidate_current','official_iris_garment_candidate',2),('parent_later','official_iris_60s_baseline_prepared',4),('candidate_later','official_iris_garment_candidate',4)]
 commands=[(name,[py,'scripts/probe_official_iris_attributes.py','--run',str(r/run),'--segment',str(seg),'--output',str(o/name),'--base-url',base]) for name,run,seg in cases]
 commands += [(name,[py,'scripts/probe_official_iris_protection.py','--parent',str(r/'official_iris_60s_baseline_prepared'),'--candidate',str(r/'official_iris_garment_candidate'),'--segment',str(seg),'--output',str(o/name),'--base-url',base]) for name,seg in [('current_guard',2),('later_guard',4)]]
 for name,cmd in commands:
  if (server/'EXITED').exists():raise RuntimeError('Judge server exited')
  t=time.monotonic()
  with (o/(name+'.log')).open('w') as log:
   result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=210)
  steps.append({'name':name,'command':cmd,'returncode':result.returncode,'wall_seconds':time.monotonic()-t})
  (o/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
  if result.returncode:raise RuntimeError('Diagnostic failed: '+name)
 status={'status':'completed'}
except Exception as error:
 status={'status':'execution_failed','error':f'{type(error).__name__}: {error}'}
finally:
 (server/'STOP').touch();status.update(steps=steps,wall_seconds=time.monotonic()-start,candidate_retained=False);(o/'status.json').write_text(json.dumps(status,indent=2)+'\n')
