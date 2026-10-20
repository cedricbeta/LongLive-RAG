import json,os,signal,subprocess,time
from pathlib import Path
out=Path(__import__('sys').argv[1]); m=json.loads((out/'launch.json').read_text()); pid=m['pid']; begun=time.monotonic(); result={'resource_observation_failures':0}
while True:
 try:
  if os.getpgid(pid)!=pid: result['reason']='launcher_exited';break
 except ProcessLookupError: result['reason']='launcher_exited';break
 if (out/'EXITED').exists(): result['reason']='server_exited';break
 try:
  output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15); other=[]
  for line in output.splitlines():
   uuid,p=line.split(','); p=int(p)
   if uuid.strip()!=m['gpu_uuid']: continue
   try:
    if os.getpgid(p)!=pid: other.append(p)
   except ProcessLookupError: pass
  if other:
   result.update(reason='other_compute_process_observed',other_pids=other);os.killpg(pid,signal.SIGTERM);break
 except (subprocess.TimeoutExpired,subprocess.CalledProcessError): result['resource_observation_failures']+=1
 if time.time()-m['started_unix']>600:
  result['reason']='bounded_server_lifetime';os.killpg(pid,signal.SIGTERM);break
 time.sleep(5)
result['wall_seconds_since_launch']=time.time()-m['started_unix'];(out/'watchdog_result.json').write_text(json.dumps(result,indent=2)+'\n')
