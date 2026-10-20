import json,os,signal,subprocess,time
from pathlib import Path
out=Path(__import__('sys').argv[1]);m=json.loads((out/'launch.json').read_text());pg=m['process_group'];result={'resource_observation_failures':0}
def alive():
 try: os.killpg(pg,0);return True
 except ProcessLookupError:return False
while alive() and not (out/'EXITED').exists():
 reason=None
 try:
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=15);other=[]
  for line in apps.splitlines():
   uuid,pid=line.split(',');pid=int(pid)
   if uuid.strip()!=m['gpu_uuid']:continue
   try:
    if os.getpgid(pid)!=pg:other.append(pid)
   except ProcessLookupError:pass
  if other:reason='resource_interrupted';result['other_compute_pids']=other
 except (subprocess.TimeoutExpired,subprocess.CalledProcessError):result['resource_observation_failures']+=1
 if time.time()-m['started_unix']>600:reason='server_deadline'
 if reason:
  result['reason']=reason
  if alive():os.killpg(pg,signal.SIGTERM)
  deadline=time.monotonic()+20
  while alive() and time.monotonic()<deadline:time.sleep(1)
  if alive():os.killpg(pg,signal.SIGKILL);result['forced_after20s']=True
  (out/'EXITED').write_text(reason+'\n');break
 time.sleep(5)
result.setdefault('reason','server_exited');result['wall_seconds_since_launch']=time.time()-m['started_unix'];(out/'watchdog_result.json').write_text(json.dumps(result,indent=2)+'\n')
