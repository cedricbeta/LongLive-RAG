#!/usr/bin/env python3
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;workspace=root.parents[2]
started=time.monotonic();python=str(workspace/'venv/bin/python')
while True:
 states={name:json.loads((root/name/'status.json').read_text()) for name in ('P_prompt','R_group')}
 if all(s['status'] not in ('running','preflight') for s in states.values()):break
 for name,s in states.items():
  if s['status']=='running':assert Path('/proc',str(s['pid'])).exists(),name+' worker absent; inspect instead of restarting'
 if time.monotonic()-started>1950:raise TimeoutError('Finalizer observation budget; generation supervisors remain authoritative')
 time.sleep(5)
assert all(s['status']=='completed' for s in states.values()),states
commands=[
 [python,str(workspace/'scripts/audit_native_fast_repair.py'),'--folder',str(root/'R_group'),'--parent',str(root/'B_original')],
 [python,str(workspace/'scripts/audit_native_posture_probe.py'),'--root',str(root)],
 [python,str(workspace/'scripts/review_native_posture_probe.py'),'--root',str(root),'--arms','B_original','P_prompt','R_group','--output',str(root/'review')]]
for index,command in enumerate(commands):
 with (root/f'finalize_{index}.log.txt').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Completed',index,flush=True)
(root/'finalize_status.json').write_text(json.dumps({'status':'completed','wall_seconds_including_wait':time.monotonic()-started,'commands':commands,'new_generations_started':0},indent=2)+'\n')
print('Complete comparison available; semantic inspection and export pending.',flush=True)
