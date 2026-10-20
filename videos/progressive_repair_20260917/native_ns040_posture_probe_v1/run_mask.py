#!/usr/bin/env python3
import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;workspace=root.parents[2]
assert not (root/'group_mask').exists()
command=[str(root.parent/'env_longlive_rag/bin/python'),'-u',str(workspace/'scripts/prepare_native_group_mask.py'),'--root',str(root)]
started=time.monotonic()
with (root/'mask.log.txt').open('x') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},start_new_session=True)
 (root/'mask_launch.json').write_text(json.dumps({'pid':child.pid,'command':command,'started_unix':time.time(),'timeout_seconds':600},indent=2)+'\n')
 try:code=child.wait(timeout=600)
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  record={'status':'timeout','pid':child.pid,'wall_seconds':time.monotonic()-started,'remaining_mask_attempts':0}
  (root/'mask_supervisor_status.json').write_text(json.dumps(record,indent=2)+'\n');raise
print((root/'group_mask/status.json').read_text(),flush=True)
