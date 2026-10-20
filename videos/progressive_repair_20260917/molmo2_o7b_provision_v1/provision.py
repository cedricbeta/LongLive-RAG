#!/usr/bin/env python3
import hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
metadata=json.loads((root/'public_metadata.json').read_text());assert metadata['gated'] is False
files=metadata['files'];expected=sum(x['size'] for x in files)
free=shutil.disk_usage(root).free;assert free-expected>25*2**30
assert not (root/'status.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,value):
 p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
plan={**metadata,'frozen_unix':time.time(),'expected_bytes':expected,'free_bytes_before':free,'minimum_free_reserve_bytes':25*2**30,
 'timeout_seconds':900,'max_workers':2,'transport':'HTTP with HF_HUB_DISABLE_XET=1','source_sha256':sha(Path(__file__)),
 'purpose':'Provision pinned public OLMo-based Molmo2-O-7B for a bounded cross-model evaluation preflight; not validated independent acceptance.',
 'failure_policy':'One bounded transfer attempt. Preserve partial files and failure; no retry or model substitution.',
 'GPU_jobs':0,'inference_calls':0}
write('plan.json',plan);started=time.monotonic();state={'status':'preflight','pid':os.getpid(),'started_unix':time.time()};child=None;write('status.json',state)
try:
 code="from huggingface_hub import snapshot_download; import json; from pathlib import Path; p=json.loads(Path('plan.json').read_text()); print(snapshot_download(p['model'],revision=p['revision'],allow_patterns=[f['name'] for f in p['files']],max_workers=2))"
 with (root/'download.log.txt').open('x') as log:
  child=subprocess.Popen([sys.executable,'-u','-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT,
   env={**os.environ,'HF_HUB_DISABLE_XET':'1'},start_new_session=True)
  state.update(status='downloading',download_pid=child.pid);write('status.json',state)
  while child.poll() is None:
   if time.monotonic()-started>900:raise TimeoutError('900-second provisioning budget exceeded')
   if shutil.disk_usage(root).free<25*2**30:raise RuntimeError('Free-space reserve reached; preserve partial download')
   time.sleep(5)
  if child.returncode:raise RuntimeError('Downloader exit '+str(child.returncode))
 snapshot=Path.home()/'.cache/huggingface/hub/models--allenai--Molmo2-O-7B/snapshots'/plan['revision']
 records=[]
 for row in files:
  p=snapshot/row['name'];assert p.stat().st_size==row['size'];value=sha(p)
  if row.get('lfs'):assert value==row['lfs']['sha256'],row['name']
  records.append({'file':row['name'],'bytes':row['size'],'sha256':value,'publisher_lfs_sha256':row.get('lfs',{}).get('sha256')})
 write('assets_manifest.json',{'model':plan['model'],'revision':plan['revision'],'snapshot':str(snapshot),'files':records,'model_loaded':False})
 state.update(status='completed',files_verified=len(records),bytes_verified=sum(x['bytes'] for x in records),snapshot=str(snapshot))
except BaseException as error:state.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 state.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());write('status.json',state)
print(json.dumps(state),flush=True)
