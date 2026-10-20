#!/usr/bin/env python3
import hashlib,json,os,shutil,time
from pathlib import Path
root=Path(__file__).resolve().parent;prior=root.parent/'molmo2_o7b_provision_v1'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,value):
 p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
old=read(prior/'status.json');assert old['status']=='failed' and old['error']=="AttributeError: 'NoneType' object has no attribute 'get'"
assert not Path('/proc',str(old['pid'])).exists() and not Path('/proc',str(old['download_pid'])).exists()
metadata=read(prior/'public_metadata.json');snapshot=Path.home()/'.cache/huggingface/hub/models--allenai--Molmo2-O-7B/snapshots'/metadata['revision']
plan={**metadata,'protocol':'existing_download_manifest_recovery_v2','frozen_unix':time.time(),'source_sha256':sha(Path(__file__)),
 'prior_status_sha256':sha(prior/'status.json'),'prior_plan_sha256':sha(prior/'plan.json'),
 'new_downloads':0,'new_GPU_jobs':0,'reason':'Use (lfs or {}) when recording ordinary non-LFS files. Verify all already-downloaded files; preserve v1 failure.'}
write('plan.json',plan);shutil.copyfile(prior/'status.json',root/'prior_status.json')
started=time.monotonic();status={'status':'preflight','pid':os.getpid(),'started_unix':time.time(),'new_downloads':0};write('status.json',status)
try:
 records=[]
 for row in metadata['files']:
  p=snapshot/row['name'];assert p.stat().st_size==row['size'];value=sha(p)
  expected=(row.get('lfs') or {}).get('sha256')
  if expected:assert value==expected,row['name']
  records.append({'file':row['name'],'bytes':row['size'],'sha256':value,'publisher_lfs_sha256':expected})
 write('assets_manifest.json',{'model':metadata['model'],'revision':metadata['revision'],'snapshot':str(snapshot),'files':records,'model_loaded':False})
 shutil.copyfile(prior/'cpu_preflight.json',root/'cpu_preflight.json')
 status.update(status='completed',files_verified=len(records),bytes_verified=sum(x['bytes'] for x in records),snapshot=str(snapshot))
except BaseException as error:status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());write('status.json',status)
print(json.dumps(status),flush=True)
