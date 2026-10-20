#!/usr/bin/env python3
import hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;r=root.parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(name,value):
 p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
metadata=read(r/'molmo2_o7b_provision_v1/public_metadata.json');prior=read(r/'molmo2_o7b_provision_v2/status.json')
assert prior['status']=='failed' and not Path('/proc',str(prior['pid'])).exists()
checks=read(r/'molmo2_o7b_provision_v2/all_shard_digests.json')
bad=[x['file'] for x in checks['rows'] if not x['matches']]
assert bad==[f'model-{i:05d}-of-00007.safetensors' for i in [1,2,3]]
needed=sum(x['size'] for x in metadata['files'] if x['name'] in bad)
assert shutil.disk_usage(root).free-needed>25*2**30
plan={**metadata,'protocol':'three_invalid_shards_fresh_http_cache_v3','frozen_unix':time.time(),'recover_files':bad,
 'prior_digest_check_sha256':sha(r/'molmo2_o7b_provision_v2/all_shard_digests.json'),'source_sha256':sha(Path(__file__)),
 'timeout_seconds':600,'minimum_free_reserve_bytes':25*2**30,'fresh_cache':str(root/'fresh_http_cache'),
 'reason':'Three existing-cache shard contents differ from publisher LFS digests despite correct sizes. Cause is not established. Recover only those files into a new empty cache; retain original bytes.',
 'new_model_calls':0,'failure_policy':'One bounded recovery transfer; no model/checkpoint substitution and no recursive retry loop.'}
write('plan.json',plan);started=time.monotonic();status={'status':'preflight','pid':os.getpid(),'started_unix':time.time()};child=None;write('status.json',status)
try:
 code="from huggingface_hub import snapshot_download; from pathlib import Path; import json; p=json.loads(Path('plan.json').read_text()); print(snapshot_download(p['model'],revision=p['revision'],allow_patterns=p['recover_files'],cache_dir=p['fresh_cache'],max_workers=2))"
 with (root/'download.log.txt').open('x') as log:
  child=subprocess.Popen([sys.executable,'-u','-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_DISABLE_XET':'1'},start_new_session=True)
  status.update(status='downloading',download_pid=child.pid);write('status.json',status)
  while child.poll() is None:
   if time.monotonic()-started>600:raise TimeoutError('Fresh-cache transfer exceeded 600 seconds')
   if shutil.disk_usage(root).free<25*2**30:raise RuntimeError('Free-space reserve reached')
   time.sleep(5)
  if child.returncode:raise RuntimeError('Recovery transfer exit '+str(child.returncode))
 base=Path.home()/'.cache/huggingface/hub/models--allenai--Molmo2-O-7B/snapshots'/metadata['revision']
 fresh=root/'fresh_http_cache/models--allenai--Molmo2-O-7B/snapshots'/metadata['revision']
 snapshot=root/'checkpoint';snapshot.mkdir();records=[]
 for row in metadata['files']:
  source=(fresh if row['name'] in bad else base)/row['name'];assert source.stat().st_size==row['size'];value=sha(source)
  expected=(row.get('lfs') or {}).get('sha256')
  if expected:assert value==expected,row['name']
  (snapshot/row['name']).symlink_to(source.resolve())
  records.append({'file':row['name'],'bytes':row['size'],'sha256':value,'publisher_lfs_sha256':expected,'fresh_transfer':row['name'] in bad})
 write('assets_manifest.json',{'model':metadata['model'],'revision':metadata['revision'],'snapshot':str(snapshot),'files':records,'model_loaded':False})
 shutil.copyfile(r/'molmo2_o7b_provision_v1/cpu_preflight.json',root/'cpu_preflight.json')
 status.update(status='completed',files_verified=len(records),bytes_verified=sum(x['bytes'] for x in records),snapshot=str(snapshot))
except BaseException as error:status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());write('status.json',status)
print(json.dumps(status),flush=True)
