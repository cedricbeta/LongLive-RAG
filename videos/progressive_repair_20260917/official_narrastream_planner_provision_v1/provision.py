#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
root=Path(__file__).resolve().parent
repo='Qwen/Qwen3.5-27B'
revision='fc05daec18b0a78c049392ed2e771dde82bdf654'
started=time.monotonic()
status={'status':'preparing','pid':os.getpid(),'started_unix':time.time(),'new_metric_runs':0}
def write(name,value):
    temporary=root/(name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(root/name)
write('status.json',status)
try:
    with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/revision/'+revision+'?blobs=true',timeout=30) as stream:
        metadata=json.load(stream)
    assert metadata['sha']==revision and metadata['gated'] is False
    selected=[r for r in metadata['siblings'] if r['rfilename'].endswith(('.safetensors','.json','.jinja','.txt')) or r['rfilename'] in ('README.md','LICENSE')]
    byte_count=sum(r.get('size',0) for r in selected)
    free=shutil.disk_usage(Path.home()/'.cache').free
    assert byte_count < 60*2**30 and free-byte_count > 25*2**30
    plan={'repository':repo,'revision':revision,'frozen_unix':time.time(),'files':selected,
        'expected_bytes':byte_count,'free_bytes_before':free,'minimum_free_reserve_bytes':25*2**30,
        'timeout_seconds':1800,'max_download_workers':2,'GPU_jobs':0,
        'purpose':'Obtain the exact public planner named in the frozen official NarraStream config; no scores or model execution.',
        'failure_policy':'One bounded provisioning attempt; preserve all failures and partial files. No source environment changes.',
        'official_model_url':'https://huggingface.co/'+repo+'/tree/'+revision,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write('plan.json',plan)
    write('public_metadata.json',metadata)
    code="from huggingface_hub import snapshot_download; import json; from pathlib import Path; p=json.loads(Path('plan.json').read_text()); print(snapshot_download(p['repository'],revision=p['revision'],allow_patterns=[r['rfilename'] for r in p['files']],max_workers=2))"
    with (root/'download.log.txt').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u','-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(status='downloading',download_pid=child.pid);write('status.json',status)
        try:
            result=child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(child.pid,signal.SIGTERM)
            child.wait(timeout=20)
            raise
    if result:raise RuntimeError('Download child exit '+str(result))
    cache=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots'/revision
    records=[]
    for row in selected:
        path=cache/row['rfilename'];assert path.is_file()
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
        value=h.hexdigest()
        if row.get('lfs'):assert value==row['lfs']['sha256'],path
        assert path.stat().st_size==row['size']
        records.append({'file':row['rfilename'],'sha256':value,'bytes':path.stat().st_size,'publisher_lfs_sha256':row.get('lfs',{}).get('sha256')})
    write('assets_manifest.json',{'repository':repo,'revision':revision,'cache':str(cache),'files':records,'model_loaded':False,'metric_runs':0})
    status.update(status='completed',bytes_verified=sum(r['bytes'] for r in records),files_verified=len(records),cache=str(cache))
except BaseException as error:
    status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time())
    write('status.json',status)
print(json.dumps(status),flush=True)
