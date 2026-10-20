#!/usr/bin/env python3
import hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
prior=root.parent/'official_narrastream_planner_provision_v1'
read=lambda p:json.loads(p.read_text())
def write(name,value):
    p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()
old=read(prior/'status.json')
assert old['status']=='failed','Wait for the existing attempt to terminate; do not duplicate a live transfer'
assert not Path('/proc',str(old['download_pid'])).exists()
assert 'TimeoutExpired' in old.get('error',''),old
plan=read(prior/'plan.json');base=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots'/plan['revision']
missing=[r for r in plan['files'] if not (base/r['rfilename']).is_file()]
assert [r['rfilename'] for r in missing]==['model.safetensors-00001-of-00011.safetensors'],missing
assert shutil.disk_usage(root).free-missing[0]['size']>25*2**30
plan.update(protocol='same_planner_missing_shard_http_recovery_v2',frozen_unix=time.time(),
    prior_plan_sha256=sha(prior/'plan.json'),prior_terminal_status_sha256=sha(prior/'status.json'),
    timeout_seconds=600,missing_files=missing,transport='HF_HUB_DISABLE_XET=1 in a separate new cache; do not truncate the retained preallocated partial file.',
    failure_policy='One HTTP recovery attempt, no model/checkpoint change or retry loop.',source_sha256=sha(Path(__file__)))
write('plan.json',plan);shutil.copyfile(prior/'status.json',root/'prior_terminal_status.json')
started=time.monotonic();status={'status':'preflight','pid':os.getpid(),'started_unix':time.time(),'new_metric_runs':0};write('status.json',status);child=None
try:
    code="from huggingface_hub import hf_hub_download; import json; from pathlib import Path; p=json.loads(Path('plan.json').read_text()); print(hf_hub_download(p['repository'],p['missing_files'][0]['rfilename'],revision=p['revision'],cache_dir='http_cache'))"
    with (root/'download.log.txt').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u','-c',code],cwd=root,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'HF_HUB_DISABLE_XET':'1'},start_new_session=True)
        status.update(status='downloading',download_pid=child.pid);write('status.json',status)
        result=child.wait(timeout=600)
    if result:raise RuntimeError('HTTP recovery exit '+str(result))
    recovered=root/'http_cache/models--Qwen--Qwen3.5-27B/snapshots'/plan['revision']/missing[0]['rfilename']
    assert sha(recovered)==missing[0]['lfs']['sha256']
    snapshot=root/'checkpoint';snapshot.mkdir()
    records=[]
    for row in plan['files']:
        source=recovered if row['rfilename']==missing[0]['rfilename'] else base/row['rfilename']
        value=sha(source)
        assert source.stat().st_size==row['size']
        if row.get('lfs'):assert value==row['lfs']['sha256'],row['rfilename']
        target=snapshot/row['rfilename'];target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source.resolve())
        records.append({'file':row['rfilename'],'sha256':value,'bytes':source.stat().st_size,'publisher_lfs_sha256':row.get('lfs',{}).get('sha256')})
    write('assets_manifest.json',{'repository':plan['repository'],'revision':plan['revision'],'cache':str(snapshot),
        'files':records,'model_loaded':False,'metric_runs':0,'same_expected_shards_as_v1':True})
    shutil.copyfile(prior/'cpu_import_config_preflight_v2.json',root/'cpu_import_config_preflight_v2.json')
    status.update(status='completed',bytes_verified=sum(r['bytes'] for r in records),files_verified=len(records),cache=str(snapshot))
except BaseException as error:status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    if child and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());write('status.json',status)
print(json.dumps(status),flush=True)
