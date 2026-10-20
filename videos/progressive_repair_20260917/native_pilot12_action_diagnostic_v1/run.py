#!/usr/bin/env python3
import base64,hashlib,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path

root=Path(__file__).resolve().parent;r=root.parent;workspace=r.parents[1]
sys.path.insert(0,str(workspace/'scripts'))
from run_official_vbench_long import admission
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((root/'plan.json').read_text())
assert sha(Path(__file__))==plan['run_source_sha256'] and not (root/'status.json').exists()
def write(name,value):
    p=root/(name+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(root/name)
observer_log=(root/'observer.log.txt').open('x')
observer=subprocess.Popen(['/home/chendong/miniconda3/envs/sglang/bin/python',str(workspace/'scripts/monitor_native_gpu.py'),
    '--folder',str(root),'--output',str(root/'telemetry'),'--interval','1','--max-wait','1000'],
    stdout=observer_log,stderr=subprocess.STDOUT,start_new_session=True)
until=time.monotonic()+15
while not (root/'telemetry/samples.jsonl').exists() or not (root/'telemetry/samples.jsonl').stat().st_size:
    assert observer.poll() is None and time.monotonic()<until;time.sleep(.2)
started=time.monotonic();status={'status':'preflight','supervisor_pid':os.getpid(),'gpu':2,'started_unix':time.time()}
server=None;responses=[];write('status.json',status)
try:
    status.update(admission(2))
    for row in plan['cases']:
        assert sha(Path(row['video']))==row['video_sha256']
        for frame in row['frames']:assert sha(Path(frame['path']))==frame['sha256']
    directory=root/'server'
    command=[str(workspace/'venv/bin/python'),str(workspace/'scripts/launch_qwen3vl_server.py'),
        '--model',plan['model'],'--revision',plan['revision'],'--gpu','2','--out',str(directory),
        '--mem-fraction-static','0.86','--context-length','16384','--max-running-requests','1',
        '--startup-timeout',str(plan['startup_timeout_seconds'])]
    write('launch.json',{'command':command,'launcher_sha256':sha(workspace/'scripts/launch_qwen3vl_server.py')})
    with (root/'launcher.log.txt').open('x') as log:
        server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        status.update(status='running',launcher_pid=server.pid);write('status.json',status)
        until=time.monotonic()+plan['startup_timeout_seconds']+15
        while not (directory/'base_url.txt').exists():
            if server.poll() is not None:raise RuntimeError('VLM server startup exited '+str(server.returncode))
            if time.monotonic()>until:raise TimeoutError('VLM startup budget')
            time.sleep(1)
        status.update(pid=int((directory/'server.pid').read_text()),startup_seconds=time.monotonic()-started);write('status.json',status)
        endpoint=(directory/'base_url.txt').read_text().strip()+'/chat/completions'
        for row in plan['cases']:
            before=time.monotonic();record={'case':row['id'],'request_number':len(responses)+1,'attempts':1}
            content=[{'type':'text','text':row['diagnostic_prompt']}]
            for frame in row['frames']:
                image=base64.b64encode(Path(frame['path']).read_bytes()).decode()
                content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}})
            payload={'model':plan['model'],'messages':[{'role':'user','content':content}],
                'temperature':0,'seed':0,'max_tokens':plan['max_tokens'],
                'response_format':{'type':'json_schema','json_schema':{'name':plan['protocol'],'strict':True,'schema':row['schema']}}}
            try:
                request=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
                with urllib.request.urlopen(request,timeout=plan['request_timeout_seconds']) as response:raw=json.loads(response.read())
                record['response']=raw
                parsed=json.loads(raw['choices'][0]['message']['content'])
                import jsonschema
                jsonschema.validate(parsed,row['schema']);record.update(status='completed',parsed=parsed)
            except Exception as error:record.update(status='failed',error=type(error).__name__+': '+str(error))
            record['wall_seconds']=time.monotonic()-before;responses.append(record);write('responses.json',responses)
        status.update(status='completed' if all(x['status']=='completed' for x in responses) else 'failed',requests=len(responses))
except BaseException as error:status.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    if server:
        # The launcher and its local server descendants share this owned group.
        if (root/'server').exists():(root/'server/STOP').touch()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        try:os.killpg(server.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        status['server_launcher_returncode']=server.returncode
    status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time());write('status.json',status)
observer.wait(timeout=20);observer_log.close()
print(json.dumps(status),flush=True)
