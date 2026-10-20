#!/usr/bin/env python3
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;workspace=root.parents[2]
arm=sys.argv[1];assert arm in ('P_prompt','R_group')
folder=root/arm;telemetry=root/(arm+'_telemetry')
assert not (folder/'status.json').exists()
observer_log=(root/(arm+'_observer.log.txt')).open('x')
observer=subprocess.Popen(['/home/chendong/miniconda3/envs/sglang/bin/python',str(workspace/'scripts/monitor_native_gpu.py'),
 '--folder',str(folder),'--output',str(telemetry),'--interval','2','--max-wait','2000'],stdout=observer_log,stderr=subprocess.STDOUT,start_new_session=True)
until=time.monotonic()+15
while not (telemetry/'samples.jsonl').exists() or not (telemetry/'samples.jsonl').stat().st_size:
 assert observer.poll() is None and time.monotonic()<until;time.sleep(.2)
with (folder/'supervisor.log.txt').open('x') as log:
 worker=subprocess.Popen([str(root.parent/'env_longlive_rag/bin/python'),str(folder/'supervisor.py'),str(folder)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (root/(arm+'_launch.json')).write_text(json.dumps({'supervisor_pid':worker.pid,'observer_pid':observer.pid,'started_unix':time.time()},indent=2)+'\n')
 print(arm,worker.pid,observer.pid,flush=True);worker.wait()
observer.wait(timeout=30);observer_log.close()
print((folder/'status.json').read_text(),flush=True)
