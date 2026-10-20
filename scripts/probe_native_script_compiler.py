#!/usr/bin/env python3
"""Controlled text-only compiler comparison on a frozen exposed script."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent if BASE.name=='implementation' else BASE
IMPLEMENTATION=ROOT/'implementation' if (ROOT/'implementation').is_dir() else ROOT
sys.path[:0]=[str(IMPLEMENTATION),str(ROOT/'adapters/official_longlive_rag')]
from utils.agent_script_entities import compile_entities_by_quotes
from utils.agent_runtime import file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    status={'status':'running','started_unix':time.time(),'pid':os.getpid(),'pgid':os.getpgrp()}
    def write(name,data):(out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    write('status.json',status)
    try:
        plan=json.loads(args.plan.read_text());prompts=plan['raw_prompts']
        sources={}
        for p in [Path(__file__),*(IMPLEMENTATION/'utils').glob('agent_*.py')]:
            target=out/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            sources[str(p.relative_to(ROOT))]=file_sha256(target)
        write('manifest.json',{'plan_sha256':file_sha256(args.plan),'source_sha256':sources,'model':plan['model'],
            'revision':plan['revision'],'images':0,'new_video_generations':0,'semantically_validated':False,'held_out':False})
        client=FingerprintedVLMClient(base_url=args.base_url,model=plan['model'],cache_dir=out/'vlm_cache',
            seed=0,max_tokens=4096,max_parse_retries=0)
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        class Checked:
            def call(self,**kwargs):
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in raw.splitlines():
                    device,pid=(x.strip() for x in row.split(','))
                    try:group=os.getpgid(int(pid))
                    except ProcessLookupError:continue
                    if device==uuid and group!=args.judge_pgid:raise RuntimeError('resource_interrupted: foreign GPU 0 process')
                return client.call(**kwargs)
        result=compile_entities_by_quotes(Checked(),prompts);write('compiled_script.json',result)
        status['status']='completed'
    except Exception as error:
        status.update(status='execution_failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(wall_seconds=time.monotonic()-start,finished_unix=time.time(),new_video_generations=0,quality_claim=False)
        write('status.json',status);(args.server_dir/'STOP').touch()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','output','server-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--base-url',required=True);p.add_argument('--judge-pgid',type=int,required=True)
    main(p.parse_args())
