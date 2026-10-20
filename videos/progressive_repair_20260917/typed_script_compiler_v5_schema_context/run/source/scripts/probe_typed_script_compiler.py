#!/usr/bin/env python3
"""Bounded real text-only development evaluation of the typed phrase compiler."""
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
sys.path.insert(0,str(IMPLEMENTATION))
from utils.agent_typed_script import compile_typed_entities
from utils.agent_script_candidates import compile_candidate_entities
from utils.agent_schema_context import SchemaContextClient
from utils.agent_runtime import file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    status={'status':'running','pid':os.getpid(),'pgid':os.getpgrp(),'started_unix':time.time()};write('status.json',status)
    results=[]
    try:
        plan=json.loads(args.plan.read_text());sources={}
        for p in [Path(__file__),*(IMPLEMENTATION/'utils').glob('agent_*.py')]:
            target=out/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(p.relative_to(ROOT))]=file_sha256(target)
        write('manifest.json',{'plan_sha256':file_sha256(args.plan),'source_sha256':sources,'model':plan['model'],'revision':plan['revision'],'images':0,'new_video_generations':0,'semantically_validated':False,'held_out':False})
        client=FingerprintedVLMClient(base_url=args.base_url,model=plan['model'],cache_dir=out/'vlm_cache',seed=0,max_tokens=4096,max_parse_retries=0)
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
        for case in plan['cases']:
            before=time.monotonic();record={'source_benchmark_index':case['source_benchmark_index']}
            try:
                compiler=compile_candidate_entities if plan.get('compiler_protocol')=='typed_candidates_v5' else compile_typed_entities
                checked=Checked();client_for_compiler=SchemaContextClient(checked) if plan.get('schema_in_prompt') else checked
                compiled=compiler(client_for_compiler,case['raw_prompts']);name=f"compiled_ns{case['source_benchmark_index']:03d}.json";write(name,compiled)
                record.update(status='program_ready' if compiled['ready_for_development'] else 'unresolved',compiled_path=name,
                    entity_count=len(compiled['entities']),unresolved_count=len(compiled['unresolved']))
            except Exception as error:record.update(status='execution_failed',error=f'{type(error).__name__}: {error}')
            record['wall_seconds']=time.monotonic()-before;results.append(record);write('results.json',results)
        status['status']='completed'
    except Exception as error:
        status.update(status='execution_failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(wall_seconds=time.monotonic()-start,finished_unix=time.time(),case_count=len(results),new_video_generations=0,quality_claim=False)
        write('status.json',status);(args.server_dir/'STOP').touch()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','output','server-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--base-url',required=True);p.add_argument('--judge-pgid',type=int,required=True)
    main(p.parse_args())
