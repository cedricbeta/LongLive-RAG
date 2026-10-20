#!/usr/bin/env python3
"""Run a preconstructed bounded control set; never update a repair branch."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.agent_identity_comparison import compare
from utils.agent_runtime import file_sha256
from utils.agent_vlm import FingerprintedVLMClient


def run(folder,base_url):
    manifest=json.loads((folder/'manifest.json').read_text())
    if (folder/'status.json').exists():
        raise FileExistsError('Probe already launched; do not silently repeat it')
    for rel,digest in manifest['input_sha256'].items():
        if file_sha256(folder/rel)!=digest:
            raise ValueError('Frozen input changed: '+rel)
    source=folder/'source';source.mkdir()
    frozen={}
    for rel in ['scripts/run_identity_isolation_probe.py','utils/agent_identity_comparison.py',
                'utils/agent_runtime.py','utils/agent_vlm.py','utils/agent_edit.py']:
        p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
        frozen[rel]=file_sha256(p)
    (folder/'source_manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
    client=FingerprintedVLMClient(base_url=base_url,model=manifest['model'],cache_dir=folder/'vlm_cache',
                                 max_tokens=2048,seed=0,max_parse_retries=0)
    started=time.monotonic();rows=[]
    status={'status':'running','pid':__import__('os').getpid(),'started_unix':time.time()}
    (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        for case in manifest['cases']:
            result=compare(client,case['frames_a'],case['frames_b'])
            expected=case['expected_control_status']
            rows.append({'id':case['id'],'expected_control_status':expected,
                         'control_matches_expectation':None if expected is None else result['decision']['status']==expected,
                         **result})
            (folder/'partial_results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(case['id'],result['decision'],flush=True)
        summary={'phase':'development_identity_isolation','rows':rows,
                 'constructed_controls':sum(r['expected_control_status'] is not None for r in rows),
                 'controls_matching_expectation':sum(r['control_matches_expectation'] is True for r in rows),
                 'wall_seconds':time.monotonic()-started,'integrated_into_repair_acceptance':False,
                 'independent_final_evaluation':False,'live_branch_promoted':False,
                 'limitations':manifest['limits']}
        (folder/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        status.update(status='completed',wall_seconds=time.monotonic()-started)
    except Exception as error:
        status.update(status='execution_failed',error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-started)
        raise
    finally:
        (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--base-url',required=True)
    args=p.parse_args();run(args.folder.resolve(),args.base_url)
