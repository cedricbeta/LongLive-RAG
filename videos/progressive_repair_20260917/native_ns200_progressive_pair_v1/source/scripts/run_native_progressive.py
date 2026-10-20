#!/usr/bin/env python3
"""Automatic native entity/region/progressive session; development verifier only."""
import argparse
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import importlib.metadata

import torch
import yaml

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent if BASE.name=='implementation' else BASE
IMPLEMENTATION=ROOT/'implementation' if (ROOT/'implementation').is_dir() else ROOT
sys.path[:0]=[str(IMPLEMENTATION),str(ROOT/'adapters/official_longlive_rag')]
from entity_verifier import NativeEntityVerifier, NativePartSegmenter, PinnedSam2Tracker
from progressive_backend import NativeReplayBackend
from utils.agent_bounded_feedback import BoundedFeedbackPolicy
from utils.agent_progressive_repair import ProgressiveShotSession,PairedVisualProtection
from utils.agent_runtime import AgentRuntimeConfig,NormativeGraph,AbstainingController,file_sha256
from utils.agent_script_entities import compile_entities_by_quotes
from utils.agent_typed_script import compile_typed_entities,require_typed_ready
from utils.agent_script_candidates import compile_candidate_entities
from utils.agent_name_candidates import compile_name_candidate_entities
from utils.agent_schema_context import SchemaContextClient
from utils.agent_vlm import FingerprintedVLMClient
from utils.agent_gpu_ownership import diagnostic_owners, require_owned_compute


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def requested_shots(total, start=0, count=None):
    if isinstance(start,bool) or not isinstance(start,int) or not 0<=start<total:
        raise ValueError('Invalid diagnostic start shot')
    count=total-start if count is None else count
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=total-start:
        raise ValueError('Invalid diagnostic shot count')
    return list(range(start,start+count))


class ResourceCheckedClient:
    def __init__(self,client,check):self.client,self.check=client,check
    @property
    def cache_dir(self):return self.client.cache_dir
    def call(self,*args,**kwargs):self.check();return self.client.call(*args,**kwargs)


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();status={'status':'running','pid':os.getpid(),'pgid':os.getpgrp(),'started_unix':time.time()}
    write(out/'status.json',status);backend=None;results=[];verifier=None
    def resources():
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        require_owned_compute(raw,gpu_owners)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.segment_gpu):raise ValueError('Bind diagnostic worker to its designated GPU')
        gpu_owners=diagnostic_owners(args.segment_gpu,args.render_gpu,args.judge_gpu,
            worker_group=os.getpgrp(),judge_group=args.judge_pgid,
            uuid_for_gpu=lambda gpu:subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
            share_diagnostic_judge=getattr(args,'share_diagnostic_judge_gpu',False))
        lock=open(f'/tmp/longlive_feedback_render_gpu{args.segment_gpu}.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);resources()
        parent=args.parent.resolve();config=yaml.safe_load((parent/'config.yaml').read_text())
        prompts=[item['prompts'][0] for item in config['timed_segments']]
        audit=json.loads((parent/'audit.json').read_text());grid=audit['latent_shape'][-2:]
        shots=requested_shots(len(prompts),args.start_shot,args.shot_count)
        source_hashes={}
        sources=list((IMPLEMENTATION/'utils').glob('agent_*.py'))+list((ROOT/'adapters/official_longlive_rag').glob('*.py'))+[Path(__file__)]
        for path in sources:
            name=str(path.relative_to(ROOT));target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            source_hashes[name]=file_sha256(target)
        import sam2
        sam_root=Path(sam2.__file__).parent
        for path in [sam_root/'build_sam.py',sam_root/'sam2_video_predictor.py',sam_root/'configs/sam2.1/sam2.1_hiera_l.yaml']:
            target=out/'external_source/sam2'/path.relative_to(sam_root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_sha256':source_hashes,'source_benchmark_index':json.loads((parent/'manifest.json').read_text())['source_benchmark_index'],
            'parent_manifest_sha256':file_sha256(parent/'manifest.json'),'parent_video_sha256':audit['video_sha256'],
            'raw_prompts':prompts,'model':'Qwen/Qwen3-VL-8B-Instruct','judge_revision':'0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
            'sam2_revision':'665f8e2ad61cf5f53d65644ff27c8ee525124610','sam2_checkpoint_sha256':file_sha256(args.sam2_checkpoint),
            'sam2_source_sha256':{str(p.relative_to(out/'external_source')):file_sha256(p) for p in (out/'external_source').rglob('*') if p.is_file()},
            'installed_sam2_sha256':{str(p.relative_to(sam_root)):file_sha256(p) for p in sam_root.rglob('*')
                                   if p.is_file() and p.suffix in ('.py','.yaml','.so')},
            'environment':{'python':sys.version,**{name:importlib.metadata.version(name) for name in
                           ['torch','transformers','numpy','opencv-python','sam2']}},
            'clipseg_revision':'999e0328d9e10b484360c477313983f9afdd7050',
            'clipseg_files_sha256':{p.name:file_sha256(p) for p in args.clipseg_model.iterdir() if p.is_file()},
            'strength':1.,'max_edits_per_shot':args.max_edits,'lookahead_shots':args.lookahead,
            'diagnostic_gpu_allowed_process_groups':{uuid:sorted(groups) for uuid,groups in gpu_owners.items()},
            'current_shots_requested':shots,'full_video_verified':False,'held_out':False,
            'budget_semantics':'Global wall budget includes compiler/models/diagnosis; no new shot after deadline; in-flight calls finish. Renderer also has separate timeout.',
            'limits':['Development checker is not independent or calibrated; earlier Qwen identity confounds remain.',
                      'No manual entity specs, pixel masks, reference images, or hidden appearance traits.',
                      'Sparse joint role assignment uses first/middle/last frame; SAM2 propagates over every export frame.',
                      'Paired action/quality protection uses still frames, not full motion certification.',
                      'The selected parent is a full video; only requested current/future windows are assessed.']}
        write(out/'manifest.json',manifest)
        compiler=ResourceCheckedClient(FingerprintedVLMClient(base_url=args.base_url,model=manifest['model'],
            cache_dir=out/'vlm_cache',seed=0,max_tokens=4096,max_parse_retries=0),resources)
        if args.compiler_schema_context:compiler=SchemaContextClient(compiler)
        compile_script={'typed_v4':compile_typed_entities,'candidates_v5':compile_candidate_entities,'names_v6':compile_name_candidate_entities,'quotes_v3':compile_entities_by_quotes}[args.compiler]
        compiled=compile_script(compiler,prompts);write(out/'compiled_script.json',compiled)
        if args.compiler!='quotes_v3':require_typed_ready(compiled)
        resources()
        tracker=PinnedSam2Tracker(args.sam2_checkpoint,'cuda:0')
        segmenter=NativePartSegmenter(args.clipseg_model,'cuda:0',grid)
        client=ResourceCheckedClient(FingerprintedVLMClient(base_url=args.base_url,model=manifest['model'],
            cache_dir=out/'vlm_cache',seed=0,max_tokens=2048,max_parse_retries=0),resources)
        verifier=NativeEntityVerifier(compiled,client=client,tracker=tracker,segmenter=segmenter,work_dir=out/'verifier',grid=grid)
        backend=NativeReplayBackend(parent=parent,output=out/'backend',checkout=args.checkout,assets_manifest=args.assets_manifest,
            python=args.generator_python,gpu=args.render_gpu,noise_seed=args.noise_seed,timeout=args.render_timeout)
        scene=out/'scene';scene.mkdir();write(scene/'global.json',{'caption':''})
        for k,prompt in enumerate(prompts):write(scene/f'{k}.json',{'caption':prompt})
        cfg=AgentRuntimeConfig.from_config({'enabled':True,'max_edits_per_shot':args.max_edits,
            'commit_lookahead_shots':args.lookahead,'lock_path':str(out/'session.lock')})
        with ProgressiveShotSession(cfg,NormativeGraph.compile_scene(scene),backend,AbstainingController(),verifier,
                protection=PairedVisualProtection(client,protocol='v2'),feedback_policy=BoundedFeedbackPolicy(
                    client, allow_mixed_evidence_revision=getattr(args, 'mixed_evidence_revision', False),
                    expression_protocol=getattr(args, 'target_expression', 'literal_clauses_v1'),
                    compiled_script=compiled),
                ledger_path=out/'ledger.jsonl',max_repair_seconds=args.max_seconds) as session:
            if args.start_shot:
                session.begin_offline_window(args.start_shot)
            for shot in shots:
                resources();remaining=args.max_seconds-(time.monotonic()-started)
                if remaining<=0:status['status']='timeout';break
                session.max_repair_seconds=remaining
                result=session.run_shot(shot)
                results.append({'shot':shot,'accepted':result.accepted,'quarantined':result.quarantined,
                    'attempts':result.attempts,'delivered':result.delivered,'reason':result.final_verdict.reason})
                write(out/'selected.json',results)
            else:status['status']='completed'
            write(out/'memory_audit.json',{'counts':session.memory.partition_counts(),'invisible_leaks':session.memory.audit_invisible(),
                'identity_registry':verifier.snapshot_state(),'identity_references_promoted':False})
        write(out/'segmentation_cost.json',segmenter.cost)
    except Exception as error:
        status.update(status='execution_failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if verifier is not None:write(out/'diagnosis_calls.json',verifier.calls)
        status.update(wall_seconds=time.monotonic()-started,finished_unix=time.time(),shots_completed=len(results),
            selected_video=backend.active_branch['video'] if backend else str(next((args.parent/'videos').glob('*.mp4'))),
            quality_claim=False)
        write(out/'status.json',status)
        # This server belongs exclusively to this predeclared development run.
        (args.server_dir/'STOP').touch()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','output','checkout','assets-manifest','sam2-checkpoint','clipseg-model','server-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--generator-python',required=True);p.add_argument('--base-url',required=True)
    p.add_argument('--segment-gpu',type=int,default=1);p.add_argument('--render-gpu',type=int,default=0);p.add_argument('--judge-gpu',type=int,default=2)
    p.add_argument('--judge-pgid',type=int,required=True);p.add_argument('--shot-count',type=int)
    p.add_argument('--share-diagnostic-judge-gpu',action='store_true',
                   help='Explicitly colocate segmentation and the exclusive judge on one GPU; rendering stays separate')
    p.add_argument('--start-shot',type=int,default=0,help='First current shot to diagnose; earlier shots remain unassessed')
    p.add_argument('--max-edits',type=int,default=1);p.add_argument('--lookahead',type=int,default=1)
    p.add_argument('--max-seconds',type=float,default=3600);p.add_argument('--render-timeout',type=float,default=1800)
    p.add_argument('--noise-seed',type=int,default=4242)
    p.add_argument('--compiler',choices=['quotes_v3','typed_v4','candidates_v5','names_v6'],default='quotes_v3')
    p.add_argument('--compiler-schema-context',action='store_true')
    p.add_argument('--mixed-evidence-revision',action='store_true',
                   help='Allow one changed target expression after bounded mixed-frame evidence review; never accept unknown')
    p.add_argument('--target-expression', choices=['literal_clauses_v1', 'authored_garment_v1'],
                   default='literal_clauses_v1',
                   help='Opt-in script-bound joint garment expression; no acceptance-policy change')
    torch.set_num_threads(4)
    run(p.parse_args())
