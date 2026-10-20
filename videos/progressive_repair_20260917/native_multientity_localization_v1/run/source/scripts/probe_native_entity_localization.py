#!/usr/bin/env python3
"""Native multi-entity localization diagnostic, independently of failed atomic parsing.

Uses exact previously model-extracted cast quotes, new joint visual assignment,
and real SAM2/CLIPSeg. Produces masks only; cannot render or accept a repair.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import yaml

BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent if BASE.name=='implementation' else BASE
IMPLEMENTATION=ROOT/'implementation' if (ROOT/'implementation').is_dir() else ROOT
sys.path[:0]=[str(IMPLEMENTATION),str(ROOT/'adapters/official_longlive_rag')]
from entity_verifier import _NativeJudge,NativePartSegmenter,PinnedSam2Tracker,project_frame_mask
from utils.agent_entity_judge import EntitySpec,VLMLocalizer,validate_role_assignments
from utils.agent_runtime import file_sha256,TemporalInput
from utils.agent_edit import save_mask_payload
from utils.agent_vlm import FingerprintedVLMClient
from run_native_progressive import ResourceCheckedClient


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    status={'status':'running','pid':os.getpid(),'pgid':os.getpgrp(),'started_unix':time.time()};write(out/'status.json',status)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Localization worker must use GPU 1')
        lock=open('/tmp/longlive_feedback_render_gpu1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owners={subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip():group
                for gpu,group in [(1,os.getpgrp()),(2,args.judge_pgid)]}
        def resources():
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            for line in raw.splitlines():
                uuid,pid=(x.strip() for x in line.split(','))
                try:group=os.getpgid(int(pid))
                except ProcessLookupError:continue
                if uuid in owners and group!=owners[uuid]:raise RuntimeError('resource_interrupted: foreign compute process')
        resources()
        parent=args.parent.resolve();config=yaml.safe_load((parent/'config.yaml').read_text())
        prompts=[r['prompts'][0] for r in config['timed_segments']]
        timing=json.loads((parent/'diagnostics/timing_0000.json').read_text());shot=args.segment;s=timing['segments'][shot]
        audit=json.loads((parent/'audit.json').read_text());video,=(parent/'videos').glob('*.mp4');assert file_sha256(video)==audit['video_sha256']
        parsed=json.loads(args.cast_result.read_text())['parsed'];specs=[]
        for entity in parsed['entities']:
            if entity['presence'][str(shot)]!='required':continue
            source=entity['name_source'];assert source['quote'] in prompts[source['shot']] and entity['name'] in source['quote']
            quotes=[row['quote'] for row in entity['appearance_quotes'] if row['shot']<=shot]
            for row in entity['appearance_quotes']:assert row['quote'] in prompts[row['shot']]
            specs.append(EntitySpec(entity['name'],'; '.join(quotes),shots=(shot,)))
        manifest={'parent_video_sha256':audit['video_sha256'],'parent_latents_sha256':audit['latent_sha256'],
            'cast_result_sha256':file_sha256(args.cast_result),'source_cast_model':'Qwen/Qwen3-VL-8B-Instruct',
            'source_cast_protocol':'native_cast_quotes_v3','segment':shot,'sample_local_frames':[0,s['export_frames']//2,s['export_frames']-1],
            'entity_specs':[{'name':x.name,'description':x.description} for x in specs],
            'manual_pixel_coordinates':False,'automatic_atomic_diagnosis':False,'new_video_generations':0,
            'repair_acceptance':False,'quality_claim':False,'source_sha256':{}}
        manifest.update(sam2_checkpoint_sha256=file_sha256(args.sam2_checkpoint),
            clipseg_files_sha256={p.name:file_sha256(p) for p in args.clipseg_model.iterdir() if p.is_file()},
            environment={'python':sys.version,**{name:importlib.metadata.version(name) for name in ['torch','transformers','sam2']}},
            gpu_roles={'segmentation':1,'judge':2,'reserved':3})
        for path in [Path(__file__),ROOT/'adapters/official_longlive_rag/entity_verifier.py',IMPLEMENTATION/'utils/agent_entity_judge.py',IMPLEMENTATION/'utils/agent_edit.py',IMPLEMENTATION/'utils/agent_vlm.py']:
            dst=out/'source'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
            manifest['source_sha256'][str(path.relative_to(ROOT))]=file_sha256(dst)
        write(out/'manifest.json',manifest)
        client=ResourceCheckedClient(FingerprintedVLMClient(base_url=args.base_url,model='Qwen/Qwen3-VL-8B-Instruct',cache_dir=out/'vlm_cache',seed=0,max_tokens=2048,max_parse_retries=0),resources)
        localizer=VLMLocalizer(client)
        tracker=PinnedSam2Tracker(args.sam2_checkpoint,'cuda:0')
        part=NativePartSegmenter(args.clipseg_model,'cuda:0',audit['latent_shape'][-2:])
        judge=_NativeJudge(specs,localizer=localizer,comparer=None,tracker=tracker,work_dir=out/'entities',
            grid=tuple(audit['latent_shape'][-2:]),frames_per_chunk=1,part_mask_fn=part)
        judge.progressive_feedback_options=True
        first=s['start_export_frame'];count=s['export_frames'];folder,n,hw=judge._shot_frames(video,first,first+count-1)
        requests={e.name:{} for e in specs};assignments=[];previous={}
        for frame in manifest['sample_local_frames']:
            path=folder/f'chunk{frame:02d}.png';inputs=(TemporalInput('current_shot_frame',str(path),file_sha256(path),shot),)
            boxes=localizer.enumerate_subjects(path,shot_index=shot,inputs=inputs)
            raw=localizer.assign_subjects(specs,{},path,boxes,caption=prompts[shot],previous_boxes=previous,shot_index=shot,inputs=inputs)
            rows=validate_role_assignments(specs,boxes,raw);previous={}
            for name,row in rows.items():
                if row['status']=='assigned':
                    box=boxes[row['subject_index']];requests[name][frame]=box;previous[name]=box
            assignments.append({'frame':frame,'export_frame':first+frame,'boxes':boxes,'roles':rows})
            write(out/'assignments.json',assignments)
        resources();before=time.monotonic();tracks=tracker.tracks(folder,requests,n,hw);track_seconds=time.monotonic()-before
        soft={name:judge._soft(track,n) for name,track in tracks.items()};masks=[]
        for name,track in soft.items():
            if not requests[name]:continue
            for target in ['hair','garment']:
                resources()
                info=judge._attribute_mask(name,[target],video,first,n,track,[values for other,values in soft.items() if other!=name])
                frame_mask=save_mask_payload({i+first:m for i,m in info['masks'].items()},out/f'{name}_{target}_frames.pt',phrase=target,video=video,grid=judge.grid,frames_per_chunk=1)
                protection=save_mask_payload({i+first:m for i,m in info['protected'].items()},out/f'{name}_{target}_protected.pt',phrase='faces and other actors',video=video,grid=judge.grid,frames_per_chunk=1)
                projected=project_frame_mask(frame_mask,parent=parent,segment=shot,output=out/f'{name}_{target}_native.pt',protected=protection)
                masks.append({'entity':name,'part':target,**projected})
                write(out/'masks.json',masks)
        result={'phase':'native_multientity_localization_diagnostic','assignments':assignments,'masks':masks,
            'tracking_wall_seconds':track_seconds,'segmentation_cost':part.cost,'wall_seconds':time.monotonic()-start,
            'new_video_generations':0,'candidate_retained':False,'quality_claim':False,
            'limits':['No atomic diagnosis or repair decision; compiler failures remain unresolved.',
                      'This checks actual multi-actor assignment/tracking/projection, not ground-truth mask accuracy.']}
        write(out/'result.json',result);status['status']='completed'
    except Exception as error:
        status.update(status='execution_failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(wall_seconds=time.monotonic()-start,finished_unix=time.time());write(out/'status.json',status)
        (args.server_dir/'STOP').touch()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','cast-result','output','sam2-checkpoint','clipseg-model','server-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--base-url',required=True);p.add_argument('--segment',type=int,default=0)
    p.add_argument('--judge-pgid',type=int,required=True)
    torch.set_num_threads(4);run(p.parse_args())
