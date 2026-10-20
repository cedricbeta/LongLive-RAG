#!/usr/bin/env python3
"""Recompute frozen automatic assignments to isolate measured mask components."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent if BASE.name=='implementation' else BASE
IMPLEMENTATION=ROOT/'implementation' if (ROOT/'implementation').is_dir() else ROOT
sys.path[:0]=[str(IMPLEMENTATION),str(ROOT/'adapters/official_longlive_rag')]
from entity_verifier import _NativeJudge,NativePartSegmenter,PinnedSam2Tracker
from utils.agent_entity_judge import EntitySpec
from utils.agent_runtime import file_sha256


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    status={'status':'running','pid':os.getpid(),'pgid':os.getpgrp(),'started_unix':time.time()};write('status.json',status)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('GPU 1 worker required')
        lock=open('/tmp/longlive_feedback_render_gpu1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        def resources():
            for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
                device,pid=(x.strip() for x in line.split(','))
                try:group=os.getpgid(int(pid))
                except ProcessLookupError:continue
                if device==uuid and group!=os.getpgrp():raise RuntimeError('resource_interrupted: foreign GPU 1 process')
        resources();original=args.original.resolve();m=json.loads((original/'manifest.json').read_text());parent=args.parent.resolve()
        video,=(parent/'videos').glob('*.mp4');assert file_sha256(video)==m['parent_video_sha256']
        assert file_sha256(args.sam2_checkpoint)==m['sam2_checkpoint_sha256']
        for f,h in m['clipseg_files_sha256'].items():assert file_sha256(args.clipseg_model/f)==h
        manifest={'parent_video_sha256':file_sha256(video),'source_run_manifest_sha256':file_sha256(original/'manifest.json'),
            'automatic_assignments_sha256':file_sha256(original/'assignments.json'),'new_model_calls':0,'new_video_generations':0,
            'manual_boxes':False,'scope':'Recompute same 160 frames using recorded automatic assignments, unchanged models and recipe; inspect components only.',
            'source_sha256':{}}
        for path in [Path(__file__),ROOT/'adapters/official_longlive_rag/entity_verifier.py',IMPLEMENTATION/'utils/agent_entity_judge.py',IMPLEMENTATION/'utils/agent_repair_regions.py']:
            target=out/'source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);manifest['source_sha256'][str(path.relative_to(ROOT))]=file_sha256(target)
        write('manifest.json',manifest)
        audit=json.loads((parent/'audit.json').read_text());grid=tuple(audit['latent_shape'][-2:]);timing=json.loads((parent/'diagnostics/timing_0000.json').read_text());segment=timing['segments'][m['segment']]
        first,count=segment['start_export_frame'],segment['export_frames'];names=[e['name'] for e in m['entity_specs']]
        specs=[EntitySpec(e['name'],e['description']) for e in m['entity_specs']]
        tracker=PinnedSam2Tracker(args.sam2_checkpoint,'cuda:0');part=NativePartSegmenter(args.clipseg_model,'cuda:0',grid)
        judge=_NativeJudge(specs,localizer=None,comparer=None,tracker=tracker,work_dir=out/'entities',grid=grid,frames_per_chunk=1,part_mask_fn=part);judge.progressive_feedback_options=True
        folder,n,hw=judge._shot_frames(video,first,first+count-1);requests={name:{} for name in names}
        for row in json.loads((original/'assignments.json').read_text()):
            for name,decision in row['roles'].items():
                if decision['status']=='assigned':requests[name][row['frame']]=row['boxes'][decision['subject_index']]
        resources();before=time.monotonic();tracks=tracker.tracks(folder,requests,n,hw);track_seconds=time.monotonic()-before
        soft={name:judge._soft(track,n) for name,track in tracks.items()};samples=m['sample_local_frames'];components={};rows=[];reproduction=[]
        for name,track in soft.items():
            for target,phrase in [('hair','hair'),('garment','clothing')]:
                resources();info=judge._attribute_mask(name,[target],video,first,n,track,[t for other,t in soft.items() if other!=name])
                previous=torch.load(original/f'{name}_{target}_frames.pt',map_location='cpu',weights_only=True)
                exact=all(torch.equal(info['masks'][i],previous['masks'][i+first]) for i in range(n))
                reproduction.append({'entity':name,'part':target,'all_frame_masks_exact':exact,'frames':n})
                part.context=track;raw=part(video,list(range(first,first+n)),phrase);face=part(video,list(range(first,first+n)),'face');part.context=None
                for i in samples:
                    person=track[i]>=.35;part_bin=raw[i+first]>=.35;face_bin=face[i+first]>=.35
                    label=f'{name}_{target}_{i}';components[label]=np.stack([track[i][0].numpy(),raw[i+first][0].numpy(),face[i+first][0].numpy(),info['masks'][i][0].numpy()])
                    rows.append({'entity':name,'part':target,'frame':i+first,'person_cells':int(person.sum()),'raw_part_cells':int(part_bin.sum()),'person_part_cells':int((person&part_bin).sum()),
                        'face_cells':int(face_bin.sum()),'part_face_overlap_cells':int((person&part_bin&face_bin).sum()),
                        'final_cells':int((info['masks'][i]>=.35).sum()),'dilation_steps':info['dilation_steps'][str(i)]})
        np.savez_compressed(out/'sample_components.npz',**components)
        result={'rows':rows,'reproduction':reproduction,'tracking_wall_seconds':track_seconds,'segmentation_cost':part.cost,'wall_seconds':time.monotonic()-start,
            'new_model_calls':0,'new_video_generations':0,'quality_claim':False,'ground_truth_masks':False}
        write('result.json',result);status['status']='completed'
    except Exception as error:status.update(status='execution_failed',error=f'{type(error).__name__}: {error}');raise
    finally:status.update(wall_seconds=time.monotonic()-start,finished_unix=time.time());write('status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','original','output','sam2-checkpoint','clipseg-model']:p.add_argument('--'+name,type=Path,required=True)
    torch.set_num_threads(4);main(p.parse_args())
