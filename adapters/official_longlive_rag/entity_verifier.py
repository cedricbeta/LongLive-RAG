"""Native time adapter for the existing multi-entity VLM/SAM2 attribute checker.

One virtual chunk is one actual export frame. The unchanged entity judge runs
joint assignment and SAM2 propagation, then this adapter maps its program masks
to native latents. Model accuracy and identity protection remain unvalidated.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'implementation' if (ROOT/'implementation').is_dir() else ROOT))
from utils.agent_attribute_diagnosis import AttributeEvidenceComparer
from utils.agent_entity_judge import EntityDriftVerifier, Sam2Tracker, VLMLocalizer
from utils.agent_runtime import Verdict, file_sha256, fingerprint
from utils.agent_script_entities import entity_specs_at, repair_prompt


def latent_export_indices(index,total):
    frames=[0] if index==0 else list(range(4*index-4,4*index))
    if min(frames)<0 or max(frames)>=total: raise ValueError('Mask projection exceeds export timing')
    return frames


def project_frame_mask(path,*,parent,segment,output,protected=None):
    """Union all associated export frames, then exclude protected latent cells."""
    parent=Path(parent);timing_path=parent/'diagnostics/timing_0000.json'
    timing=json.loads(timing_path.read_text());part=timing['segments'][segment]
    video,=(parent/'videos').glob('*.mp4');latent=parent/'diagnostics/latents_0000.pt'
    payload=torch.load(path,map_location='cpu',weights_only=True)
    first=part['start_latent_frame'];last=first+part['latent_frames']
    expected=set(range(part['start_export_frame'],part['start_export_frame']+part['export_frames']))
    def validate(data):
        if (data['video_sha256']!=file_sha256(video) or data['frames_per_chunk']!=1
                or set(data['masks'])!=expected):
            raise ValueError('Per-frame mask has another parent or time interval')
        grid=list(payload['grid'])
        if list(data['grid'])!=grid: raise ValueError('Mismatched mask grid')
        for tensor in data['masks'].values():
            if (list(tensor.shape)!=[1,*grid] or not torch.isfinite(tensor).all()
                    or tensor.min()<0 or tensor.max()>1):
                raise ValueError('Invalid frame mask')
    validate(payload)
    protected_data=torch.load(protected,map_location='cpu',weights_only=True) if protected else None
    if protected_data: validate(protected_data)
    masks=[];sources=[]
    for index in range(first,last):
        frames=latent_export_indices(index,timing['export_frames'])
        # Only the initial omitted warm-up latent duplicates frame 0. All
        # subsequent authored segments must map entirely inside their interval.
        if not set(frames)<=expected: raise ValueError('Mask crosses authored segment boundary')
        mask=torch.stack([payload['masks'][frame][0]>=.35 for frame in frames]).any(dim=0)
        if protected_data:
            exclusion=torch.stack([protected_data['masks'][frame][0]>=.35 for frame in frames]).any(dim=0)
            mask &= ~exclusion
        masks.append(mask);sources.append({'latent':index,'export_frames':frames})
    tensor=torch.stack(masks)
    result={'mask':tensor,'grid':list(payload['grid']),'latent_range':[first,last],
        'parent_video_sha256':file_sha256(video),'parent_latents_sha256':file_sha256(latent),
        'parent_timing_sha256':file_sha256(timing_path),'projection':'all-associated-export-frames union at .35; protected union excluded',
        'source_mask_sha256':file_sha256(path),'source_protected_sha256':file_sha256(protected) if protected else None,
        'source_frame_mapping':sources,'manual_pixel_coordinates':False}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError('Native projected mask already exists')
    torch.save(result,output)
    return {'path':str(output),'sha256':file_sha256(output),'nonempty':bool(tensor.any()),
            'coverage':float(tensor.float().mean()),'latent_range':[first,last],
            'source_mask_sha256':result['source_mask_sha256'],'source_protected_sha256':result['source_protected_sha256']}


class FrozenAttributeComparer(AttributeEvidenceComparer):
    def install(self,specs,requirements):
        for spec in specs: self._requirements[(spec.name,spec.description)]=deepcopy(requirements[spec.name])

    def compare(self,entity,reference_crop,crops,*,shot_index,inputs):
        if not self._requirements[(entity.name,entity.description)]:
            return {'matches_reference':False,'confidence':0.,'differences':'No explicit appearance predicates active',
                'attribute_assessments':[],'attribute_rechecks':[],'vetoed_parts':[],
                'reference_registration_eligible':False,'crop_votes':[],'identity_review':None}
        return super().compare(entity,reference_crop,crops,shot_index=shot_index,inputs=inputs)


class PinnedSam2Tracker(Sam2Tracker):
    def __init__(self,checkpoint,device):
        from sam2.build_sam import build_sam2_video_predictor
        self.pred=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(checkpoint),device=device)


class NativePartSegmenter:
    """Pinned CLIPSeg part inputs cropped by program SAM2 tracks; faces full-frame."""
    def __init__(self,model_path,device,grid):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        self.processor=CLIPSegProcessor.from_pretrained(str(model_path),local_files_only=True)
        self.model=CLIPSegForImageSegmentation.from_pretrained(str(model_path),local_files_only=True).to(device).eval()
        self.device,self.grid=device,tuple(grid);self.context=None;self.cache={};self.cost=[]

    def __call__(self,video,chunks,phrase):
        from utils.agent_edit import read_video_frames
        context=self.context if phrase!='face' else None
        if phrase!='face' and context is None: raise ValueError('Part segmentation needs an assigned person track')
        context_hash=fingerprint({c:hashlib.sha256(v.numpy().tobytes()).hexdigest() for c,v in context.items()}) if context else None
        key=fingerprint([file_sha256(video),chunks,phrase,context_hash])
        if key in self.cache:return self.cache[key]
        started=time.monotonic();output={}
        first=chunks[0]
        if chunks!=list(range(first,first+len(chunks))): raise ValueError('Contiguous native export frames required')
        frames=read_video_frames(video,first,len(chunks))
        if len(frames)!=len(chunks): raise ValueError('Incomplete native segmentation frames')
        for start in range(0,len(frames),8):
            images=[];boxes=[];indices=[]
            for local in range(start,min(start+8,len(frames))):
                image=frames[local];box=None
                if context:
                    ys,xs=torch.where(context[local][0]>=.35)
                    if not len(xs):
                        output[chunks[local]]=torch.zeros(1,*self.grid);continue
                    h,w=self.grid;W,H=image.size
                    box=(max(0,int(xs.min())-1)*W//w,max(0,int(ys.min())-1)*H//h,
                         min(w,int(xs.max())+2)*W//w,min(h,int(ys.max())+2)*H//h)
                    image=image.crop(box)
                images.append(image);boxes.append(box);indices.append(local)
            if not images: continue
            inputs=self.processor(text=[phrase]*len(images),images=images,return_tensors='pt',padding=True).to(self.device)
            with torch.inference_mode():
                logits=self.model(**inputs).logits
                if logits.ndim==2:logits=logits[None]
                for j,local in enumerate(indices):
                    value=logits[j:j+1,None].sigmoid()
                    if boxes[j]:
                        x0,y0,x1,y1=boxes[j];W,H=frames[local].size
                        full=torch.zeros(1,1,H,W,device=self.device)
                        full[:,:,y0:y1,x0:x1]=torch.nn.functional.interpolate(value,(y1-y0,x1-x0),mode='bilinear',align_corners=False)
                        value=full
                    output[chunks[local]]=torch.nn.functional.interpolate(value,self.grid,mode='bilinear',align_corners=False)[0].cpu().clone()
        self.cache[key]=output
        self.cost.append({'key':key,'phrase':phrase,'entity_context':context_hash,'frames':len(frames),'wall_seconds':time.monotonic()-started})
        return output


class _NativeJudge(EntityDriftVerifier):
    def _attribute_mask(self,name,parts,video,c0,num_chunks,track,protected_tracks=()):
        self._part_mask_fn.context=track
        try:return super()._attribute_mask(name,parts,video,c0,num_chunks,track,protected_tracks)
        finally:self._part_mask_fn.context=None


class NativeEntityVerifier:
    """Adapt existing joint entity diagnosis; never trust appearance-only references."""
    def __init__(self,compiled,*,client,tracker,segmenter,work_dir,grid):
        if compiled.get('compiler_protocol') in ('typed_noun_phrases_v4','typed_candidates_v5','typed_name_candidates_v6','typed_name_candidates_v7'):
            from utils.agent_typed_script import require_typed_ready
            require_typed_ready(compiled)
        self.compiled=deepcopy(compiled);self.work_dir=Path(work_dir);self.work_dir.mkdir(parents=True,exist_ok=False)
        self.comparer=FrozenAttributeComparer(client,max_rechecks=1)
        self.judge_impl=_NativeJudge([],localizer=VLMLocalizer(client),comparer=self.comparer,tracker=tracker,
            work_dir=self.work_dir/'entities',edit_entry_sigma=1.,place_absent='never',uncertain_policy='reject',
            attribute_masks=True,part_mask_fn=segmenter,judge_chunks='sparse',grid=tuple(grid),frames_per_chunk=1)
        self.judge_impl.progressive_attribute_options=True
        self.judge_impl.progressive_feedback_options=True
        self.calls=[];self.results={}

    def snapshot_state(self):return self.judge_impl.snapshot_state()
    def restore_state(self,state):self.judge_impl.restore_state(state)

    def judge(self,observation,normative):
        started=time.monotonic();shot=observation.shot_index;parent=Path(observation.telemetry['native_run'])
        config=yaml.safe_load((parent/'config.yaml').read_text())
        manifest=json.loads((parent/'manifest.json').read_text())
        audit=json.loads((parent/'audit.json').read_text())
        if (file_sha256(parent/'config.yaml')!=manifest['config_sha256']
                or file_sha256(observation.telemetry['shot_video'])!=audit['video_sha256']
                or file_sha256(parent/'diagnostics/latents_0000.pt')!=audit['latent_sha256']):
            raise ValueError('Native diagnostic parent artifacts changed')
        prompts=[item['prompts'][0] for item in config['timed_segments']]
        if prompts!=self.compiled['prompts']:raise ValueError('Compiler and native run use different authored prompts')
        timing_path=parent/'diagnostics/timing_0000.json';timing=json.loads(timing_path.read_text())
        if timing['protocol']!='continuous_vae_drop_first_frame_v1':raise ValueError('Unsupported native timing')
        key=fingerprint([observation.input_fingerprint,self.snapshot_state(),file_sha256(timing_path)])
        if key in self.results:
            result=deepcopy(self.results[key])
            for option in result.metrics.get('repair_options',[]):
                if file_sha256(option['action']['params']['mask_path'])!=option['mask_sha256']:
                    raise ValueError('Cached native diagnostic mask changed')
            self.calls.append({'shot':shot,'attempt':observation.attempt,'cached_judgment':key,
                               'wall_seconds':time.monotonic()-started})
            return result
        segment=timing['segments'][shot];first=segment['start_export_frame'];last=first+segment['export_frames']
        specs,requirements=entity_specs_at(self.compiled,shot)
        self.judge_impl.entities=specs;self.comparer.install(specs,requirements)
        translated=replace(observation,telemetry={**observation.telemetry,'chunk_start':first,'chunk_end':last-1})
        state=self.snapshot_state()
        try:verdict=self.judge_impl.judge(translated,normative)
        finally:self.restore_state(state)
        metrics=deepcopy(dict(verdict.metrics));metrics['repair_references']={}
        metrics['reference_registration']={name:{'status':'withheld_appearance_is_not_identity'} for name in requirements}
        metrics['native_adapter']={'export_frame_range':[first,last],'timing_sha256':file_sha256(timing_path),
            'script_fingerprint':self.compiled['script_fingerprint'],'part_context':'SAM2 assigned track bounding box',
            'identity_references_promoted':False,'full_action_or_quality_evaluation':False}
        # A script with no active appearance is unknown, not a fabricated failure
        # of unspecified attributes. Presence/relations keep their real evidence.
        for row in metrics['constraint_assessments']:
            c=row['constraint'];names=c.get('subjects',[])
            if c['kind']=='attribute' and len(names)==1 and not requirements[names[0]]:
                row.update(status='unknown',violations=[],score=None)
        options=[];projection=[]
        destination=self.work_dir/'native_masks'/key
        for index,option in enumerate(metrics.get('repair_options',[])):
            if not option['id'].startswith('attribute:'):continue
            raw=deepcopy(option);name=option['id'].split(':')[1]
            mask=Path(raw['action']['params']['mask_path'])
            if file_sha256(mask)!=raw['mask_sha256']:raise ValueError('Offered source mask changed')
            protect=raw.get('protected_mask_path')
            if protect and file_sha256(protect)!=raw['protected_mask_sha256']:raise ValueError('Protected source mask changed')
            projected=project_frame_mask(mask,parent=parent,segment=shot,output=destination/f'{index:03d}.pt',protected=protect)
            projection.append({'id':raw['id'],**projected})
            if not projected['nonempty']:continue
            raw['action']['params'].update(mask_path=projected['path'],entry_sigma=1.,
                prompt=repair_prompt(prompts[shot],name,requirements[name]))
            raw.update(mask_sha256=projected['sha256'],native_projection=projected)
            # Preview reads the exact protected native mask; no chunked payload
            # is passed to the official execution adapter.
            for metadata_key in ['protected_mask_path','protected_mask_sha256','mask_content_sha256']:
                raw.pop(metadata_key,None)
            options.append(raw)
        metrics.update(repair_options=options,native_mask_projection=projection)
        accept=verdict.accept and bool(metrics['constraint_assessments']) and all(row['status']=='satisfied' for row in metrics['constraint_assessments'])
        result=replace(verdict,accept=accept,metrics=metrics,directed_actions=())
        self.results[key]=deepcopy(result)
        record={'shot':shot,'attempt':observation.attempt,'native_run':str(parent),'input_fingerprint':observation.input_fingerprint,
            'wall_seconds':time.monotonic()-started,'verdict':{'accept':result.accept,'reason':result.reason,'metrics':metrics}}
        self.calls.append(record)
        target=self.work_dir/f'judgment_{len(self.calls):04d}.json';target.write_text(json.dumps(record,indent=2)+'\n')
        return result
