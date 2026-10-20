#!/usr/bin/env python3
"""Freeze a two-call development diagnostic on completed pilot originals."""
import hashlib,json,time
from pathlib import Path
import cv2
from PIL import Image

root=Path(__file__).resolve().parent
r=root.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
planner=r/'official_narrastream_planner_preflight_v2'
assert read(planner/'status.json')['status']=='completed'
extractions=read(planner/'responses.json')
prompts=read(planner/'plan.json')['cases']
assert [x['case'] for x in extractions]==['ns040','ns130']
assert not (root/'plan.json').exists()
rows=[]
for request,source in zip(extractions,prompts):
    case=request['case'];assert source['id']==case and request['official_parser_accepted']
    parent=r/'native_development_pilot12_v1'/case/'baseline'
    assert read(parent/'status.json')['status']=='completed'
    video,=(parent/'videos').glob('*.mp4')
    audit=read(parent/'audit.json');assert sha(video)==audit['video_sha256']
    timing=read(parent/'diagnostics/timing_0000.json')['segments'][1]
    assert timing['start_export_frame']==160 and timing['export_frames']==160
    frames=[];cap=cv2.VideoCapture(str(video))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==16
    for index in [160,200,240,280,319]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
        path=root/'frames'/case/f'{index:04d}.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(path)
        frames.append({'path':str(path),'sha256':sha(path),'export_index':index,'seconds':index/16})
    cap.release()
    predicates=[]
    for ei,entity in enumerate(request['normalized_payload']['entities']):
        for ai,attribute in enumerate(entity['attributes']):
            predicates.append({'id':f'e{ei}:a{ai}','entity':entity['name'],'predicate':attribute})
    rows.append({'id':case,'prompt':source['prompt'],'shot_index':1,'video':str(video),
                 'video_sha256':audit['video_sha256'],'frames':frames,'predicates':predicates})
assessment={'type':'object','additionalProperties':False,'properties':{
    'status':{'type':'string','enum':['satisfied','violated','unknown']},
    'evidence_frames':{'type':'array','items':{'type':'integer','minimum':0,'maximum':4},'maxItems':5},
    'observed':{'type':'string','maxLength':240}},'required':['status','evidence_frames','observed']}
for row in rows:
    row['schema']={'type':'object','additionalProperties':False,'properties':{
        'assessments':{'type':'object','additionalProperties':False,
            'properties':{p['id']:assessment for p in row['predicates']},'required':[p['id'] for p in row['predicates']]},
        'scene_observation':{'type':'string','maxLength':500},
        'coverage_limit':{'type':'string','maxLength':300}},
        'required':['assessments','scene_observation','coverage_limit']}
    row['diagnostic_prompt']=json.dumps({'task':'Assess each authored entity predicate from these five chronological actual video frames.',
        'original_script':row['prompt'],'predicates':row['predicates'],
        'frame_times_seconds':[x['seconds'] for x in row['frames']],
        'rules':[
            'The extracted predicates are model-derived and may be incomplete or unsupported. The original script is authoritative; unsupported predicates are unknown.',
            'Use satisfied only with directly visible evidence; violated requires a visible contradiction; insufficient or ambiguous evidence is unknown.',
            'Names alone do not identify a face, gender or left/right position. Do not assign a name using stereotypes or invent hidden appearance. If actor correspondence is uncertain, mark the affected entity predicate unknown and describe unnamed visible actors in scene_observation.',
            'A held object in one frame does not establish a completed retrieval or handover. Do not infer actions between frames. A seated visible actor can contradict a requirement that both actors stand, even if individual names are uncertain; explain that joint evidence.',
            'evidence_frames are zero-based indices into the five supplied frames. No video repair, quality acceptance or success is assumed.',
            'Return only the supplied JSON schema. Describe what is visible, not a repair recommendation.'],
        'output_schema':row['schema']},ensure_ascii=False)
plan={'protocol':'native_pilot12_action_diagnostic_v1','frozen_unix':time.time(),'gpu':2,
    'model':'Qwen/Qwen3-VL-30B-A3B-Instruct','revision':'9c4b90e1e4ba969fd3b5378b57d966d725f1b86c',
    'cases':rows,'maximum_requests':2,'request_timeout_seconds':180,'max_tokens':2048,
    'startup_timeout_seconds':420,'total_timeout_seconds':900,'temperature':0,'seed':0,
    'planner_responses_sha256':sha(planner/'responses.json'),'prepare_source_sha256':sha(Path(__file__)),
    'run_source_sha256':sha(root/'run.py'),'new_video_generations':0,
    'limits':['Development action diagnostic, not an official NarraStream metric or final independent evaluator.',
              'One call per case with no transport/parse retries or fallback; all failures retained.',
              'Five sampled frames cannot certify complete actions or the whole 60-second video.',
              'Same Qwen family as existing developer judges. No automatic retention or human labels.']}
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(root/'manifest.json').write_text(json.dumps({'gpu':2,'phase':plan['protocol']},indent=2)+'\n')
print(json.dumps({'cases':len(rows),'predicates':[len(x['predicates']) for x in rows],'frames':10,'launched':False}))
