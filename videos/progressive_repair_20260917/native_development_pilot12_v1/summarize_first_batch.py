#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
f=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(f/'plan.json');rows=[]
for name in ('ns040','ns130'):
    folder=f/name/'baseline';case=next(c for c in plan['cases'] if c['id']==name)
    switches=read(folder/'diagnostics/switches_0000.json')
    assert len(switches)==5
    for i,row in enumerate(switches,1):
        assert row['prompt_sha256']==[hashlib.sha256(case['prompts'][i].encode()).hexdigest()]
    audit=read(folder/'audit.json');telemetry=read(f/(name+'_telemetry')/'summary.json')
    assert audit['engineering_checks_passed'] and read(folder/'status.json')['status']=='completed'
    assert read(f/(name+'_telemetry')/'target_terminal_status.json')==read(folder/'status.json')
    rows.append({'id':name,'generation_seconds':audit['generation_wall_seconds'],'audit_seconds':audit['audit_wall_seconds'],
        'video_sha256':audit['video_sha256'],'latent_sha256':audit['latent_sha256'],'all_five_actual_prompt_switch_hashes_match':True,
        'full_frames_decoded':960,'fixed_frames_viewed':plan['review_frames'],'device_energy_bounds':telemetry['device_energy_bounds'],
        'observed_foreign_compute_pids':telemetry['observed_foreign_compute_pids_inside']})
summary={'phase':'first_two_originals_for_preselected_twelve_script_pilot','selection_plan_sha256':sha(f/'plan.json'),
    'planned_scripts':12,'selected_indices':[x['index'] for x in plan['cases']],'first_batch_attempted':2,'completed':2,
    'new_repairs':0,'new_control_generations':0,'remaining_first_batch_generation_budget':0,'remaining_selected_originals_unlaunched':10,
    'full_comparative_pilot_completed':False,'full_comparative_protocol_frozen':False,'runs':rows,
    'generation_seconds_sum':sum(x['generation_seconds'] for x in rows),'preparation_seconds':read(f/'preparation.json')['wall_seconds'],
    'queue_seconds':read(f/'queue_status.json')['wall_seconds'],'developer_observations_sha256':sha(f/'developer_observations.json'),
    'held_out':False,'human_labels':0,'automatic_retained_improvements':0,'quality_acceptance':False,'submission_ready':False,
    'scope':'Two official LongLive-RAG originals, not repair efficacy. Twelve cases were selected before these outputs; no action target was provided from sampling metadata.'}
(f/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'generation_seconds_sum':summary['generation_seconds_sum'],'queue_seconds':summary['queue_seconds'],'new_repairs':0}))
