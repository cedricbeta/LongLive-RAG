"""Development-only identity evidence, isolated from authored appearance targets.

This checker has not been calibrated for branch acceptance. No script appearance,
planner claim, repair intent or candidate-role label is exposed to its VLM call.
"""
import json

from utils.agent_runtime import TemporalInput, file_sha256

FEATURES = ('eyes_brows', 'nose', 'jaw_cheeks')
EVIDENCE_SCHEMA = {'type':'object','additionalProperties':False,
    'properties': {
        'face_visibility_A': {'type':'string','enum':['clear','partial','unobservable']},
        'face_visibility_B': {'type':'string','enum':['clear','partial','unobservable']},
        'actor_correspondence': {'type':'string','enum':['one_unambiguous_subject_per_slot','ambiguous']},
        **{feature:{'type':'object','additionalProperties':False,'properties':{
            'observation_A':{'type':'string','minLength':1,'maxLength':250},
            'observation_B':{'type':'string','minLength':1,'maxLength':250},
            'comparison':{'type':'string','enum':['compatible','contradictory','unobservable']}},
            'required':['observation_A','observation_B','comparison']} for feature in FEATURES}},
    'required':['face_visibility_A','face_visibility_B','actor_correspondence',*FEATURES]}


def decide(evidence):
    from utils.agent_vlm import _validate_json
    _validate_json(evidence,EVIDENCE_SCHEMA)
    if (evidence['face_visibility_A'] != 'clear' or evidence['face_visibility_B'] != 'clear'
        or evidence['actor_correspondence'] != 'one_unambiguous_subject_per_slot'):
        return {'status':'unknown','reason':'Visibility or actor correspondence insufficient'}
    comparisons=[evidence[k]['comparison'] for k in FEATURES]
    if 'contradictory' in comparisons:
        return {'status':'fail','reason':'At least one visible facial-structure contradiction'}
    if comparisons.count('compatible') >= 2:
        return {'status':'pass','reason':'At least two compatible visible facial structures and no contradiction'}
    return {'status':'unknown','reason':'Fewer than two observable compatible structures'}


def compare(client, frames_a, frames_b):
    if not frames_a or len(frames_a)!=len(frames_b):
        raise ValueError('Aligned nonempty sequences required')
    packet={'task':'Compare the facial structure of the depicted subject in slot A and slot B.',
        'scope':'Only visible face correspondence; no claim about unshown times or real-world identity.',
        'instructions':'First describe each requested visible facial feature in each slot; then compare it. '
            'Compatible means the observed face geometry could be the same depicted person after ordinary pose/expression differences. '
            'Contradictory requires a visible structural difference; missing detail must be unobservable. '
            'Use eyes/brows, nose, and jaw/cheeks. Ignore hair color, hair style, clothes, background and whether a script is satisfied. '
            'Do not infer a face from hair/clothes. If facial detail is too small, occluded or absent, mark visibility accordingly. '
            'There is no preferred outcome, and no claim that either image was repaired. Return the requested observations.',
        'images':[{'image':i+1,'slot':'A' if i<len(frames_a) else 'B','sequence_index':i%len(frames_a)}
                  for i in range(len(frames_a)+len(frames_b))]}
    paths=[*frames_a,*frames_b]
    response=client.call(prompt=json.dumps(packet,sort_keys=True),image_paths=paths,
        schema=EVIDENCE_SCHEMA,call_kind='isolated_identity_evidence_v1',shot_index=0,
        temporal_inputs=tuple(TemporalInput(kind='development_identity_frame',identifier=str(p),
            sha256=file_sha256(p),available_after_shot=0) for p in paths))
    return {'decision':decide(response['parsed']),'evidence':response['parsed'],
            'call':{k:response.get(k) for k in ['call_id','transcript','wall_seconds','cached']},
            'independent_final_evaluation':False}
