"""Exact program-generated phrase candidates; model chooses owner IDs only.

This keeps the typed grammar's unresolved cases visible. It removes free-form
physical quote generation, not the need to evaluate coreference/completeness.
"""
from copy import deepcopy
import json
import re
from utils.agent_typed_script import (HEAD_PATTERN,TOKEN,MODIFIERS,BOUNDARIES,NEGATIVE,
    parse_physical_phrase,compile_typed_proposal)
from utils.agent_script_entities import SPAN
from utils.agent_runtime import TemporalInput
from utils.agent_vlm import _validate_json


def physical_candidates(prompts):
    candidates=[]
    for shot,text in enumerate(prompts):
        tokens=list(TOKEN.finditer(text));positions={m.start():i for i,m in enumerate(tokens)}
        for head in HEAD_PATTERN.finditer(text):
            start,end=head.start(),head.end();index=positions[start]
            while index>0 and tokens[index-1].group().lower() in MODIFIERS and text[tokens[index-1].end():start].isspace():
                index-=1;start=tokens[index].start()
            quote=text[start:end];parsed=parse_physical_phrase(quote)
            context=text[:end].rsplit('.',1)[-1].rsplit(';',1)[-1].rsplit(',',1)[-1]
            if set(m.group().lower() for m in TOKEN.finditer(context))&NEGATIVE:
                parsed={'atoms':[],'unresolved':'Negated/hypothetical physical mention needs explicit semantics'}
            elif index>0 and text[tokens[index-1].end():start].isspace() and tokens[index-1].group().lower() not in BOUNDARIES and not tokens[index-1].group()[0].isupper():
                start=tokens[index-1].start();quote=text[start:end]
                parsed={'atoms':[],'unresolved':'Unknown adjacent modifier or unsupported syntactic context'}
            if not parsed['atoms'] and not parsed['unresolved']:continue
            candidates.append({'id':f'p{len(candidates):03d}','shot':shot,'quote':quote,'start_char':start,'end_char':end,
                'atoms':parsed['atoms'],'unresolved':parsed['unresolved'],'context':text})
    return candidates


def compile_candidate_assignments(prompts,cast,candidates,assignments):
    """Owners may change; literal atoms, offsets and source timing may not."""
    if candidates!=physical_candidates(prompts):raise ValueError('Candidate inventory differs from original script')
    names=[e['name'] for e in cast]
    if len(set(names))!=len(names):raise ValueError('Duplicate cast name')
    if set(assignments)!={c['id'] for c in candidates}:raise ValueError('Every program phrase requires one ownership decision')
    entities=[{**deepcopy(e),'appearance_quotes':[],'unresolved_appearance':[]} for e in cast]
    by_name={e['name']:e for e in entities};unassigned=[];binding_audit=[]
    for row in candidates:
        decision=assignments[row['id']];owners=decision['owners'];status=decision['status']
        if len(set(owners))!=len(owners) or any(o not in by_name for o in owners):raise ValueError('Owners must be unique named cast IDs')
        if status not in ('appearance','nonappearance','ambiguous'):raise ValueError('Unknown ownership status')
        if (status=='appearance')!=bool(owners):raise ValueError('Only appearance decisions require named owners')
        quote=prompts[row['shot']][row['start_char']:row['end_char']]
        if quote!=row['quote']:raise ValueError('Candidate offset mismatch')
        # A directly named grammatical subject/possessor takes precedence over
        # a conflicting model owner. Other coreference remains unvalidated.
        before=prompts[row['shot']][:row['start_char']]
        explicit=[]
        for name in names:
            pattern=r'(?<!\w)'+re.escape(name)+r"(?:'s|,\s*(?:a|an|the)?|\s+(?:has|wears|sports)\s+(?:a|an|the)?)?\s*$"
            if re.search(pattern,before):explicit.append(name)
        if explicit and status=='appearance' and owners!=explicit:
            raise ValueError('Owner conflicts with directly named subject or possessor')
        binding_audit.append({'id':row['id'],'status':status,'owners':owners,'explicit_subjects':explicit,'reason':decision['reason']})
        reason=row['unresolved'] or ('Unresolved phrase ownership' if status!='appearance' else None)
        if reason:
            # Nonappearance remains unresolved in this conservative version;
            # the model cannot discard a program physical mention as context.
            unassigned.append({'entity':None,'source':{'shot':row['shot'],'quote':quote},'reason':reason,'candidate_id':row['id']})
            continue
        for owner in owners:by_name[owner]['appearance_quotes'].append({'shot':row['shot'],'quote':quote})
    compiled=compile_typed_proposal(prompts,{'entities':entities})
    compiled['unresolved'].extend(unassigned)
    compiled['ready_for_development']=not compiled['unresolved']
    compiled.update(compiler_protocol='typed_candidates_v5',candidate_inventory=deepcopy(candidates),ownership_audit=binding_audit,
        cast_proposal=deepcopy(cast))
    compiled['limits'] += ['Every recognized physical mention is accounted for; nonappearance/ambiguous ownership remains unresolved.',
        'Finite known-head coverage does not prove extraction of unseen heads or correct nonlocal pronoun/group attribution.']
    return compiled


def compile_candidate_entities(client,prompts):
    presence={'type':'object','additionalProperties':False,'properties':{str(k):{'type':'string','enum':['required','ambiguous','not_required']} for k in range(len(prompts))},'required':[str(k) for k in range(len(prompts))]}
    schema={'type':'object','additionalProperties':False,'properties':{'entities':{'type':'array','maxItems':8,'items':{
        'type':'object','additionalProperties':False,'properties':{'name':{'type':'string','minLength':1,'maxLength':80},'name_source':SPAN,'presence':presence},
        'required':['name','name_source','presence']}}},'required':['entities']}
    inputs=(TemporalInput.from_value(kind='fixed_authored_script',identifier='native_script',value=prompts,available_after_shot=-1,authored=True),)
    cast=client.call(prompt=json.dumps({'task':'List all named recurring characters in the fixed script. '
        'Copy each exact name and an exact source quote containing it. For every shot report depicted presence: required, ambiguous, or not_required. '
        'Do not extract any appearance. A name mentioned in a recollection does not require visible presence. Strict JSON.',
        'prompts':{str(k):p for k,p in enumerate(prompts)}},ensure_ascii=False),image_paths=[],schema=schema,
        call_kind='native_candidate_cast_v5',shot_index=0,temporal_inputs=inputs)['parsed']
    _validate_json(cast,schema)
    return compile_candidate_owners(client,prompts,cast['entities'])


def compile_candidate_owners(client,prompts,entities):
    """Shared immutable physical-candidate ownership stage (unchanged v5 prompt)."""
    inputs=(TemporalInput.from_value(kind='fixed_authored_script',identifier='native_script',value=prompts,available_after_shot=-1,authored=True),)
    # Validate cast without suppressing its initially missing physical inventory.
    compile_typed_proposal(prompts,{'entities':[{**e,'appearance_quotes':[],'unresolved_appearance':[]} for e in entities]})
    candidates=physical_candidates(prompts);names=[e['name'] for e in entities]
    if not candidates:return compile_candidate_assignments(prompts,entities,candidates,{})
    owner_schema={'type':'object','additionalProperties':False,'properties':{c['id']:{'type':'object','additionalProperties':False,
        'properties':{'owners':{'type':'array','maxItems':len(names),'items':{'type':'string','enum':names}},
            'status':{'type':'string','enum':['appearance','nonappearance','ambiguous']},'reason':{'type':'string','maxLength':200}},
        'required':['owners','status','reason']} for c in candidates},'required':[c['id'] for c in candidates]}
    if not names:
        unknown={c['id']:{'owners':[],'status':'ambiguous','reason':'No named character extracted'} for c in candidates}
        return compile_candidate_assignments(prompts,[],candidates,unknown)
    response=client.call(prompt=json.dumps({'task':'Assign each immutable physical-description candidate ID to its actual named owner(s) in the script. '
        'Do not change any quote, value or attribute type. A phrase about multiple characters may have multiple owners only when the script clearly supports that. '
        'Use appearance with nonempty owners only for explicitly worn/possessed/physical traits. Use nonappearance with [] for another object or a non-descriptive mention, '
        'or ambiguous with [] when ownership or meaning is unclear. Do not infer attributes from names. Decide every candidate ID exactly once. Strict JSON.',
        'named_cast':entities,'candidates':candidates,'prompts':{str(k):p for k,p in enumerate(prompts)}},ensure_ascii=False),
        image_paths=[],schema=owner_schema,call_kind='native_candidate_owners_v5',shot_index=0,temporal_inputs=inputs)
    _validate_json(response['parsed'],owner_schema)
    return compile_candidate_assignments(prompts,entities,candidates,response['parsed'])
