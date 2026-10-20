"""Compile named characters and atomic appearance predicates from exact script spans.

The model proposes attribution and temporal scope; the program validates every
quoted value and interval. This prevents invented literal traits, not semantic
misattribution or omissions, which still require independent evaluation.
"""
from copy import deepcopy
import json
from utils.agent_attribute_diagnosis import PARTS, ASPECTS, freeze_requirements, validate_required_aspects
from utils.agent_entity_judge import EntitySpec
from utils.agent_runtime import TemporalInput, fingerprint
from utils.agent_vlm import _validate_json

SPAN={'type':'object','additionalProperties':False,'properties':{
    'shot':{'type':'integer','minimum':0},'quote':{'type':'string','minLength':1}},'required':['shot','quote']}
SCHEMA={'type':'object','additionalProperties':False,'properties':{'entities':{'type':'array','maxItems':12,'items':{
    'type':'object','additionalProperties':False,'properties':{
        'name':{'type':'string','minLength':1,'maxLength':80},'name_source':SPAN,
        'presence':{'type':'array','items':{'type':'object','additionalProperties':False,'properties':{
            'shot':{'type':'integer','minimum':0},'status':{'type':'string','enum':['required','ambiguous','not_required']},
            'quote':{'type':'string'}},'required':['shot','status','quote']}},
        'appearance':{'type':'array','maxItems':32,'items':{'type':'object','additionalProperties':False,'properties':{
            'part':{'type':'string','enum':list(PARTS)},'aspect':{'type':'string','enum':list(ASPECTS)},
            'expected':{'type':'string','minLength':1},'source':SPAN,
            'last_active_shot':{'type':'integer','minimum':0}},
            'required':['part','aspect','expected','source','last_active_shot']}}},
    'required':['name','name_source','presence','appearance']}}},'required':['entities']}


def validate_compilation(prompts, parsed):
    _validate_json(parsed,SCHEMA)
    result=deepcopy(parsed);names=set();count=len(prompts)
    if not count or any(not isinstance(p,str) or not p.strip() for p in prompts):
        raise ValueError('Nonempty authored prompts required')
    for entity in result['entities']:
        name=entity['name'];source=entity['name_source'];index=source['shot']
        if name!=name.strip() or name in names or not 0<=index<count or source['quote'] not in prompts[index] or name not in source['quote']:
            raise ValueError('Entity name must have a unique exact authored source')
        names.add(name)
        presence=entity['presence']
        if sorted(row['shot'] for row in presence)!=list(range(count)):
            raise ValueError('Presence must explicitly cover every authored shot once')
        for row in presence:
            if row['quote'] not in prompts[row['shot']] or (row['status']=='required' and not row['quote']):
                raise ValueError('Required presence needs an exact source quote')
        seen=set()
        for row in entity['appearance']:
            first=row['source']['shot'];last=row['last_active_shot'];quote=row['source']['quote']
            if (not 0<=first<=last<count or quote not in prompts[first] or row['expected'] not in quote
                    or row['expected']!=row['expected'].strip()):
                raise ValueError('Appearance and activation must be bound to an authored quote')
            key=(row['part'],row['aspect'],row['expected'],first,last)
            if key in seen: raise ValueError('Duplicate appearance predicate')
            seen.add(key)
        for row in entity['appearance']:
            # Missing repeated mention is not an authored change. Persistent
            # attributes end only at a later explicit replacement of the same
            # scalar aspect; detail predicates remain active conservatively.
            later=[other['source']['shot'] for other in entity['appearance']
                if row['aspect'] in ('color','style','type','age','build')
                and (other['part'],other['aspect'])==(row['part'],row['aspect'])
                and other['expected']!=row['expected'] and other['source']['shot']>row['source']['shot']]
            expected_end=min(later)-1 if later else count-1
            if row['last_active_shot']!=expected_end:
                raise ValueError(f"{name} {row['part']}/{row['aspect']}={row['expected']!r} must persist through shot {expected_end}; "
                                 'omitted mentions do not terminate a requirement')
        # Reject conflicting simultaneous scalar colors/ages/builds instead of
        # arbitrarily selecting one value. Multiple details/types remain legal.
        for shot in range(count):
            active=[row for row in entity['appearance'] if row['source']['shot']<=shot<=row['last_active_shot']]
            for part in PARTS:
                for aspect in ('color','age','build'):
                    values={row['expected'] for row in active if (row['part'],row['aspect'])==(part,aspect)}
                    if len(values)>1: raise ValueError('Conflicting active appearance predicates require clarification')
    return {'schema_version':1,'prompts':list(prompts),'script_fingerprint':fingerprint(prompts),
        'entities':result['entities'],'semantically_validated':False,
        'limits':['Exact source checking does not prove correct attribution, presence interpretation or completeness.',
                  'Appearance becomes active at its source shot; it is never imposed on earlier shots.',
                  'No appearance-only observation is a trusted identity reference.']}


def compile_entities(client,prompts):
    inputs=(TemporalInput.from_value(kind='fixed_authored_script',identifier='native_script',
        value=prompts,available_after_shot=-1,authored=True),)
    packet={'task':
        'Extract ALL named recurring characters (people/animals) and their explicit appearance predicates from the authored prompts. '
        'Use exact names and exact source quotes. Never infer gender, age, build or clothing from a name, role or scene. '
        'Split color/style/type into atomic rows: curly red hair gives hair/style=curly and hair/color=red; '
        'long tan coat gives garment/detail=long, garment/color=tan and garment/type=coat. '
        'Each expected string must be verbatim within its source quote. Attribute a pronoun only when its referent is clear. '
        'Include explicitly stated age, build and subject type as well as hair and garments. '
        'Appearance starts at its first source shot. last_active_shot MUST be the final shot unless a later explicit '
        'different value of the same part/aspect replaces it; in that case it is one shot before the replacement. '
        'A later shot that omits the description does not end it. Detail predicates persist to the final shot. '
        'For each character, include EVERY shot in presence with required, ambiguous or not_required. '
        'Required means actually depicted by that shot, not merely mentioned in a recollection. '
        'For required/ambiguous presence quote the exact supporting text. For not_required without a mention use empty quote. '
        'Return no entities if no named characters are specified. Do not invent names or positive appearance descriptions. Strict JSON.',
        'authored_prompts':{str(i):p for i,p in enumerate(prompts)}}
    failures=[]
    for attempt in range(2):
        response=client.call(prompt=json.dumps(packet,ensure_ascii=False),image_paths=[],schema=SCHEMA,
            call_kind='native_script_entities_quotes_v2' if attempt==0 else 'native_script_entities_correction_v1',
            shot_index=0,temporal_inputs=inputs)
        try:
            result=validate_compilation(prompts,response['parsed'])
            result['compiler_attempts']=attempt+1;result['compiler_validation_failures']=failures
            return result
        except ValueError as error:
            failures.append({'attempt':attempt+1,'error':str(error),'parsed':response['parsed']})
            if attempt:raise
            packet.update(previous_invalid_proposal=response['parsed'],program_validation_error=str(error),
                correction_rule='Correct the entire proposal once. Recheck ALL literal expected values and temporal endpoints. '
                    'Do not replace invalid words with another inferred trait; every value must occur verbatim in its own source quote.')


def entity_specs_at(compiled,shot):
    specs=[];requirements={}
    for entity in compiled['entities']:
        active=[row for row in entity['appearance'] if row['source']['shot']<=shot<=row['last_active_shot']]
        # The description is assembled from exact authored values; labels are
        # syntax, not new traits. Empty appearance is explicitly unassessed.
        description='; '.join(f"{row['part']} {row['aspect']}: {row['expected']}" for row in active)
        rows=freeze_requirements(description,active) if active else []
        if rows: validate_required_aspects(description,rows)
        presence={row['shot']:row['status'] for row in entity['presence'] if row['status']!='not_required'}
        specs.append(EntitySpec(entity['name'],description,shots=(),presence=presence))
        requirements[entity['name']]=rows
    return specs,requirements


def repair_prompt(caption,name,requirements):
    """Keep authored action/scene verbatim and append only fixed attribute values."""
    clauses=[]
    for row in requirements:
        part=row['part'];aspect=row['aspect'];expected=row['expected']
        clauses.append(f"{name}'s {part} {aspect} is {expected}.")
    return caption.rstrip()+(' '+' '.join(clauses) if clauses else '')
