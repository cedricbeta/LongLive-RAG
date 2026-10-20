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


def compile_entities_by_quotes(client,prompts):
    """Bounded cast extraction followed by isolated, verbatim per-entity atoms.

    This avoids one free-form joint list mixing different actors' attributes.
    Temporal endpoints are program-derived, never proposed by the language model.
    """
    import re
    from utils.agent_attribute_diagnosis import REQUIREMENTS_SCHEMA
    presence={'type':'object','additionalProperties':False,'properties':{
        str(k):{'type':'string','enum':['required','ambiguous','not_required']} for k in range(len(prompts))},
        'required':[str(k) for k in range(len(prompts))]}
    cast_schema={'type':'object','additionalProperties':False,'properties':{'entities':{'type':'array','maxItems':8,'items':{
        'type':'object','additionalProperties':False,'properties':{'name':{'type':'string','minLength':1,'maxLength':80},
        'name_source':SPAN,'presence':presence,'appearance_quotes':{'type':'array','maxItems':12,'items':SPAN}},
        'required':['name','name_source','presence','appearance_quotes']}}},'required':['entities']}
    inputs=(TemporalInput.from_value(kind='fixed_authored_script',identifier='native_script',value=prompts,available_after_shot=-1,authored=True),)
    logs=[]
    def bounded(packet,schema,kind,validator):
        for attempt in range(2):
            response=client.call(prompt=json.dumps(packet,ensure_ascii=False),image_paths=[],schema=schema,
                call_kind=kind if not attempt else kind+'_correction',shot_index=0,temporal_inputs=inputs)
            try:
                result=validator(response['parsed']);logs.append({'kind':kind,'attempt':attempt+1,'valid':True});return result
            except ValueError as error:
                logs.append({'kind':kind,'attempt':attempt+1,'valid':False,'error':str(error)})
                if attempt:raise
                packet={**packet,'previous_invalid':response['parsed'],'program_error':str(error),
                    'correction':'One correction allowed. Use only the original text; do not infer or invent missing traits.'}
    def validate_cast(parsed):
        _validate_json(parsed,cast_schema)
        entities=parsed['entities'];names=[e['name'] for e in entities]
        if len(set(names))!=len(names):raise ValueError('Duplicate character name')
        for entity in entities:
            name=entity['name'];source=entity['name_source'];k=source['shot']
            if name!=name.strip() or not 0<=k<len(prompts) or source['quote'] not in prompts[k] or name not in source['quote']:
                raise ValueError('Name needs an exact authored quote')
            for span in entity['appearance_quotes']:
                k=span['shot'];quote=span['quote']
                if not 0<=k<len(prompts) or quote not in prompts[k]:raise ValueError('Appearance quote is not verbatim')
                if any(other!=name and re.search(r'(?<!\w)'+re.escape(other)+r'(?!\w)',quote) for other in names):
                    raise ValueError('Isolate this character: appearance quote includes another named character')
        return deepcopy(entities)
    cast=bounded({'task':'List all named recurring characters (people/animals) from the fixed prompts. '
        'Return exact names and exact name_source. For every shot classify actually depicted presence as required, ambiguous, or not_required. '
        'appearance_quotes contains only minimal VERBATIM physical descriptions of this character, each with its source shot. '
        'Include age/build, hair and garments when explicitly written. Copy text only; never add a trait or include another actor in the quote. '
        'Exclude actions/locations from appearance quotes. Do not duplicate repeated identical descriptions. Empty appearance_quotes is allowed '
        'when none is explicit. Do not split appearance into attributes in this call. Strict JSON.',
        'prompts':{str(k):v for k,v in enumerate(prompts)}},cast_schema,'native_cast_quotes_v3',validate_cast)
    entities=[]
    function_words=set('a an the and or with has have is are his her their its he she they wearing wears worn in of hair garment clothing face body'.split())
    for entity in cast:
        name=entity['name'];spans=entity['appearance_quotes'];description='; '.join(s['quote'] for s in spans)
        def validate_atoms(parsed):
            _validate_json(parsed,REQUIREMENTS_SCHEMA)
            frozen=freeze_requirements(description,parsed['requirements'])
            validate_required_aspects(description,frozen)
            covered=set()
            for row in frozen:
                if not re.search(r'(?<!\w)'+re.escape(row['expected'])+r'(?!\w)',description):
                    raise ValueError('Expected value must quote complete source words, not a substring of another word')
                covered.update(re.findall(r"[\w]+",row['expected'].lower()))
            tokens=set(re.findall(r"[\w]+",description.lower()))
            leftover=tokens-covered-function_words-set(re.findall(r"[\w]+",name.lower()))
            if leftover:raise ValueError('Uncovered words in physical-description quotes: '+', '.join(sorted(leftover)))
            return frozen
        atoms=bounded({'task':'Split ONLY this character\'s quoted physical descriptions into all atomic requirements. '
            'Separate hair color/style; garment color/type/detail; explicit body age/build/type. '
            'Every expected value must be an exact sequence of COMPLETE words in the quoted text. '
            'Do not infer hair texture, hair color, age, gender, length or any missing detail. '
            'Cover every descriptive word. Output each part/aspect/expected tuple once. Do not repeat the quotes or report time endpoints. Strict JSON.',
            'entity':name,'verbatim_physical_descriptions':[s['quote'] for s in spans]},
            REQUIREMENTS_SCHEMA,'native_entity_atoms_v3',validate_atoms) if spans else []
        rows=[]
        for atom in atoms:
            sources=[s for s in spans if re.search(r'(?<!\w)'+re.escape(atom['expected'])+r'(?!\w)',s['quote'])]
            source=min(sources,key=lambda s:s['shot'])
            rows.append({'part':atom['part'],'aspect':atom['aspect'],'expected':atom['expected'],
                         'source':source,'last_active_shot':len(prompts)-1})
        for row in rows:
            later=[other['source']['shot'] for other in rows if row['aspect'] in ('color','style','type','age','build')
                and (other['part'],other['aspect'])==(row['part'],row['aspect']) and other['expected']!=row['expected']
                and other['source']['shot']>row['source']['shot']]
            if later:row['last_active_shot']=min(later)-1
        entities.append({'name':name,'name_source':entity['name_source'],
            'presence':[{'shot':k,'status':entity['presence'][str(k)],'quote':prompts[k] if entity['presence'][str(k)]!='not_required' else ''}
                        for k in range(len(prompts))],'appearance':rows})
    result=validate_compilation(prompts,{'entities':entities})
    result.update(compiler_protocol='cast_quotes_then_isolated_atoms_v3',cast_proposal=cast,compiler_call_log=logs,
        lexical_coverage_scope='Only extracted description quotes, not a proof that every relevant script span was extracted')
    return result
