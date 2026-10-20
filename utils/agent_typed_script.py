"""Opt-in noun-phrase compiler with program-derived appearance types.

The language model selects exact physical noun phrases and their owners. It
cannot assign attribute types/values. The finite English grammar below derives
those from the phrase head and retains unsupported phrases as unresolved. This
is not a completeness or coreference guarantee and is not a formal evaluator.
"""
from copy import deepcopy
import json
import re

from utils.agent_script_entities import SPAN, validate_compilation
from utils.agent_runtime import TemporalInput
from utils.agent_vlm import _validate_json

# Literal English vocabulary. No normalization to color families is applied to
# the expected value, and no character-specific descriptor is supplied here.
COLORS=set('red auburn ginger copper crimson scarlet burgundy maroon orange yellow blonde blond golden gold green olive emerald sage mint lime blue navy indigo cobalt azure teal cyan purple violet lavender plum lilac mauve pink magenta rose fuchsia salmon coral brown chestnut hazel bronze chocolate tan beige khaki camel taupe cream ivory ecru black ebony charcoal white silver platinum grey gray slate'.split())
COLOR_MODIFIERS=set('dark light deep pale bright vivid muted dusty warm cool moss forest sky ash jet off'.split())
HAIR_STYLE=set('curly straight wavy short long longer shorter bob shaved buzz bald ponytail braids braided cropped tidy neat messy slicked sleek spiky tousled frizzy coiled kinky afro shoulder-length waist-length chin-length close-cropped tightly loosely slightly neatly very'.split())
AGES=set('young younger old older elderly middle-aged teenage adult adolescent mature youthful'.split())
BUILDS=set('tall taller short shorter slim slender petite stocky muscular athletic lean broad heavyset thin small large stout average-height broad-shouldered'.split())
DETAILS=set('long short longer shorter knee-length ankle-length full-length cropped loose tight fitted oversized sleeveless short-sleeved long-sleeved hooded collared buttoned unbuttoned open closed zip zip-up zipped knit knitted wool woolen woollen cotton linen denim leather silk satin velvet corduroy striped checked checkered plaid floral patterned plain solid polka-dot dotted embroidered frayed worn ripped tattered work formal casual double-breasted single-breasted belted trench puffer quilted'.split())
GARMENTS='coat jacket shirt sweater dress blouse t-shirt tee hoodie blazer cardigan turtleneck vest apron uniform top jumper overcoat parka suit gown robe tunic polo pullover skirt trousers pants shorts jeans scarf hat cap shoes boots gloves socks'.split()
BODIES='woman man women men girl boy person people child children adult teenager baby infant dog cat horse bird rabbit fox wolf bear lion tiger'.split()
# Longest matches first; special hair nouns cannot become colors.
HEADS={**{word:('garment','type') for word in GARMENTS},
       **{word:('body','type') for word in BODIES},
       'hair':('hair',None),'hairstyle':('hair',None),
       'buzz cut':('hair','style'),'shaved head':('hair','style'),
       'ponytail':('hair','style'),'braids':('hair','style'),'afro':('hair','style'),
       'eyes':('face',None),'beard':('face','detail'),'moustache':('face','detail'),
       'mustache':('face','detail'),'glasses':('other','type'),'spectacles':('other','type'),
       'fur':('body',None),'feathers':('body',None)}
TOKEN=re.compile(r"[\w]+(?:[-'][\w]+)*")
NEGATIVE=set('no not never without neither imaginary imagined hypothetical'.split())
MODIFIERS=COLORS|COLOR_MODIFIERS|HAIR_STYLE|AGES|BUILDS|DETAILS
BOUNDARIES=set('a an the his her their its my your with and has have having wearing wears wore wear worn in of sports sporting is are was were appears featuring'.split())
HEAD_PATTERN=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in sorted(HEADS,key=len,reverse=True))+r')(?!\w)',re.I)


def audit_known_phrase_coverage(prompts,entities):
    """Find omitted recognized modifiers and unknown adjacent modifiers.

    This inventory is independent of the model's selected quotes. Repeated
    identical values may be covered by earlier declarations; owner/coreference
    and unknown physical noun heads remain outside this finite grammar audit.
    """
    unresolved=[];audit=[]
    for shot,text in enumerate(prompts):
        tokens=list(TOKEN.finditer(text));positions={m.start():i for i,m in enumerate(tokens)}
        for head in HEAD_PATTERN.finditer(text):
            end=head.end();start=head.start();index=positions[start]
            while index>0 and tokens[index-1].group().lower() in MODIFIERS and text[tokens[index-1].end():start].isspace():
                index-=1;start=tokens[index].start()
            quote=text[start:end];parsed=parse_physical_phrase(quote)
            context=text[:end].rsplit('.',1)[-1].rsplit(';',1)[-1].rsplit(',',1)[-1]
            # Any supported phrase under an unresolved negation is withheld.
            if set(m.group().lower() for m in TOKEN.finditer(context))&NEGATIVE:
                reason='Negated/hypothetical physical mention is not compiled as a positive predicate'
            elif index>0 and text[tokens[index-1].end():start].isspace() and tokens[index-1].group().lower() not in BOUNDARIES and not tokens[index-1].group()[0].isupper():
                start=tokens[index-1].start();quote=text[start:end];reason='Unknown adjacent modifier or unsupported syntactic context'
            elif parsed['unresolved']:reason=parsed['unresolved']
            else:
                available={(r['part'],r['aspect'],r['expected']) for e in entities for r in e['appearance'] if r['source']['shot']<=shot<=r['last_active_shot']}
                required={(r['part'],r['aspect'],r['expected']) for r in parsed['atoms']}
                reason='Recognized authored predicate omitted from active extraction' if required-available else None
            row={'shot':shot,'quote':quote,'reason':reason,'owner_validated':False}
            audit.append(row)
            if reason:unresolved.append({'entity':None,'source':{'shot':shot,'quote':quote},'reason':reason})
    return audit,unresolved


def parse_physical_phrase(quote, *, entity_name=''):
    """Return literal atoms only when the whole selected noun phrase is supported.

    Unknown words are not discarded or mislabeled as arbitrary detail. No
    predicate is emitted from an unsupported phrase, including its known subset.
    """
    tokens=list(TOKEN.finditer(quote));words=[t.group().lower() for t in tokens]
    if not words:return {'atoms':[],'unresolved':'Empty physical phrase'}
    start=0
    name_tokens=[m.group().lower() for m in TOKEN.finditer(entity_name)]
    if name_tokens and words[:len(name_tokens)]==name_tokens:start=len(name_tokens)
    elif name_tokens and len(words)>=len(name_tokens) and words[:len(name_tokens)-1]+[words[len(name_tokens)-1].removesuffix("'s")] == name_tokens:
        start=len(name_tokens)
    if start<len(words) and words[start] in ('a','an','the','his','her','their','its'):start+=1
    head=None
    for key in sorted(HEADS,key=lambda x:len(x.split()),reverse=True):
        units=key.split()
        if len(words)-start>=len(units) and words[-len(units):]==units:
            head=key;head_start=len(words)-len(units);break
    if head is None:return {'atoms':[],'unresolved':'No supported final physical noun head'}
    part,head_aspect=HEADS[head];prefix=words[start:head_start]
    if set(words)&NEGATIVE:return {'atoms':[],'unresolved':'Negated or hypothetical description needs explicit semantics'}
    labels=[]
    for index,word in enumerate(prefix):
        if word in COLORS:label='color'
        elif word in COLOR_MODIFIERS:
            # A modifier combines with an adjacent literal color, or dark/light
            # can be the explicit color by itself. Material denim stays detail.
            neighboring=(index+1<len(prefix) and prefix[index+1] in COLORS)
            if neighboring or word in ('dark','light'):label='color'
            else:return {'atoms':[],'unresolved':f'Unbound color modifier: {word}'}
        elif part=='hair' and word in HAIR_STYLE:label='style'
        elif part=='body' and head_aspect=='type' and word in AGES:label='age'
        elif part=='body' and head_aspect=='type' and word in BUILDS:label='build'
        elif part=='garment' and word in DETAILS:label='detail'
        elif part=='face' and word in HAIR_STYLE:label='detail'
        else:return {'atoms':[],'unresolved':f'Unsupported {part} modifier: {word}'}
        labels.append(label)
    def literal(a,b):return quote[tokens[a].start():tokens[b-1].end()]
    atoms=[]
    k=0
    while k<len(labels):
        end=k+1
        while end<len(labels) and labels[end]==labels[k]:end+=1
        values=prefix[k:end]
        if labels[k]=='color' and sum(x in COLORS for x in values)>1:
            return {'atoms':[],'unresolved':'Multiple color words require explicit multi-color semantics'}
        atoms.append({'part':part,'aspect':labels[k],'expected':literal(start+k,start+end)})
        k=end
    if head_aspect:atoms.append({'part':part,'aspect':head_aspect,'expected':literal(head_start,len(words))})
    return {'atoms':atoms,'head':literal(head_start,len(words)),'part':part,'unresolved':None}


def compile_typed_proposal(prompts,parsed):
    """Validate ownership quotes, derive types, and preserve unresolved input."""
    count=len(prompts);names=[e['name'] for e in parsed['entities']]
    if len(set(names))!=len(names):raise ValueError('Duplicate named entity')
    entities=[];unresolved=[];phrase_audit=[]
    for entity in parsed['entities']:
        name=entity['name'];source=entity['name_source'];k=source['shot']
        if name!=name.strip() or not 0<=k<count or source['quote'] not in prompts[k] or not re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',source['quote']):
            raise ValueError('Exact named source required')
        if set(entity['presence'])!={str(k) for k in range(count)}:raise ValueError('Every authored shot needs presence')
        rows=[]
        for span in entity['appearance_quotes']:
            k,quote=span['shot'],span['quote']
            if not 0<=k<count or quote not in prompts[k]:raise ValueError('Appearance quote must be verbatim')
            if any(other!=name and re.search(r'(?<!\w)'+re.escape(other)+r'(?!\w)',quote) for other in names):
                raise ValueError('Appearance phrase includes another named entity')
            parsed_phrase=parse_physical_phrase(quote,entity_name=name)
            phrase_audit.append({'entity':name,'source':deepcopy(span),**parsed_phrase})
            if parsed_phrase['unresolved']:
                unresolved.append({'entity':name,'source':deepcopy(span),'reason':parsed_phrase['unresolved']});continue
            for atom in parsed_phrase['atoms']:
                rows.append({**atom,'source':deepcopy(span),'last_active_shot':count-1})
        for span in entity.get('unresolved_appearance',[]):
            k=span['shot']
            if not 0<=k<count or span['quote'] not in prompts[k]:raise ValueError('Unresolved text must be verbatim')
            unresolved.append({'entity':name,'source':{'shot':k,'quote':span['quote']},'reason':span['reason']})
        # Repeated statements are evidence for the same active value, not extra
        # predicates. Retain a return to a value after an explicit replacement.
        ordered=sorted(rows,key=lambda x:x['source']['shot']);rows=[];last_values={}
        for row in ordered:
            key=(row['part'],row['aspect']);value=row['expected']
            if last_values.get(key)==value:continue
            rows.append(row);last_values[key]=value
        for row in rows:
            later=[other['source']['shot'] for other in rows if row['aspect'] in ('color','style','type','age','build')
                and (other['part'],other['aspect'])==(row['part'],row['aspect']) and other['expected']!=row['expected']
                and other['source']['shot']>row['source']['shot']]
            if later:row['last_active_shot']=min(later)-1
        entities.append({'name':name,'name_source':source,
            'presence':[{'shot':k,'status':entity['presence'][str(k)],'quote':prompts[k] if entity['presence'][str(k)]!='not_required' else ''} for k in range(count)],
            'appearance':rows})
    result=validate_compilation(prompts,{'entities':entities})
    coverage_audit,missing=audit_known_phrase_coverage(prompts,entities);unresolved.extend(missing)
    result.update(compiler_protocol='typed_noun_phrases_v4',phrase_audit=phrase_audit,unresolved=unresolved,
                  ready_for_development=not unresolved,semantically_validated=False,
                  cast_proposal=deepcopy(parsed['entities']),known_phrase_coverage_audit=coverage_audit)
    result['limits'] += ['Finite typed English grammar: unsupported phrases remain unresolved and prevent generation.',
        'Known-head coverage catches omitted recognized modifiers, but does not prove owner attribution or unknown-head completeness. No formal semantic-accuracy claim.',
        'Lighting and action clauses are not forced into appearance categories; full authored prompts remain unchanged.']
    return result


def require_typed_ready(compiled):
    if compiled.get('compiler_protocol') not in ('typed_noun_phrases_v4','typed_candidates_v5','typed_name_candidates_v6','typed_name_candidates_v7') or not compiled.get('ready_for_development') or compiled.get('unresolved'):
        raise ValueError('Unresolved typed script requirements; rendering is not authorized by this compiler result')


def compile_typed_entities(client,prompts):
    presence={'type':'object','additionalProperties':False,'properties':{
        str(k):{'type':'string','enum':['required','ambiguous','not_required']} for k in range(len(prompts))},'required':[str(k) for k in range(len(prompts))]}
    unresolved_span={'type':'object','additionalProperties':False,'properties':{
        'shot':{'type':'integer','minimum':0},'quote':{'type':'string','minLength':1},'reason':{'type':'string','minLength':1}},'required':['shot','quote','reason']}
    schema={'type':'object','additionalProperties':False,'properties':{'entities':{'type':'array','maxItems':8,'items':{
        'type':'object','additionalProperties':False,'properties':{'name':{'type':'string','minLength':1,'maxLength':80},
        'name_source':SPAN,'presence':presence,'appearance_quotes':{'type':'array','maxItems':24,'items':SPAN},
        'unresolved_appearance':{'type':'array','maxItems':12,'items':unresolved_span}},
        'required':['name','name_source','presence','appearance_quotes','unresolved_appearance']}}},'required':['entities']}
    inputs=(TemporalInput.from_value(kind='fixed_authored_script',identifier='native_script',value=prompts,available_after_shot=-1,authored=True),)
    packet={'task':'Extract ALL named recurring characters and explicit physical descriptions from the fixed script. '
        'Use exact names and name_source. For each shot classify depicted presence as required, ambiguous or not_required. '
        'appearance_quotes must be minimal VERBATIM NOUN PHRASES with one final physical head: split a young woman with curly red hair and a long tan coat '
        'into a young woman / curly red hair / a long tan coat. Keep complete noun modifiers and exact source shot. '
        'Include body type, stated age/build, hair, garments and other explicit appearance. Do not infer anything from names or actions. '
        'Do not output attribute categories or invent values; the program derives them from heads. '
        'Do not include trailing actions or lighting: her red hair catching the light contributes red hair only. '
        'No duplicate identical descriptions. Attribute pronouns only when clear. '
        'Put any physical description you cannot safely isolate or attribute in unresolved_appearance with its exact quote and reason. '
        'Do not drop difficult appearance requirements. Keep actions, locations, and lighting out of appearance_quotes. Strict JSON.',
        'prompts':{str(k):p for k,p in enumerate(prompts)}}
    logs=[]
    for attempt in range(2):
        response=client.call(prompt=json.dumps(packet,ensure_ascii=False),image_paths=[],schema=schema,
            call_kind='native_typed_noun_phrases_v4' if not attempt else 'native_typed_noun_phrases_v4_correction',
            shot_index=0,temporal_inputs=inputs)
        _validate_json(response['parsed'],schema)
        try:
            result=compile_typed_proposal(prompts,response['parsed'])
            logs.append({'attempt':attempt+1,'unresolved':deepcopy(result['unresolved'])})
            if result['ready_for_development'] or attempt:
                result['compiler_call_log']=logs;return result
            reason=json.dumps(result['unresolved'],ensure_ascii=False)
        except ValueError as error:
            logs.append({'attempt':attempt+1,'error':str(error)})
            if attempt:raise
            reason=str(error)
        packet={**packet,'previous_proposal':response['parsed'],'program_findings':reason,
            'correction':'One correction allowed. Split exact noun phrases if needed. Do not drop a physical descriptor to make the validator pass. Unsupported descriptions must stay in unresolved_appearance. Do not add new words or reinterpret actions/lighting as physical traits.'}
