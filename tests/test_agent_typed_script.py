"""Constructed text contracts, not human labels or video efficacy evidence."""
from copy import deepcopy
import pytest
from utils.agent_typed_script import parse_physical_phrase,compile_typed_proposal,compile_typed_entities,require_typed_ready
from utils.agent_script_entities import entity_specs_at


def atoms(text):
    d=parse_physical_phrase(text)
    assert d['unresolved'] is None,d
    return {(r['part'],r['aspect'],r['expected']) for r in d['atoms']}


@pytest.mark.parametrize('text,expected',[
 ('a long tan coat',{('garment','type','coat'),('garment','color','tan'),('garment','detail','long')}),
 ('curly red hair',{('hair','color','red'),('hair','style','curly')}),
 ('a buzz cut',{('hair','style','buzz cut')}),
 ('a middle-aged woman',{('body','age','middle-aged'),('body','type','woman')}),
 ('a slim man',{('body','build','slim'),('body','type','man')}),
 ('a moss green coat',{('garment','color','moss green'),('garment','type','coat')}),
 ('slightly longer dark hair',{('hair','style','slightly longer'),('hair','color','dark')}),
 ('a navy knit sweater',{('garment','color','navy'),('garment','detail','knit'),('garment','type','sweater')}),
 ('a denim jacket',{('garment','detail','denim'),('garment','type','jacket')}),
 ('green eyes',{('face','color','green')}),
 ('silver bob hair',{('hair','color','silver'),('hair','style','bob')}),
 ('a shaved head',{('hair','style','shaved head')}),
])
def test_head_derives_type_and_preserves_literal_words(text,expected):assert atoms(text)==expected


@pytest.mark.parametrize('text',['auburn hair catching the soft natural light','a not red coat','an iridescent coat','a red and blue coat','a long tan coat and red hair'])
def test_unsupported_or_nonphysical_phrases_do_not_emit_partial_facts(text):
    d=parse_physical_phrase(text);assert d['unresolved'] and not d['atoms']


def proposal(prompts,quotes):
    return {'entities':[{'name':'Iris','name_source':{'shot':0,'quote':'Iris'},
        'presence':{str(k):'required' for k in range(len(prompts))},
        'appearance_quotes':[{'shot':k,'quote':q} for k,q in quotes],'unresolved_appearance':[]}]}


def test_temporal_appearance_never_uses_future_age_or_coat():
    prompts=['Iris walks.','Iris has curly red hair.','Her moss green coat moves.','Iris waits.']
    d=compile_typed_proposal(prompts,proposal(prompts,[(1,'curly red hair'),(2,'moss green coat')]))
    require_typed_ready(d)
    assert not entity_specs_at(d,0)[1]['Iris']
    assert {r['part'] for r in entity_specs_at(d,1)[1]['Iris']}=={'hair'}
    assert {r['part'] for r in entity_specs_at(d,3)[1]['Iris']}=={'hair','garment'}
    assert all(r['aspect']!='age' for r in d['entities'][0]['appearance'])


def test_omitted_known_modifier_is_unresolved_even_when_model_quote_is_literal():
    prompts=['Iris wears a long tan coat.']
    d=compile_typed_proposal(prompts,proposal(prompts,[(0,'tan coat')]))
    assert not d['ready_for_development']
    assert any('omitted' in x['reason'] for x in d['unresolved'])
    with pytest.raises(ValueError,match='Unresolved'):require_typed_ready(d)


def test_unknown_modifier_cannot_be_dropped_to_claim_coverage():
    prompts=['Iris has iridescent hair.']
    d=compile_typed_proposal(prompts,proposal(prompts,[(0,'hair')]))
    assert not d['ready_for_development'] and any('Unknown adjacent' in x['reason'] for x in d['unresolved'])


def test_negative_context_is_not_a_positive_constraint():
    prompts=['Iris does not have red hair.']
    d=compile_typed_proposal(prompts,proposal(prompts,[(0,'red hair')]))
    assert not d['ready_for_development']


def test_explicit_replacement_and_return_to_prior_value():
    prompts=['Iris has red hair.','Iris now has blue hair.','Iris again has red hair.']
    d=compile_typed_proposal(prompts,proposal(prompts,[(0,'red hair'),(1,'blue hair'),(2,'red hair')]))
    require_typed_ready(d)
    assert [row['last_active_shot'] for row in d['entities'][0]['appearance']]==[0,1,2]


def test_no_name_substring_or_other_actor_ownership():
    prompts=['Iris and Miles meet. Iris has red hair.']
    p=proposal(prompts,[(0,'Miles')]);other=deepcopy(p['entities'][0]);other['name']='Miles';other['name_source']['quote']='Miles';other['appearance_quotes']=[];p['entities'].append(other)
    with pytest.raises(ValueError,match='another named'):compile_typed_proposal(prompts,p)
    assert parse_physical_phrase('hair',entity_name='Mary Jones')['atoms']==[]


def test_model_correction_is_bounded_and_unresolved_phrases_are_retained():
    prompts=['Iris has iridescent hair.'];p=proposal(prompts,[(0,'iridescent hair')])
    class Client:
        def __init__(self):self.calls=[]
        def call(self,**kw):self.calls.append(kw);return {'parsed':deepcopy(p)}
    c=Client();d=compile_typed_entities(c,prompts)
    assert len(c.calls)==2 and not d['ready_for_development'] and d['unresolved']
    assert 'program_findings' in c.calls[1]['prompt']
