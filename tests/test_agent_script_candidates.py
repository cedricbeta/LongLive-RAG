from copy import deepcopy
import pytest
from utils.agent_script_candidates import physical_candidates,compile_candidate_assignments
from utils.agent_typed_script import require_typed_ready


def cast(name,shots):return {'name':name,'name_source':{'shot':0,'quote':name},'presence':{str(k):'required' for k in range(shots)}}

def decisions(rows,name):return {r['id']:{'owners':[name],'status':'appearance','reason':'Explicitly described'} for r in rows}


def test_candidates_have_exact_offsets_and_types_without_model_quotes():
    prompts=['Iris has curly red hair and wears a long tan coat.']
    rows=physical_candidates(prompts)
    assert [r['quote'] for r in rows]==['curly red hair','long tan coat']
    for r in rows:assert prompts[r['shot']][r['start_char']:r['end_char']]==r['quote']
    d=compile_candidate_assignments(prompts,[cast('Iris',1)],rows,decisions(rows,'Iris'));require_typed_ready(d)
    assert ('hair','style','long') not in {(r['part'],r['aspect'],r['expected']) for r in d['entities'][0]['appearance']}


def test_multientity_explicit_owner_conflict_rejected():
    prompts=['Iris and Miles meet. Iris has red hair. Miles wears a blue coat.']
    rows=physical_candidates(prompts);bad=decisions(rows,'Miles')
    with pytest.raises(ValueError,match='directly named'):compile_candidate_assignments(prompts,[cast('Iris',1),cast('Miles',1)],rows,bad)


@pytest.mark.parametrize('change',['omit','modify','nonappearance','ambiguous','unknown_modifier'])
def test_model_cannot_discard_or_mutate_physical_candidates(change):
    prompts=['Iris has red hair.' if change!='unknown_modifier' else 'Iris has iridescent hair.'];rows=physical_candidates(prompts);d=decisions(rows,'Iris')
    if change=='omit':
        d.clear()
        with pytest.raises(ValueError,match='Every program phrase'):compile_candidate_assignments(prompts,[cast('Iris',1)],rows,d)
    elif change=='modify':
        rows[0]['quote']='blue hair'
        with pytest.raises(ValueError,match='inventory differs'):compile_candidate_assignments(prompts,[cast('Iris',1)],rows,d)
    else:
        if change!='unknown_modifier':d[rows[0]['id']].update(owners=[],status=change)
        compiled=compile_candidate_assignments(prompts,[cast('Iris',1)],rows,d)
        assert compiled['unresolved'] and not compiled['ready_for_development']


def test_future_description_is_not_added_to_earlier_shots():
    prompts=['Iris walks.','Her curly red hair catches light.','Her moss green coat moves.']
    rows=physical_candidates(prompts);d=compile_candidate_assignments(prompts,[cast('Iris',3)],rows,decisions(rows,'Iris'));require_typed_ready(d)
    assert sorted({r['source']['shot'] for r in d['entities'][0]['appearance']})==[1,2]
