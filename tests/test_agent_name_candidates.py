from copy import deepcopy
import pytest
from utils.agent_name_candidates import name_candidates, classified_cast, compile_name_candidate_entities
from utils.agent_script_candidates import physical_candidates
from utils.agent_typed_script import require_typed_ready


def decision(kind, count, presence=None):
    return {'kind': kind, 'presence': {str(k): presence or ('required' if kind == 'character' else 'not_required') for k in range(count)},
            'reason': 'Constructed contract fixture'}


def test_multiword_possessive_offsets_and_repeated_names_are_exact():
    prompts = ["At dawn, Mary Jane's coat moves.", 'Mary Jane waits.']
    rows = name_candidates(prompts)
    assert [r['name'] for r in rows] == ['At', 'Mary Jane']
    assert len(rows[1]['sources']) == 2
    for row in rows:
        for span in row['sources']:
            assert prompts[span['shot']][span['start_char']:span['end_char']] == row['name'] == span['quote']


def test_capitalization_does_not_automatically_label_a_character():
    prompts = ['The Rose Hotel stands by London.']
    rows = name_candidates(prompts)
    assert [r['name'] for r in rows] == ['The', 'Rose Hotel', 'London']
    cast, unresolved, audit = classified_cast(prompts, rows, {r['id']: decision('noncharacter', 1) for r in rows})
    assert not cast and unresolved and len(audit) == 3


@pytest.mark.parametrize('mutation', ['omit_id', 'rename', 'offset'])
def test_name_inventory_cannot_be_omitted_or_mutated(mutation):
    prompts = ['Zara waits.']; rows = name_candidates(prompts); decisions = {'n000': decision('character', 1)}
    if mutation == 'omit_id':
        decisions.clear()
    elif mutation == 'rename':
        rows[0]['name'] = 'Someone else'
    else:
        rows[0]['sources'][0]['start_char'] += 1
    with pytest.raises(ValueError, match='inventory|Every name'):
        classified_cast(prompts, rows, decisions)


def test_noncharacter_presence_is_rejected():
    prompts = ['Soft light falls.']; rows = name_candidates(prompts)
    with pytest.raises(ValueError, match='noncharacter'):
        classified_cast(prompts, rows, {'n000': decision('noncharacter', 1, 'required')})


def test_presence_must_cover_all_authored_shots():
    prompts = ['Zara waits.', 'Zara walks.']; rows = name_candidates(prompts)
    with pytest.raises(ValueError, match='Every authored shot'):
        classified_cast(prompts, rows, {'n000': decision('character', 1)})


def test_inventory_budget_rejects_instead_of_truncating():
    prompts = [' '.join(f'Name{i} waits.' for i in range(33))]
    with pytest.raises(ValueError, match='Too many name candidates'):
        name_candidates(prompts)


class RecordedFixtureClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return {'parsed': deepcopy(self.replies.pop(0))}


def test_two_stages_preserve_literal_appearance_and_future_activation():
    prompts = ['Zara waits.', 'Her curly red hair moves.', 'Her moss green coat sways.']
    names = name_candidates(prompts)
    cast = {r['id']: decision('character' if r['name'] == 'Zara' else 'noncharacter', 3) for r in names}
    owners = {r['id']: {'owners': ['Zara'], 'status': 'appearance', 'reason': 'Constructed explicit fixture'} for r in physical_candidates(prompts)}
    client = RecordedFixtureClient([cast, owners])
    result = compile_name_candidate_entities(client, prompts)
    require_typed_ready(result)
    assert result['compiler_protocol'] == 'typed_name_candidates_v6'
    assert not result['semantically_validated']
    assert len(client.calls) == 2
    assert result['entities'][0]['name_source'] == {'shot': 0, 'quote': 'Zara'}
    appearance = result['entities'][0]['appearance']
    assert {r['source']['shot'] for r in appearance if r['part'] == 'hair'} == {1}
    assert {r['source']['shot'] for r in appearance if r['part'] == 'garment'} == {2}
    assert not any(r['part'] == 'body' for r in appearance)


def test_ambiguous_name_blocks_otherwise_compiled_requirements():
    prompts = ['Zara wears a red coat beside Echo.']; names = name_candidates(prompts)
    cast = {r['id']: decision('character' if r['name'] == 'Zara' else 'ambiguous', 1) for r in names}
    owners = {r['id']: {'owners': ['Zara'], 'status': 'appearance', 'reason': 'Explicit fixture'} for r in physical_candidates(prompts)}
    result = compile_name_candidate_entities(RecordedFixtureClient([cast, owners]), prompts)
    assert any(r['reason'] == 'Ambiguous name classification' for r in result['unresolved'])
    with pytest.raises(ValueError, match='Unresolved'):
        require_typed_ready(result)


def test_lowercase_unrecognized_name_does_not_silently_pass():
    client = RecordedFixtureClient([])
    result = compile_name_candidate_entities(client, ['zara wears a red coat.'])
    assert not client.calls and not result['ready_for_development']
    with pytest.raises(ValueError, match='Unresolved'):
        require_typed_ready(result)
