from copy import deepcopy
import pytest

from utils.agent_conditional_names import compile_conditional_name_entities
from utils.agent_name_candidates import compile_name_candidate_entities, name_candidates, classified_cast
from utils.agent_script_candidates import physical_candidates
from utils.agent_typed_script import require_typed_ready
from utils.agent_vlm import _validate_json


class Client:
    def __init__(self, replies):
        self.replies, self.calls = deepcopy(replies), []

    def call(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return {'parsed': self.replies.pop(0)}


def fixture():
    prompts = ['Zara waits.', 'Her curly red hair moves.', 'Her moss green coat sways.']
    cast = {row['id']: {'kind': 'character' if row['name'] == 'Zara' else 'noncharacter',
        'presence': {str(i): 'required' if row['name'] == 'Zara' else 'not_required' for i in range(3)},
        'reason': 'Constructed fixture'} for row in name_candidates(prompts)}
    owners = {row['id']: {'owners': ['Zara'], 'status': 'appearance', 'reason': 'Constructed fixture'}
        for row in physical_candidates(prompts)}
    return prompts, cast, owners


def test_only_first_response_schema_changes_and_owner_stage_is_identical():
    prompts, cast, owners = fixture()
    old, new = Client([cast, owners]), Client([cast, owners])
    before = compile_name_candidate_entities(old, prompts)
    after = compile_conditional_name_entities(new, prompts)
    require_typed_ready(after)
    assert before['compiler_protocol'] == 'typed_name_candidates_v6'
    assert after['compiler_protocol'] == 'typed_name_candidates_v7'
    assert not after['semantically_validated']
    assert old.calls[0]['prompt'] == new.calls[0]['prompt']
    assert old.calls[0]['temporal_inputs'] == new.calls[0]['temporal_inputs']
    assert old.calls[1] == new.calls[1]
    assert before['entities'] == after['entities']
    assert old.calls[0]['schema']['properties']['n001']['properties']['kind']['enum'] == ['character', 'noncharacter', 'ambiguous']
    assert new.calls[0]['call_kind'] == 'native_name_candidate_cast_v7_conditional'
    assert len(old.calls) == len(new.calls) == 2


@pytest.mark.parametrize('presence', ['required', 'ambiguous'])
def test_ignored_output_grammar_is_rejected_before_owner_call(presence):
    prompts, cast, owners = fixture()
    cast['n001']['presence']['1'] = presence
    client = Client([cast, owners])
    with pytest.raises(ValueError, match='exactly one bounded schema branch'):
        compile_conditional_name_entities(client, prompts)
    assert len(client.calls) == 1
    # The original semantic invariant is preserved independently of the schema.
    with pytest.raises(ValueError, match='noncharacter'):
        classified_cast(prompts, name_candidates(prompts), cast)


@pytest.mark.parametrize('mutation', ['missing_id', 'extra_id', 'missing_shot', 'invented_kind'])
def test_conditional_schema_keeps_all_existing_coverage_requirements(mutation):
    prompts, cast, owners = fixture()
    client = Client([cast, owners]); compile_conditional_name_entities(client, prompts)
    broken = deepcopy(cast)
    if mutation == 'missing_id': broken.pop('n001')
    elif mutation == 'extra_id': broken['invented'] = broken['n000']
    elif mutation == 'missing_shot': broken['n000']['presence'].pop('1')
    else: broken['n000']['kind'] = 'probably_a_character'
    with pytest.raises(ValueError): _validate_json(broken, client.calls[0]['schema'])


def test_ambiguous_names_remain_unresolved_in_the_new_protocol():
    prompts, cast, owners = fixture()
    cast['n001']['kind'] = 'ambiguous'
    result = compile_conditional_name_entities(Client([cast, owners]), prompts)
    assert result['unresolved'] and not result['ready_for_development']
    with pytest.raises(ValueError, match='Unresolved'):
        require_typed_ready(result)


def test_structure_does_not_assign_pronoun_like_names_by_program():
    prompts = ['She is the proper name of a robot wearing a red coat.']
    cast = {'n000': {'kind': 'character', 'presence': {'0': 'required'}, 'reason': 'Constructed unusual name'}}
    owners = {row['id']: {'owners': ['She'], 'status': 'appearance', 'reason': 'Constructed fixture'}
        for row in physical_candidates(prompts)}
    result = compile_conditional_name_entities(Client([cast, owners]), prompts)
    assert result['entities'][0]['name'] == 'She'


def test_empty_inventory_never_certifies_no_characters():
    client = Client([])
    result = compile_conditional_name_entities(client, ['zara wears a red coat.'])
    assert not client.calls and not result['ready_for_development']
    with pytest.raises(ValueError, match='Unresolved'): require_typed_ready(result)
