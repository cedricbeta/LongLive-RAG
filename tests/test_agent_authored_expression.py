"""Source binding, action budget and unchanged-default contracts; not quality tests."""
import copy

import pytest

from utils.agent_authored_expression import authored_garment_prompt
from utils.agent_bounded_feedback import BoundedFeedbackPolicy
from utils.agent_runtime import Action
from tests.test_agent_bounded_feedback import choices, executed, menu


def fixture(name='Mira', color='olive', kind='coat'):
    phrase = f'{color} {kind}'
    prompts = [f'{name} wears an {phrase} while carrying a book.']
    rows = [{'part': 'garment', 'aspect': aspect, 'expected': value,
             'source': {'shot': 0, 'quote': phrase}, 'last_active_shot': 0}
            for aspect, value in [('color', color), ('type', kind)]]
    compiled = {'prompts': prompts, 'entities': [{'name': name, 'appearance': rows}]}
    targets = [{'id': aspect, 'subjects': [name], 'expected': value,
                'parameters': {'part': 'garment', 'aspect': aspect, 'basis': 'fixed_script_quote'}}
               for aspect, value in [('color', color), ('type', kind)]]
    return prompts[0], targets, compiled


def test_authored_phrase_preserves_entire_action_and_other_requirements():
    base, requirements, compiled = fixture()
    base += ' Mira has curly hair. Keep the book on the table.'
    assert authored_garment_prompt(base, requirements, compiled, 0) == base + ' Mira wears an olive coat.'
    # Sorting constraint IDs must not change the grammatical phrase.
    assert authored_garment_prompt(base, list(reversed(requirements)), compiled, 0) == base + ' Mira wears an olive coat.'


@pytest.mark.parametrize('failure', ['wrong_actor', 'inactive', 'conflicting_value', 'unquoted', 'separate_words', 'changed_base', 'single_target'])
def test_unsupported_binding_does_not_offer_an_expression(failure):
    base, requirements, compiled = fixture()
    if failure == 'wrong_actor': requirements[1]['subjects'] = ['Other']
    elif failure == 'inactive':
        compiled['prompts'].append(base)
        for row in compiled['entities'][0]['appearance']:
            row['source']['shot'] = row['last_active_shot'] = 1
    elif failure == 'conflicting_value': requirements[0]['expected'] = 'red'
    elif failure == 'unquoted': compiled['entities'][0]['appearance'][0]['source']['quote'] = 'invented olive coat'
    elif failure == 'separate_words':
        base = compiled['prompts'][0] = 'Mira wears a coat by an olive tree.'
        for row in compiled['entities'][0]['appearance']: row['source']['quote'] = base
    elif failure == 'changed_base': base = 'Changed scene: ' + base
    elif failure == 'single_target': requirements = requirements[:1]
    assert authored_garment_prompt(base, requirements, compiled, 0) is None


def test_ambiguous_active_sources_and_explicit_other_actor_are_refused():
    base, requirements, compiled = fixture()
    duplicate = copy.deepcopy(compiled['entities'][0]['appearance'][0])
    compiled['entities'][0]['appearance'].append(duplicate)
    assert authored_garment_prompt(base, requirements, compiled, 0) is None
    compiled['entities'][0]['appearance'].pop()
    base = compiled['prompts'][0] = 'Mira and Sol stand near an olive coat.'
    compiled['entities'].append({'name': 'Sol', 'appearance': []})
    for row in compiled['entities'][0]['appearance']: row['source']['quote'] = base
    assert authored_garment_prompt(base, requirements, compiled, 0) is None


def test_no_duplicate_sentence_and_known_singular_dress_is_supported():
    base, requirements, compiled = fixture(kind='dress')
    changed = authored_garment_prompt(base, requirements, compiled, 0)
    assert changed == base + ' Mira wears an olive dress.'
    assert authored_garment_prompt(changed, requirements, compiled, 0) is None
    base, requirements, compiled = fixture(kind='pants')
    assert authored_garment_prompt(base, requirements, compiled, 0) is None


def test_opt_in_preserves_first_action_and_actual_budget_deduplication(choices):
    base, requirements, compiled = fixture()
    action = Action('edit_shot', {**choices['first'].params, 'prompt': base})
    data = {**choices, 'first': action, 'requirements': requirements}
    policy = BoundedFeedbackPolicy(expression_protocol='authored_garment_v1', compiled_script=compiled,
                                   allow_mixed_evidence_revision=True)
    first = menu(policy, data)[0][0]
    assert first['action'] == action
    offered, _ = menu(policy, data, history=[executed(first)], seed=4244, failure='target_evidence_mixed')
    assert len(offered) == 1
    revised = offered[0]
    assert revised['purpose'] == 'target_expression' and revised['kind'] == 'revise'
    assert revised['action'].params['prompt'] == base + ' Mira wears an olive coat.'
    assert revised['actual']['mask_tensor_sha256'] == first['actual']['mask_tensor_sha256']
    assert revised['action'].params['entry_sigma'] == 1.0
    # A repeat of the actual expression with the same seed is not a new action.
    offered, _ = menu(policy, data, history=[executed(first), executed(revised)], seed=4244, failure='target_not_improved')
    assert not any(x['actual']['action_fingerprint'] == revised['actual']['action_fingerprint'] for x in offered)
    old = BoundedFeedbackPolicy()
    old_offered, _ = menu(old, data, history=[executed(first)], seed=4244, failure='target_not_improved')
    old_focus = next(x for x in old_offered if x['purpose'] == 'target_expression')
    assert old_focus['action'].params['prompt'] == base + " Mira's garment is olive. Mira's garment is coat."


def test_bad_protocol_or_missing_script_fails_before_planning():
    with pytest.raises(ValueError, match='compiled script'):
        BoundedFeedbackPolicy(expression_protocol='authored_garment_v1')
    with pytest.raises(ValueError, match='Unknown'):
        BoundedFeedbackPolicy(expression_protocol='invented')


def test_requirement_from_another_shot_is_not_current_evidence():
    base, requirements, compiled = fixture()
    requirements[0]['shot_indices'] = [1]
    assert authored_garment_prompt(base, requirements, compiled, 0) is None


def test_policy_keeps_an_immutable_copy_of_compiled_requirements(choices):
    base, requirements, compiled = fixture()
    action = Action('edit_shot', {**choices['first'].params, 'prompt': base})
    data = {**choices, 'first': action, 'requirements': requirements}
    policy = BoundedFeedbackPolicy(expression_protocol='authored_garment_v1', compiled_script=compiled)
    compiled['prompts'][0] = 'Externally changed script'
    first = menu(policy, data)[0][0]
    offered, _ = menu(policy, data, history=[executed(first)], seed=4244, failure='target_not_improved')
    assert any(row['action'].params['prompt'] == base + ' Mira wears an olive coat.' for row in offered)
