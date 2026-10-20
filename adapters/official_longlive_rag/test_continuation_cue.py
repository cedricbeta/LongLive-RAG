import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuation_cue import compile_cue


def inputs():
    return {0: ['Kai, an older man with silver hair and a brown trench coat, stands by a desk.'],
            81: ['At the desk, Kai and Ava untangle a cable.'],
            121: ['At the desk, Kai stands alone, reflecting on Ava who has left.'],
            161: ['At the desk, Kai stands and replaces a softbox.'],
            201: ['At the desk, Kai stands and checks his work.']}, {
                'schema_version': 1, 'entity': 'Kai', 'hair_color': 'silver',
                'evidence_start_latent': 0, 'apply_at': [121, 161, 201]}


def test_disabled_is_noop():
    schedule, _ = inputs()
    out, record = compile_cue(schedule, None, earliest_latent=121)
    assert out is schedule and record == {'enabled': False, 'changes': []}


def test_only_append_literal_to_later_present_actor_actions():
    schedule, cue = inputs()
    before, cue_before = copy.deepcopy(schedule), copy.deepcopy(cue)
    out, record = compile_cue(schedule, cue, earliest_latent=121)
    assert schedule == before and cue == cue_before
    assert out[0] == before[0] and out[81] == before[81]
    for boundary in [121, 161, 201]:
        assert out[boundary] == [before[boundary][0] + ' Kai has silver hair.']
    assert record['source_quote'] == 'Kai, an older man with silver hair'
    assert record['authored_schedule_sha256'] != record['effective_schedule_sha256']
    assert record['new_prompt_boundaries'] == 0
    out[121][0] = 'mutation'
    record['cue']['apply_at'].clear()
    assert schedule == before and cue == cue_before


@pytest.mark.parametrize('patch', [
    {'hair_color': 'red'}, {'hair_color': []}, {'entity': 'Ava'},
    {'entity': 'Kai\nAva'}, {'apply_at': [81]}, {'apply_at': [121, 121]},
    {'apply_at': [201, 121]}, {'apply_at': [122]}, {'apply_at': [True]},
    {'evidence_start_latent': 121}, {'schema_version': True}, {'extra_prompt': 'anything'},
])
def test_reject_invalid_or_unbound_request(patch):
    schedule, cue = inputs()
    cue.update(patch)
    with pytest.raises(ValueError):
        compile_cue(schedule, cue, earliest_latent=121)


@pytest.mark.parametrize('prompt', [
    'At the desk, Ava remembers Kai.',
    'After Kai leaves, Ava stands at the desk.',
    'At the desk, Kai is absent.',
    'At the desk, Kai stands, then Kai leaves.',
    'At the desk, Kai stands without Kai.',
])
def test_do_not_reintroduce_an_absent_or_uncertain_actor(prompt):
    schedule, cue = inputs()
    schedule[121] = [prompt]
    with pytest.raises(ValueError):
        compile_cue(schedule, cue, earliest_latent=121)


def test_do_not_assign_another_actors_color():
    schedule, cue = inputs()
    schedule[0] = ['Kai watches Ava, an older woman with silver hair.']
    with pytest.raises(ValueError):
        compile_cue(schedule, cue, earliest_latent=121)


def test_reject_duplicate_existing_cue():
    schedule, cue = inputs()
    schedule[121][0] += ' Kai has silver hair.'
    with pytest.raises(ValueError):
        compile_cue(schedule, cue, earliest_latent=121)
