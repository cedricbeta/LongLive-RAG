import copy

from utils.agent_identity_comparison import FEATURES, decide


def evidence():
    return {'face_visibility_A':'clear','face_visibility_B':'clear',
        'actor_correspondence':'one_unambiguous_subject_per_slot',
        **{k:{'observation_A':'synthetic feature','observation_B':'synthetic feature','comparison':'compatible'} for k in FEATURES}}


def test_unobservable_identity_cannot_pass_even_with_compatible_feature_claims():
    row=evidence();row['face_visibility_B']='unobservable'
    assert decide(row)['status']=='unknown'
    row=evidence();row['actor_correspondence']='ambiguous'
    assert decide(row)['status']=='unknown'


def test_one_visible_facial_contradiction_cannot_be_outvoted():
    row=evidence();assert decide(row)['status']=='pass'
    row['nose']['comparison']='contradictory'
    assert decide(row)['status']=='fail'
