"""Synthetic evidence and state contracts; no video-quality assertions."""
import copy
import dataclasses
import hashlib
import json

import pytest

from utils.agent_constraints import ConstraintAssessment, ConstraintEvidence, entity_judge_metrics_to_assessments
from utils.agent_evidence_routing import attribute_evidence, mixed_target_revision
from utils.agent_bounded_feedback import BoundedFeedbackPolicy
from utils.agent_progressive_repair import improvement_decision
from tests.test_agent_progressive_repair import verdict, transaction
from tests.test_agent_bounded_feedback import choices, menu, executed


def with_evidence(value, labels, *, part='hair', recheck='fail', duplicate=False):
    result = copy.deepcopy(value)
    rows = result.metrics['constraint_assessments']
    row = next(r for r in rows if r['constraint']['id'] == part)
    row['constraint']['expected'] = 'red'
    row['constraint']['parameters']['basis'] = 'fixed_script_quote'
    frames = [{'path': f'frame{i}.png', 'sha256': hashlib.sha256(str(0 if duplicate else i).encode()).hexdigest()}
              for i in range(len(labels))]
    row['evidence'] = [ConstraintEvidence('attribute_frame_review_v1', observed='synthetic',
        details='Free text must not control routing: low_confidence',
        metadata={'status': status, 'frame_path': frame['path'], 'frame_sha256': frame['sha256']}).to_dict()
        for status, frame in zip(labels, frames)]
    if recheck is not None:
        row['evidence'].append(ConstraintEvidence('attribute_joint_recheck_v1', observed='synthetic',
            metadata={'status': recheck, 'frames': frames, 'independent_model': False}).to_dict())
    return result


def pair(**kwargs):
    return (with_evidence(verdict(), ['fail'] * 3, recheck=None),
            with_evidence(verdict(hair='unknown'), ['fail', 'pass', 'pass'], **kwargs))


def test_temporal_label_mix_may_revise_but_can_never_commit():
    parent, child = pair()
    report = mixed_target_revision(parent, child, ['hair'])
    assert report['allow_revision'] and not report['acceptance_changed']
    assert report['attributes']['hair']['candidate']['category'] == 'mixed_frame_labels'
    assert not improvement_decision(parent, child, parent, ['hair'])['allow']


@pytest.mark.parametrize('labels,recheck,duplicate,category', [
    (['fail','pass','pass'], None, False, 'mixed_frame_labels'),
    (['fail','pass','pass'], 'unknown', False, 'mixed_frame_labels'),
    (['unknown','unknown','unknown'], 'unknown', False, 'insufficient_visibility_or_support'),
    (['fail','pass','unknown'], 'fail', False, 'insufficient_visibility_or_support'),
    (['fail','pass','pass'], 'fail', True, 'same_frame_judgment_disagreement'),
    (['fail'], 'fail', False, 'insufficient_visibility_or_support'),
])
def test_invisibility_unreviewed_mix_and_same_frame_disagreement_stop(labels, recheck, duplicate, category):
    parent, _ = pair()
    child = with_evidence(verdict(hair='unknown'), labels, recheck=recheck, duplicate=duplicate)
    report = mixed_target_revision(parent, child, ['hair'])
    assert not report['allow_revision']
    assert report['attributes']['hair']['candidate']['category'] == category


@pytest.mark.parametrize('change', ['identity_unknown', 'non_target_unknown', 'parent_unknown',
                                  'parent_single_fail', 'changed_requirement', 'recheck_other_frames'])
def test_revision_requires_parent_support_unchanged_predicates_and_no_other_unknown(change):
    parent, child = pair()
    if change in ('identity_unknown','non_target_unknown'):
        key = 'identity' if change == 'identity_unknown' else 'garment'
        next(r for r in child.metrics['constraint_assessments'] if r['constraint']['id'] == key)['status'] = 'unknown'
    elif change == 'parent_unknown':
        parent.metrics['constraint_assessments'][0]['status'] = 'unknown'
    elif change == 'parent_single_fail':
        parent = with_evidence(verdict(), ['fail'], recheck=None)
    elif change == 'changed_requirement':
        child.metrics['constraint_assessments'][0]['constraint']['expected'] = 'blue'
    else:
        child.metrics['constraint_assessments'][0]['evidence'][-1]['metadata']['frames'][0]['sha256'] = 'f' * 64
    assert not mixed_target_revision(parent, child, ['hair'])['allow_revision']


def test_converter_retains_joint_status_and_explicit_summary_members():
    frames = [{'status': status, 'observed': 'synthetic', 'reason': 'synthetic',
               'path': f'{i}.png', 'sha256': hashlib.sha256(str(i).encode()).hexdigest()}
              for i, status in enumerate(['fail','pass','pass'])]
    attrs = [{'id': 'red', 'part': 'hair', 'aspect': 'color', 'expected': 'red',
              'status': 'unknown', 'reason': 'recheck_contradicts_frame_evidence',
              'evidence': frames, 'initial_status': 'unknown', 'initial_reason': 'mixed_temporal_evidence',
              'recheck': {'status': 'fail', 'observed': 'synthetic', 'reason': 'synthetic',
                          'frames': [{'path': r['path'], 'sha256': r['sha256']} for r in frames]}}]
    metrics = {'entities': {'Iris': {'expected': True, 'located_chunks': [0], 'matches': False,
        'comparison_basis': 'script_description', 'attribute_assessments': attrs}}}
    rows = entity_judge_metrics_to_assessments(metrics, shot_index=0)
    attribute = next(r for r in rows if r.constraint.parameters.get('part') == 'hair')
    summary = next(r for r in rows if r.constraint.parameters.get('comparison_basis') == 'script_description')
    assert attribute.evidence[-1].metadata['status'] == 'fail'
    assert summary.evidence[0].metadata['attribute_constraint_ids'] == [attribute.id]
    assert attribute.status == summary.status == 'unknown'


def test_derived_summary_cannot_hide_an_extra_unknown():
    parent, child = pair()
    for value in (parent, child):
        hair = value.metrics['constraint_assessments'][0]
        summary = copy.deepcopy(hair)
        summary['constraint'].update(id='summary', expected=True,
            parameters={'comparison_basis': 'script_description'})
        summary['evidence'] = [ConstraintEvidence('entity_judge', metadata={
            'aggregation': 'authored_attribute_summary_v1', 'attribute_constraint_ids': ['hair']}).to_dict()]
        value.metrics['constraint_assessments'].append(summary)
    assert mixed_target_revision(parent, child, ['hair'])['derived_summaries'] == ['summary']
    child.metrics['constraint_assessments'][-1]['evidence'][0]['metadata']['attribute_constraint_ids'].append('unbound')
    assert not mixed_target_revision(parent, child, ['hair'])['allow_revision']


def test_mixed_route_offers_one_distinct_expression_without_noise_or_region_retry(choices):
    policy = BoundedFeedbackPolicy(allow_mixed_evidence_revision=True)
    first = menu(policy, choices)[0][0]
    offered, reason = menu(policy, choices, history=[executed(first)], seed=4244, failure='target_evidence_mixed')
    assert reason == 'menu_ready' and len(offered) == 1
    assert offered[0]['purpose'] == 'target_expression' and offered[0]['kind'] == 'revise'
    assert offered[0]['actual']['mask_tensor_sha256'] == first['actual']['mask_tensor_sha256']
    assert offered[0]['actual']['prompt'].startswith(first['actual']['prompt'])
    assert offered[0]['actual']['entry_sigma'] == 1
    history = [executed(first), {**executed(offered[0]), 'failure': 'target_evidence_mixed'}]
    again, reason = menu(policy, choices, history=history, seed=4245, failure='target_evidence_mixed')
    assert not again and reason == 'mixed_evidence_revision_exhausted'
    disabled, reason = menu(BoundedFeedbackPolicy(), choices, history=[executed(first)], seed=4244,
                            failure='target_evidence_mixed')
    assert not disabled and reason == 'mixed_evidence_revision_disabled'


def test_mixed_route_executes_once_from_preserved_parent_and_restores_state(transaction, monkeypatch):
    from PIL import Image
    import utils.agent_repair_regions as regions
    from utils.agent_runtime import Verdict
    t = transaction
    t.verifier.outcomes[0], t.verifier.outcomes[1] = pair()
    t.verifier.outcomes[2] = t.verifier.outcomes[1]
    # This fixture's original toy prompt contains no literal authored value.
    judge = t.verifier.judge
    def authored_judge(*args, **kwargs):
        result = judge(*args, **kwargs)
        for option in result.metrics['repair_options']:
            option['action']['params']['prompt'] = 'Iris walks with red hair.'
        return result
    monkeypatch.setattr(t.verifier, 'judge', authored_judge)
    monkeypatch.setattr(regions, 'read_video_frames', lambda *args: [Image.new('RGB', (16,16))])
    class Client:
        def __init__(self): self.calls = []
        def call(self, **kwargs):
            self.calls.append(kwargs)
            packet = json.loads(kwargs['prompt'])
            assert packet['feedback']['failure'] == 'target_evidence_mixed'
            assert packet['feedback']['evidence_routing']['allow_revision']
            assert len(packet['options']) == 1 and packet['options'][0]['kind'] == 'revise'
            return {'parsed': {'option_id': packet['options'][0]['id'], 'reason': 'Synthetic bounded revision'}}
    client = Client()
    with t.session(feedback_policy=BoundedFeedbackPolicy(client, allow_mixed_evidence_revision=True)) as session:
        result = session.run_shot(0)
        records = session.ledger.records()
        assert session.memory.partition_counts()['persistent'] == 0
    assert not result.accepted and len(t.renders) == 2 and len(client.calls) == 1
    assert t.backend.active_branch_id == 'branch-0000' and t.backend.fully_verified_branch_id is None
    assert t.verifier.state == [] and not t.protection.calls
    assert len({r['inference']['replay_parent']['latent_path'] for r in t.renders}) == 1
    assert records[-1]['reason'] == 'mixed_evidence_revision_exhausted'
