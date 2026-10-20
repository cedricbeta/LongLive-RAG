import json
from types import SimpleNamespace

import pytest

from utils.agent_attribute_diagnosis import AttributeEvidenceComparer, aggregate, freeze_requirements, validate_required_aspects
from utils.agent_constraints import entity_judge_metrics_to_assessments
from utils.agent_repair_planner import classify_failure
from utils.agent_runtime import Verdict


def requirements():
    return freeze_requirements('curly red hair and a green coat', [
        {'part': 'hair', 'aspect': 'color', 'expected': 'red hair'},
        {'part': 'garment', 'aspect': 'color', 'expected': 'green coat'}])


def frame(reqs, name, statuses):
    return {'path': name, 'sha256': name, 'assessments': {
        r['id']: {'status': statuses[r['part']], 'observed': 'actual visible appearance', 'reason': 'visible evidence'}
        for r in reqs}}


def test_script_definitions_do_not_depend_on_branch_and_cannot_invent_attributes():
    assert requirements() == requirements()
    with pytest.raises(ValueError, match='script span'):
        freeze_requirements('red hair', [{'part': 'hair', 'aspect': 'color', 'expected': 'blue hair'}])
    with pytest.raises(ValueError, match='Empty'):
        freeze_requirements('red hair', [])


def test_known_garment_failure_survives_unknown_hair_and_low_confidence_label():
    reqs = requirements()
    rows = aggregate(reqs, [frame(reqs, name, {'hair': 'unknown', 'garment': 'fail'}) for name in ['f0', 'f1']])
    metrics = {'entities': {'Iris': {'expected': True, 'located_chunks': [0], 'matches': False,
               'attribute_assessments': rows}}, 'unassigned_subjects': []}
    assessments = [r.to_dict() for r in entity_judge_metrics_to_assessments(metrics, shot_index=0)]
    verdict = Verdict(False, 'reject_identity_reference_low_confidence', metrics={'constraint_assessments': assessments})
    assert classify_failure(verdict) == 'current_defect'
    garment = next(a for a in assessments if a['constraint']['parameters'].get('part') == 'garment')
    assert garment['status'] == 'violated'
    assert len(garment['evidence']) == 2


def test_insufficient_mixed_temporal_evidence_and_duplicate_images_stay_unknown():
    reqs = requirements()
    rows = aggregate(reqs, [frame(reqs, 'f0', {'hair': 'pass', 'garment': 'fail'}),
                            frame(reqs, 'f1', {'hair': 'fail', 'garment': 'pass'})])
    assert all(r['status'] == 'unknown' and r['reason'] == 'mixed_temporal_evidence' for r in rows)
    repeated = aggregate(reqs, [frame(reqs, 'same', {'hair': 'fail', 'garment': 'fail'})] * 3)
    assert all(r['status'] == 'unknown' for r in repeated)


@pytest.mark.parametrize('passing_frames', [1, 3])
def test_two_distinct_temporal_violations_cannot_be_outvoted_by_passing_frames(passing_frames):
    reqs = requirements()
    observations = [('early', 'fail'), ('middle', 'fail')] + [(f'pass_{i}', 'pass') for i in range(passing_frames)]
    rows = aggregate(reqs, [frame(reqs, name, {'hair': state, 'garment': state}) for name, state in observations])
    assert all(r['status'] == 'fail' and r['reason'] == 'supported_temporal_conflict' for r in rows)
    assert all(r['distinct_frame_counts'] == {'pass': passing_frames, 'fail': 2, 'unknown': 0} for r in rows)


def test_conflicting_judgments_of_identical_bytes_cannot_use_last_writer_as_vote():
    reqs = requirements()
    one = frame(reqs, 'same', {'hair': 'pass', 'garment': 'pass'})
    two = frame(reqs, 'same', {'hair': 'fail', 'garment': 'fail'})
    other = frame(reqs, 'other', {'hair': 'fail', 'garment': 'fail'})
    for source in [[one, two, other], [two, one, other]]:
        rows = aggregate(reqs, source)
        assert all(r['status'] == 'unknown' and r['reason'] == 'same_frame_judgment_disagreement' for r in rows)
        assert all(r['same_frame_conflicting_sha256'] == ['same'] for r in rows)


def test_absent_entity_preserves_authored_attribute_definitions():
    rows = [{**r, 'status': 'unknown', 'reason': 'not_located', 'evidence': []} for r in requirements()]
    items = entity_judge_metrics_to_assessments({'entities': {'Iris': {
        'expected': True, 'located_chunks': [], 'attribute_assessments': rows}}}, shot_index=3)
    attrs = [a for a in items if ':attribute:' in a.id]
    assert len(attrs) == 2 and all(a.status == 'unknown' for a in attrs)
    assert {a.constraint.expected for a in attrs} == {'green coat', 'red hair'}


class Client:
    def __init__(self):
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs['call_kind'] == 'attribute_requirements_v2':
            return {'parsed': {'requirements': [{'part': 'hair', 'aspect': 'color', 'expected': 'red hair'}]}}
        return {'parsed': {key: {'status': 'unknown', 'observed': 'hair is obscured', 'reason': 'occlusion'}
                           for key in kwargs['schema']['required']}}


def test_unknown_gets_one_recheck_cannot_pass_or_become_identity_reference(tmp_path):
    crops = []
    for i in range(3):
        path = tmp_path / f'{i}.png'
        path.write_bytes(f'distinct stub evidence {i}'.encode())
        crops.append(path)
    client = Client()
    comparer = AttributeEvidenceComparer(client)
    entity = SimpleNamespace(name='Iris', description='red hair')
    result = comparer.compare(entity, None, crops, shot_index=0, inputs=())
    assert not result['matches_reference'] and not result['reference_registration_eligible']
    assert len(result['attribute_rechecks']) == 1
    assert sum(c['call_kind'] == 'attribute_joint_recheck_v1' for c in client.calls) == 1
    recheck = client.calls[-1]
    assert recheck['image_paths'] == [str(p) for p in crops]
    assert 'history' not in json.loads(recheck['prompt'])
    comparer.requirements(entity, shot_index=1, inputs=())
    assert sum(c['call_kind'] == 'attribute_requirements_v2' for c in client.calls) == 1
    assert result['attribute_assessments'][0]['initial_status'] == 'unknown'


def test_compound_hair_or_garment_description_cannot_hide_missing_aspects():
    with pytest.raises(ValueError, match='Incomplete attribute extraction'):
        validate_required_aspects('curly red hair and a green coat', requirements())
    rows = requirements() + [
        {'part': 'hair', 'aspect': 'style', 'expected': 'curly'},
        {'part': 'garment', 'aspect': 'type', 'expected': 'coat'}]
    validate_required_aspects('curly red hair and a green coat', rows)


@pytest.mark.parametrize('per_frame,joint', [(['fail', 'pass', 'unknown'], 'pass'),
                                           (['pass', 'unknown', 'unknown'], 'fail')])
def test_joint_recheck_cannot_erase_opposite_explicit_frame_evidence(tmp_path, per_frame, joint):
    class ContradictoryClient(Client):
        def call(self, **kwargs):
            if kwargs['call_kind'] == 'attribute_requirements_v2':
                return super().call(**kwargs)
            self.calls.append(kwargs)
            if kwargs['call_kind'] == 'attribute_joint_recheck_v1':
                status = joint
            else:
                count = sum(c['call_kind'] == 'attribute_frame_review_v1' for c in self.calls)
                status = per_frame[count - 1]
            return {'parsed': {key: {'status': status, 'observed': 'visible evidence', 'reason': 'recorded observation'}
                               for key in kwargs['schema']['required']}}
    crops = []
    for i in range(3):
        p = tmp_path / f'frame{i}.png'
        p.write_bytes(str(i).encode())
        crops.append(p)
    result = AttributeEvidenceComparer(ContradictoryClient()).compare(
        SimpleNamespace(name='Iris', description='red hair'), None, crops, shot_index=0, inputs=())
    row = result['attribute_assessments'][0]
    assert row['status'] == 'unknown' and row['recheck_conflicts_with_prior_evidence']
    assert row['recheck']['status'] == joint
    assert not result['matches_reference'] and not result['reference_registration_eligible']
