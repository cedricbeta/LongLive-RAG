"""Program contracts and synthetic transactions; no video quality assertions."""
import json
from pathlib import Path

import pytest
import torch

from utils.agent_bounded_feedback import BoundedFeedbackPolicy, actual_action
from utils.agent_edit import save_mask_payload
from utils.agent_runtime import Action, file_sha256
from tests.test_agent_progressive_repair import transaction, verdict


@pytest.fixture
def choices(tmp_path):
    video, latents = tmp_path / 'parent.mp4', tmp_path / 'parent.pt'
    video.write_bytes(b'actual parent bytes')
    torch.save(torch.zeros(1, 2, 1, 3, 3), latents)
    masks = {'base': torch.zeros(2, 3, 3), 'narrow': torch.zeros(2, 3, 3), 'broad': torch.ones(2, 3, 3)}
    masks['base'][:, :2, :2] = 1
    masks['narrow'][:, 0, 0] = 1
    options = []
    for name, tensor in masks.items():
        path = save_mask_payload({0: tensor}, tmp_path / (name + '.pt'), phrase='hair',
                                 video=video, frames_per_chunk=2, grid=(3, 3))
        options.append({'id': 'attribute:Iris:hair:' + name, 'mask_sha256': file_sha256(path),
            'action': {'tool': 'edit_shot', 'params': {'shot': 0, 'prompt': 'Iris arranges flowers with red hair.',
                'entry_sigma': 1.0, 'mask_path': str(path)}}})
    first = Action(**options[0]['action'])
    requirements = [{'id': 'red', 'subjects': ['Iris'], 'expected': 'red',
                     'parameters': {'part': 'hair', 'aspect': 'color', 'basis': 'fixed_script_quote'}}]
    return {'first': first, 'options': options, 'requirements': requirements,
            'parent_video': video, 'parent_latents': latents, 'seed': 4243, 'history': [], 'failure': None}


def menu(policy, data, **updates):
    return policy.menu(**{**data, **updates})


def executed(row):
    return {'actual': row['actual'], 'targets': row['targets']}


def test_first_recipe_is_preserved_and_model_is_not_called(choices):
    policy = BoundedFeedbackPolicy()
    offered, reason = menu(policy, choices)
    assert reason == 'menu_ready' and len(offered) == 1
    selected, report = policy.choose(offered, feedback={})
    assert selected['action'] == choices['first']
    assert selected['kind'] == 'first_fixed_recipe' and report['calls'] == []


def test_actual_content_distinguishes_seed_only_retry_from_prompt_change(choices):
    policy = BoundedFeedbackPolicy()
    first = menu(policy, choices)[0][0]
    offered, _ = menu(policy, choices, history=[executed(first)], seed=4244, failure='target_not_improved')
    assert {row['kind'] for row in offered} == {'resample', 'revise'}
    retry = next(row for row in offered if row['kind'] == 'resample')
    revised = next(row for row in offered if row['kind'] == 'revise')
    assert retry['actual']['family_fingerprint'] == first['actual']['family_fingerprint']
    assert retry['actual']['action_fingerprint'] != first['actual']['action_fingerprint']
    assert revised['action'].params['prompt'].startswith(choices['first'].params['prompt'])
    assert revised['action'].params['mask_path'] == choices['first'].params['mask_path']


def test_changed_filenames_cannot_hide_duplicate_geometry_or_seed(choices, tmp_path):
    policy = BoundedFeedbackPolicy()
    first = menu(policy, choices)[0][0]
    payload = torch.load(choices['first'].params['mask_path'], weights_only=True)
    path = tmp_path / 'renamed.pt'
    payload['phrase'] = 'different explanation only'
    torch.save(payload, path)
    action = Action('edit_shot', {**choices['first'].params, 'mask_path': str(path)})
    offered, _ = menu(policy, choices, first=action, history=[executed(first)], failure='target_not_improved')
    assert all(row['actual']['family_fingerprint'] != first['actual']['family_fingerprint'] for row in offered)


def test_exhausted_family_is_removed_but_other_legal_action_survives(choices):
    policy = BoundedFeedbackPolicy()
    first = menu(policy, choices)[0][0]
    second = menu(policy, choices, history=[executed(first)], seed=4244, failure='target_not_improved')[0][0]
    offered, _ = menu(policy, choices, history=[executed(first), executed(second)], seed=4245, failure='target_not_improved')
    assert {row['purpose'] for row in offered} == {'target_expression', 'region_broader'}
    assert all(row['actual']['family_fingerprint'] != first['actual']['family_fingerprint'] for row in offered)


@pytest.mark.parametrize('failure,expected', [('localization_miss', 'broader'),
    ('protection_regression', 'narrower'), ('future_regression', 'narrower')])
def test_failure_selects_program_verified_region_relation(choices, failure, expected):
    policy = BoundedFeedbackPolicy()
    first = menu(policy, choices)[0][0]
    offered, _ = menu(policy, choices, history=[executed(first)], seed=4244, failure=failure)
    assert len(offered) == 1 and offered[0]['region_relation'] == expected
    assert offered[0]['action'].params['prompt'] == choices['first'].params['prompt']


@pytest.mark.parametrize('failure', ['insufficient_evidence', 'insufficient_future_evidence', 'evaluation_error'])
def test_missing_evidence_never_becomes_a_new_noise_draw(choices, failure):
    policy = BoundedFeedbackPolicy()
    first = menu(policy, choices)[0][0]
    offered, reason = menu(policy, choices, history=[executed(first)], seed=4244, failure=failure)
    assert not offered and reason == failure


def test_new_parent_requires_new_region_provenance(choices):
    choices['parent_video'].write_bytes(b'changed retained parent')
    with pytest.raises(ValueError, match='different parent'):
        menu(BoundedFeedbackPolicy(), choices)


def test_same_geometry_with_changed_time_coordinates_is_not_same_action(choices):
    action = choices['first']
    before = actual_action(action, **{k: choices[k] for k in ['parent_video', 'parent_latents', 'seed']})
    payload = torch.load(action.params['mask_path'], weights_only=True)
    payload['masks'] = {1: payload['masks'][0]}
    torch.save(payload, action.params['mask_path'])
    after = actual_action(action, **{k: choices[k] for k in ['parent_video', 'parent_latents', 'seed']})
    assert before['action_fingerprint'] != after['action_fingerprint']


def test_model_only_selects_offered_actions_and_invalid_answer_has_bounded_retry(choices):
    class Client:
        def __init__(self): self.calls = []
        def call(self, **kwargs):
            self.calls.append(kwargs)
            packet = json.loads(kwargs['prompt'])
            row = next(row for row in packet['options'] if row['purpose'] == 'target_expression')
            return {'parsed': {'option_id': 'invented' if len(self.calls) == 1 else row['id'], 'reason': 'Synthetic selection'}}
    client = Client()
    policy = BoundedFeedbackPolicy(client)
    first = menu(policy, choices)[0][0]
    offered, _ = menu(policy, choices, history=[executed(first)], seed=4244, failure='target_not_improved')
    chosen, report = policy.choose(offered, feedback={'failure': 'target_not_improved'})
    assert chosen['purpose'] == 'target_expression' and len(report['rejected_plans']) == 1
    assert len(client.calls) == 2
    assert set(client.calls[0]['schema']['properties']) == {'option_id', 'reason'}


def test_native_masks_use_same_policy_with_strict_parent_latent_binding(choices, tmp_path):
    payload = {'mask': torch.ones(2, 3, 3, dtype=torch.bool), 'grid': [3, 3], 'latent_range': [4, 6],
        'parent_video_sha256': file_sha256(choices['parent_video']),
        'parent_latents_sha256': file_sha256(choices['parent_latents']), 'parent_timing_sha256': 'fixedtiming'}
    path = tmp_path / 'native.pt'; torch.save(payload, path)
    first = Action('edit_shot', {**choices['first'].params, 'mask_path': str(path)})
    offered, _ = menu(BoundedFeedbackPolicy(), choices, first=first)
    assert offered[0]['actual']['coordinates']['format'] == 'native'
    choices['parent_latents'].write_bytes(b'changed latent parent')
    with pytest.raises(ValueError, match='different parent latents'):
        menu(BoundedFeedbackPolicy(), choices, first=first)


def test_opt_in_feedback_preserves_retained_partial_branch_without_trusting_identity(transaction):
    t = transaction
    with t.session(feedback_policy=BoundedFeedbackPolicy()) as session:
        result = session.run_shot(0)
        records = session.ledger.records()
    assert not result.accepted and t.backend.active_branch_id == 'branch-0001'
    assert t.backend.fully_verified_branch_id is None
    assert len(t.renders) == 2
    assert [r['kind'] for r in records if r['event'] == 'bounded_action_reserved'] == ['first_fixed_recipe'] * 2
    assert any(r['event'] == 'bounded_feedback_stop' and r['reason'] == 'legal_actions_exhausted' for r in records)


def test_opt_in_unknown_protection_stops_before_planner_or_resampling(transaction):
    t = transaction
    t.protection.status = 'unknown'
    with t.session(feedback_policy=BoundedFeedbackPolicy()) as session:
        session.run_shot(0)
        records = session.ledger.records()
    assert len(t.renders) == 1 and t.backend.active_branch_id == 'branch-0000'
    assert records[-1]['reason'] == 'insufficient_evidence'


def test_real_transaction_path_uses_one_noise_retry_then_excludes_it(transaction, monkeypatch):
    from PIL import Image
    import utils.agent_repair_regions as regions
    t = transaction
    t.verifier.outcomes[1] = verdict()
    t.verifier.outcomes[2] = verdict()
    monkeypatch.setattr(regions, 'read_video_frames', lambda *args: [Image.new('RGB', (16, 16))])
    class Client:
        def __init__(self): self.calls = []
        def call(self, **kwargs):
            self.calls.append(kwargs)
            packet = json.loads(kwargs['prompt'])
            row = next(row for row in packet['options'] if row['kind'] == 'resample')
            return {'parsed': {'option_id': row['id'], 'reason': 'Synthetic one-noise retry'}}
    client = Client()
    with t.session(feedback_policy=BoundedFeedbackPolicy(client)) as session:
        result = session.run_shot(0)
        records = session.ledger.records()
    assert not result.accepted and len(t.renders) == 2 and len(client.calls) == 1
    reserved = [r for r in records if r['event'] == 'bounded_action_reserved']
    assert [r['kind'] for r in reserved] == ['first_fixed_recipe', 'resample']
    assert len({r['actual']['noise_seed'] for r in reserved}) == 2
    assert len({r['actual']['family_fingerprint'] for r in reserved}) == 1
    assert records[-1]['reason'] == 'legal_actions_exhausted'


def test_planning_cost_can_exhaust_budget_before_a_new_render(transaction, monkeypatch):
    from PIL import Image
    from types import SimpleNamespace
    import utils.agent_repair_regions as regions
    import utils.agent_progressive_repair as progressive
    t = transaction
    t.verifier.outcomes[1] = verdict()
    clock = [0.0]
    monkeypatch.setattr(progressive, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(regions, 'read_video_frames', lambda *args: [Image.new('RGB', (16, 16))])
    class Client:
        def call(self, **kwargs):
            clock[0] = 2000.0
            return {'parsed': {'option_id': json.loads(kwargs['prompt'])['options'][0]['id'], 'reason': 'Synthetic delayed plan'}}
    with t.session(feedback_policy=BoundedFeedbackPolicy(Client())) as session:
        session.run_shot(0)
        records = session.ledger.records()
    assert len(t.renders) == 1
    assert records[-1]['reason'] == 'wall_budget_exhausted'


def test_post_planning_mutation_is_detected_before_execution(transaction, monkeypatch):
    from PIL import Image
    import utils.agent_repair_regions as regions
    t = transaction
    t.verifier.outcomes[1] = verdict()
    monkeypatch.setattr(regions, 'read_video_frames', lambda *args: [Image.new('RGB', (16, 16))])
    class Client:
        def call(self, **kwargs):
            path = t.out / 'mask_0.pt'
            payload = torch.load(path, weights_only=True)
            payload['masks'][0][:, 0, 0] = 0
            torch.save(payload, path)
            return {'parsed': {'option_id': json.loads(kwargs['prompt'])['options'][0]['id'], 'reason': 'Synthetic mutated input'}}
    with t.session(feedback_policy=BoundedFeedbackPolicy(Client())) as session:
        session.run_shot(0)
        records = session.ledger.records()
    assert len(t.renders) == 1
    assert any(r['event'] == 'progressive_execution_failed' and 'changed during planning' in r['error'] for r in records)
