"""Transaction tests with synthetic render data; these assert no video quality."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_agent_edit_branching import DEMO
from utils.agent_constraints import Constraint, ConstraintAssessment
from utils.agent_edit import FRAMES_PER_CHUNK, LATENT_GRID, save_mask_payload
from utils.agent_progressive_repair import ProgressiveShotSession, improvement_decision
from utils.agent_runtime import AgentRuntimeConfig, NormativeGraph, Verdict, file_sha256
from utils.parent_latent_replay import ParentLatentReplay, validate_parent_mask


def verdict(hair='violated', garment='violated', identity='satisfied', **kwargs):
    rows = []
    for part, status in [('hair', hair), ('garment', garment), ('identity', identity)]:
        constraint = Constraint(part, 'identity' if part == 'identity' else 'attribute', part,
            subjects=('Iris',), shot_indices=(0,), parameters={'part': part, 'aspect': 'color'})
        rows.append(ConstraintAssessment(constraint, status).to_dict())
    return Verdict(all(r['status'] == 'satisfied' for r in rows), 'test',
                   metrics={'constraint_assessments': rows, **kwargs})


def test_partial_requires_real_target_gain_and_preserves_original_correct_content():
    old = verdict()
    partial = verdict(hair='satisfied')
    result = improvement_decision(old, partial, old, ['hair'])
    assert result['allow'] and result['unresolved'] == ['garment']
    assert not result['fully_verified_current_shot']
    assert not improvement_decision(old, verdict(hair='unknown'), old, ['hair'])['allow']
    assert not improvement_decision(old, verdict(hair='satisfied', identity='unknown'), old, ['hair'])['allow']
    # Two small changes cannot hide a regression against the fixed anchor.
    parent = verdict(hair='satisfied', identity='violated')
    assert not improvement_decision(parent, verdict(hair='satisfied', garment='satisfied', identity='violated'),
                                    old, ['garment'])['allow']


def test_parent_replay_pins_prefix_and_edit_source_but_regenerates_suffix(tmp_path):
    raw = tmp_path / 'parent.pt'
    video = tmp_path / 'parent.mp4'
    video.write_bytes(b'parent')
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 6, 1, 2, 2)
    torch.save(tensor, raw)
    config = dict(latent_path=str(raw), latent_sha256=file_sha256(raw), video_path=str(video),
                  video_sha256=file_sha256(video), chunk_start=1, chunk_end=1, num_chunks=3)
    replay = ParentLatentReplay(config)
    fresh = torch.full((1, 2, 1, 2, 2), -1.)
    assert torch.equal(replay.before_recache(0, fresh), tensor[:, :2])
    assert torch.equal(replay.before_edit(1, fresh), tensor[:, 2:4])
    assert replay.before_recache(1, fresh) is fresh
    assert replay.before_recache(2, fresh) is fresh
    assert replay.before_edit(2, fresh) is fresh
    with pytest.raises(ValueError, match='shape/dtype'):
        replay.before_edit(1, fresh.double())
    video.write_bytes(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        ParentLatentReplay(config)


def test_masks_cannot_cross_parent_or_coordinate_system(tmp_path):
    video = tmp_path / 'v.mp4'
    video.write_bytes(b'parent')
    path = save_mask_payload({1: torch.ones(8, 2, 3)}, tmp_path / 'mask.pt',
                             phrase='hair', video=video, frames_per_chunk=8, grid=(2, 3))
    kwargs = dict(parent_video=video, chunk_start=1, chunk_end=1, frames_per_chunk=8, grid=(2, 3))
    validate_parent_mask(path, **kwargs)
    with pytest.raises(ValueError, match='coordinate'):
        validate_parent_mask(path, **{**kwargs, 'grid': (3, 2)})
    video.write_bytes(b'other')
    with pytest.raises(ValueError, match='different parent'):
        validate_parent_mask(path, **kwargs)


@pytest.fixture
def transaction(tmp_path, monkeypatch):
    scene = tmp_path / 'scene'
    scene.mkdir()
    (scene / 'shot_durations.txt').write_text('1 1\n')
    (scene / 'global.json').write_text(json.dumps({'caption': 'Iris has red hair and a blue coat'}))
    for k in range(2):
        (scene / f'{k}.json').write_text(json.dumps({'caption': 'Iris walks'}))
    out = tmp_path / 'out'
    out.mkdir()
    base = tmp_path / 'base.yaml'
    base.write_text('data: {}\ninference: {}\nlogging: {seed: 0}\n')
    video = tmp_path / 'base.mp4'
    video.write_bytes(b'base')
    raw = tmp_path / 'base.pt'
    torch.save(torch.zeros(1, 6, 1, 2, 2), raw)

    def frames(video, indices, folder, stem):
        folder.mkdir(parents=True, exist_ok=True)
        result = []
        for index in indices:
            path = folder / f'{stem}_{index}.png'
            path.write_bytes(f'{video}:{index}'.encode())
            result.append(path)
        return result

    renders = []
    def render(config, *, latent_dir, **kwargs):
        import yaml
        cfg = yaml.safe_load(config.read_text())
        renders.append(cfg)
        # A fake renderer only verifies the transaction's real I/O contract.
        parent = torch.load(cfg['inference']['replay_parent']['latent_path'], weights_only=True)
        latent_dir.mkdir(parents=True, exist_ok=True)
        torch.save(parent + 1, latent_dir / 'latent_0000.pt')
        prefix = latent_dir.name.removesuffix('_latents')
        (latent_dir.parent / f'{prefix}-rank0-scene_regular.mp4').write_bytes(prefix.encode())

    monkeypatch.setattr(DEMO, 'extract_frames', frames)
    monkeypatch.setattr(DEMO, 'render', render)
    backend = DEMO.EditRenderBackend(scene='scene', scene_dir=scene, out_dir=out,
        base_config=base, baseline_video=video, baseline_latents=raw,
        render_gpu=0, clipseg_device='cpu', progressive=True)

    class Verifier:
        def __init__(self):
            self.state = []
            self.outcomes = {0: verdict(), 1: verdict(hair='satisfied'), 2: verdict()}
        def snapshot_state(self):
            return copy.deepcopy(self.state)
        def restore_state(self, state):
            self.state = copy.deepcopy(state)
        def judge(self, observation, normative):
            self.state.append(observation.attempt)
            row = self.outcomes.get(observation.attempt, self.outcomes[2])
            masks = {0: torch.ones(FRAMES_PER_CHUNK, *LATENT_GRID)}
            path = save_mask_payload(masks, out / f'mask_{observation.attempt}.pt', phrase='part',
                                     video=observation.telemetry['shot_video'])
            options = [{'id': f'attribute:Iris:{part}', 'mask_sha256': file_sha256(path),
                        'action': {'tool': 'edit_shot', 'params': {'shot': 0, 'prompt': f'repair {part}',
                                   'entry_sigma': 1., 'mask_path': str(path)}}} for part in ('hair', 'garment')]
            return Verdict(row.accept, row.reason, metrics={**row.metrics, 'repair_options': options},
                           input_fingerprint=observation.input_fingerprint)

    class Protection:
        def __init__(self):
            self.calls = []
            self.status = 'pass'
        def review(self, *args, **kwargs):
            self.calls.append(args)
            return {'allow': self.status == 'pass', 'checks': {'identity': {'status': self.status}}}

    verifier, protection = Verifier(), Protection()
    def session(max_edits=3, lookahead=0):
        cfg = AgentRuntimeConfig.from_config({'enabled': True, 'max_edits_per_shot': max_edits,
            'commit_lookahead_shots': lookahead, 'lock_path': str(out / 'lock')})
        return ProgressiveShotSession(cfg, NormativeGraph.compile_scene(scene), backend,
            DEMO.AbstainingController(), verifier, protection=protection,
            ledger_path=out / 'ledger.jsonl', acquire_lock=False)
    return SimpleNamespace(backend=backend, verifier=verifier, protection=protection, session=session,
                           renders=renders, out=out)


def test_partial_is_next_parent_failed_child_does_not_erase_it_or_teach_identity(transaction):
    t = transaction
    with t.session() as s:
        result = s.run_shot(0)
        assert not result.accepted
        assert s.memory.partition_counts()['persistent'] == 0
        assert not s.memory.audit_invisible()
    best = t.backend.active_branch
    assert best.branch_id == 'branch-0001'
    assert best.repair_status == 'partial_improvement'
    assert best.unresolved_constraints == ('garment',)
    assert not best.verified and not best.reference_frames
    assert t.verifier.state == []
    assert t.backend.fully_verified_branch_id is None
    second = t.renders[1]['inference']
    assert second['replay_parent']['latent_path'] == best.raw_latents
    assert len(second['edit_renoise_plans']) == 1
    assert all(b.quarantined for b in list(t.backend.branches.values())[2:])


def test_full_second_repair_can_commit_and_restore_verifier_transaction(transaction):
    t = transaction
    t.verifier.outcomes[2] = verdict(hair='satisfied', garment='satisfied')
    with t.session() as s:
        result = s.run_shot(0)
        assert result.accepted
        assert s.memory.partition_counts() == {'candidate': 0, 'persistent': 1, 'quarantine': 2}
    assert t.backend.fully_verified_branch_id == 'branch-0002'
    assert t.verifier.state == [2]


def test_unknown_visual_protection_stops_without_noise_retry(transaction):
    t = transaction
    t.protection.status = 'unknown'
    with t.session() as s:
        result = s.run_shot(0)
        finished = [r for r in s.ledger.records() if r['event'] == 'repair_finished'][-1]
    assert finished['reason'] == 'insufficient_evidence'
    assert not result.accepted and len(t.renders) == 1
    assert t.backend.active_branch_id == 'branch-0000'


def test_execution_failure_delivers_best_and_keeps_failure_in_ledger(transaction, monkeypatch):
    t = transaction
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic render failure')
    monkeypatch.setattr(DEMO, 'render', fail)
    with t.session() as s:
        result = s.run_shot(0)
        records = s.ledger.records()
        assert s.memory.partition_counts()['candidate'] == 0
    assert not result.accepted and t.backend.active_branch_id == 'branch-0000'
    assert any(r['event'] == 'progressive_execution_failed' for r in records)
    assert records[-1]['termination'] == 'execution_failed'


def test_mutated_retained_artifact_cannot_be_snapshotted(transaction):
    t = transaction
    Path(t.backend.active_branch.raw_latents).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='artifacts changed'):
        t.backend.snapshot(0)


def test_future_protected_damage_vetoes_current_partial_gain(transaction):
    t = transaction
    calls = []
    def protect(parent, candidate, *args, **kwargs):
        calls.append(parent.shot_index)
        status = 'pass' if parent.shot_index == 0 else 'fail'
        return {'allow': status == 'pass', 'checks': {'required_actions': {'status': status}}}
    t.protection.review = protect
    with t.session(max_edits=1, lookahead=1) as s:
        result = s.run_shot(0)
        guards = [r for r in s.ledger.records() if r['event'] == 'progressive_future_guard']
    assert calls == [0, 1]
    assert not result.accepted and not guards[0]['passed']
    assert t.backend.active_branch_id == 'branch-0000'
    assert t.verifier.state == []
