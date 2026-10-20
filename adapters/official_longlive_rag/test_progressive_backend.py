"""Native transaction contracts with explicitly synthetic videos and render jobs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from progressive_backend import NativeReplayBackend
from utils.agent_bounded_feedback import BoundedFeedbackPolicy
from utils.agent_constraints import Constraint, ConstraintAssessment
from utils.agent_progressive_repair import ProgressiveShotSession
from utils.agent_runtime import (AbstainingController, Action, AgentRuntimeConfig, NormativeGraph,
                                 Verdict, file_sha256, fingerprint)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def make_run(path, *, parent=None, value=20):
    path.mkdir()
    (path/'videos').mkdir()
    (path/'diagnostics').mkdir()
    video=path/'videos/test.mp4'
    writer=cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 16, (32,32))
    for index in range(32):
        writer.write(np.full((32,32,3), value+index, np.uint8))
    writer.release()
    torch.save(torch.full((1,9,1,2,2),float(value)),path/'diagnostics/latents_0000.pt')
    timing={'protocol':'continuous_vae_drop_first_frame_v1','export_frames':32,'fps':16,
        'segments':[{'segment':0,'start_latent_frame':0,'latent_frames':5,'start_export_frame':0,'export_frames':16},
                    {'segment':1,'start_latent_frame':5,'latent_frames':4,'start_export_frame':16,'export_frames':16}]}
    write_json(path/'diagnostics/timing_0000.json',timing)
    write_json(path/'diagnostics/switches_0000.json',[])
    write_json(path/'videos/test_memory_log.json',[])
    write_json(path/'status.json',{'status':'completed'})
    (path/'config.yaml').write_text('timed_segments: []\n')
    manifest={'config_sha256':file_sha256(path/'config.yaml')}
    if parent:
        manifest['parent_manifest_sha256']=file_sha256(parent/'manifest.json')
        write_json(path/'diagnostics/repair_0000.json',[])
    write_json(path/'manifest.json',manifest)
    write_json(path/'audit.json',{'video_sha256':file_sha256(video),
        'latent_sha256':file_sha256(path/'diagnostics/latents_0000.pt'),
        'all_engineering_checks_passed':True,'synthetic_fixture':True})
    return path


@pytest.fixture
def transaction(tmp_path):
    parent=make_run(tmp_path/'parent')
    backend=NativeReplayBackend(parent=parent, output=tmp_path/'output',checkout=tmp_path,
        assets_manifest=tmp_path/'assets',python=sys.executable,gpu=0)
    scene=tmp_path/'scene'
    scene.mkdir()
    write_json(scene/'global.json',{'caption':'Iris has red hair and a green coat.'})
    for k in range(2):
        write_json(scene/f'{k}.json',{'caption':'Iris walks in the room.'})
    requests=[]
    def execute(choice, parent, output):
        requests.append({'choice':choice,'parent':deepcopy(parent)})
        return make_run(output,parent=Path(parent['run']),value=20+len(requests)*20)
    backend._execute=execute

    class Verifier:
        def __init__(self):
            self.state=[]
            self.calls=[]
            self.future_fail=False
            self.all_targets=False
            self.stale=False
        def snapshot_state(self): return deepcopy(self.state)
        def restore_state(self,state): self.state=deepcopy(state)
        def judge(self,observation,normative):
            self.state.append(observation.attempt)
            shot=observation.shot_index
            self.calls.append((shot,observation.attempt,observation.telemetry['native_run']))
            statuses=['violated','violated'] if observation.attempt==0 else ['satisfied','satisfied' if self.all_targets else 'violated']
            if observation.attempt>=2: statuses=['violated','violated']
            if shot==1:
                statuses=['satisfied','violated' if self.future_fail and observation.attempt else 'satisfied']
            rows=[]
            for part,status in zip(('hair','garment'),statuses):
                constraint=Constraint(f'{shot}:{part}','attribute',part,subjects=('Iris',),shot_indices=(shot,),
                    expected='red' if part=='hair' else 'green',
                    parameters={'part':part,'aspect':'color','basis':'fixed_script_quote'})
                rows.append(ConstraintAssessment(constraint,status).to_dict())
            folder=Path(observation.telemetry['native_run'])
            bound=parent if self.stale and observation.attempt else folder
            segment=backend.timing['segments'][shot]
            mask=backend.output/f'mask_{folder.name}_{shot}.pt'
            torch.save({'mask':torch.ones(segment['latent_frames'],2,2,dtype=torch.bool),
                'latent_range':[segment['start_latent_frame'],segment['start_latent_frame']+segment['latent_frames']],
                'grid':[2,2], 'parent_video_sha256':file_sha256(next((bound/'videos').glob('*.mp4'))),
                'parent_latents_sha256':file_sha256(bound/'diagnostics/latents_0000.pt'),
                'parent_timing_sha256':file_sha256(bound/'diagnostics/timing_0000.json')},mask)
            options=[{'id':f'attribute:Iris:{part}','mask_sha256':file_sha256(mask),
                'action':{'tool':'edit_shot','params':{'shot':shot,'mask_path':str(mask),'entry_sigma':1.,
                    'prompt':'Iris walks in the room. Iris has red hair and a green coat.'}}} for part in ('hair','garment')]
            return Verdict(all(s=='satisfied' for s in statuses),'synthetic',
                metrics={'constraint_assessments':rows,'repair_options':options})
    class Protection:
        def __init__(self): self.calls=[]; self.status='pass'; self.future_unknown=False
        def review(self,*args,**kwargs):
            self.calls.append((args,kwargs))
            status='unknown' if self.future_unknown and args[0].shot_index==1 else self.status
            return {'allow':status=='pass','checks':{'identity':{'status':status}}}
    verifier,protection=Verifier(),Protection()
    def session(attempts=2, lookahead=1, feedback=True):
        cfg=AgentRuntimeConfig.from_config({'enabled':True,'max_edits_per_shot':attempts,
            'commit_lookahead_shots':lookahead,'lock_path':str(tmp_path/'lock')})
        return ProgressiveShotSession(cfg,NormativeGraph.compile_scene(scene),backend,AbstainingController(),verifier,
            protection=protection,feedback_policy=BoundedFeedbackPolicy() if feedback else None,
            ledger_path=backend.output/'ledger.jsonl',acquire_lock=False)
    return SimpleNamespace(backend=backend,verifier=verifier,protection=protection,session=session,requests=requests)


def test_partial_native_child_is_next_parent_rejected_child_rolls_back(transaction):
    t=transaction
    with t.session() as session:
        result=session.run_shot(0)
        assert result.quarantined and not result.accepted
        assert session.memory.partition_counts()['persistent']==0
        assert not session.memory.audit_invisible()
    assert len(t.requests)==2
    assert t.requests[1]['parent']['branch_id']=='native-0001'
    assert t.requests[1]['choice']['actual']['parent_latents_sha256']==file_sha256(t.requests[1]['parent']['raw_latents'])
    assert t.backend.active_branch_id=='native-0001'
    assert t.backend.active_branch['unresolved_constraints']==['0:garment']
    assert t.backend.branches['native-0002']['quarantined']
    assert t.backend.fully_verified_branch_id is None and t.verifier.state==[]
    assert not any(b['trusted_identity_reference'] for b in t.backend.branches.values())
    assert t.protection.calls[1][1]['target_requirements'][0]['expected']=='red'
    assert [r['expected'] for r in t.protection.calls[1][1]['protected_requirements']]==['green']


def test_native_unknown_current_guard_stops_without_identity_promotion(transaction):
    t=transaction;t.protection.status='unknown'
    with t.session() as session:
        result=session.run_shot(0)
    assert len(t.requests)==1 and result.delivered['branch_id']=='native-0000'
    assert t.backend.active_branch['shot_outcomes'][0]['termination']=='insufficient_evidence'


@pytest.mark.parametrize('unknown',[False,True])
def test_native_future_regression_or_unknown_retains_original(transaction,unknown):
    t=transaction;t.verifier.all_targets=True
    t.verifier.future_fail=not unknown;t.protection.future_unknown=unknown
    with t.session(attempts=1) as session:
        result=session.run_shot(0)
    assert not result.accepted and result.delivered['branch_id']=='native-0000'
    assert t.backend.branches['native-0001']['quarantined']
    assert not t.backend.pending


def test_native_acceptance_does_not_certify_unchecked_suffix(transaction):
    t=transaction;t.verifier.all_targets=True
    with t.session(attempts=1) as session:
        result=session.run_shot(0)
        assert result.accepted and t.backend.fully_verified_branch_id is None
        result=session.run_shot(1)
        assert result.accepted
    assert t.backend.fully_verified_branch_id=='native-0001'
    assert not t.backend.active_branch['reference_frames']


def test_new_native_parent_rejects_old_mask_before_second_render(transaction):
    t=transaction;t.verifier.stale=True
    with t.session(attempts=2) as session:
        result=session.run_shot(0)
    assert len(t.requests)==1 and result.delivered['branch_id']=='native-0001'
    assert t.backend.active_branch['shot_outcomes'][0]['termination']=='execution_failed'


@pytest.mark.parametrize('artifact',['config.yaml','diagnostics/latents_0000.pt','diagnostics/switches_0000.json','videos/test_memory_log.json'])
def test_native_snapshot_detects_changed_reconstruction_state(transaction,artifact):
    t=transaction;snapshot=t.backend.snapshot(0)
    (Path(t.backend.active_branch['run'])/artifact).write_bytes(b'mutated')
    with pytest.raises(ValueError,match='artifact changed'): t.backend.restore(snapshot)


def test_native_future_preview_does_not_mutate_branch_or_cursor(transaction):
    t=transaction;generated=t.backend.generate(0,0);before=t.backend.snapshot(0)
    preview=t.backend.preview(generated,1)
    assert preview.telemetry['frame_indices']==[19,24,28]
    assert fingerprint(before)==fingerprint(t.backend.snapshot(0))


def test_native_preview_uses_latent_export_projection(transaction):
    t=transaction;generated=t.backend.generate(0,0)
    observation=t.backend.observe(generated);verdict=t.verifier.judge(observation,None)
    option=verdict.metrics['repair_options'][0]
    preview=t.backend.region_preview(option,video=observation.telemetry['shot_video'],output_dir=t.backend.output/'previews')
    assert preview['frame_indices']==[0,4,12]
    assert file_sha256(preview['path'])==preview['sha256']


def test_render_failure_preserves_original_and_records_execution_failure(transaction):
    t=transaction
    def fail(*args): raise RuntimeError('Synthetic failed native subprocess')
    t.backend._execute=fail
    with t.session() as session: result=session.run_shot(0)
    assert result.delivered['branch_id']=='native-0000'
    assert t.backend.active_branch['shot_outcomes'][0]['termination']=='execution_failed'


def test_changed_evidence_cannot_silently_replace_cached_native_frames(transaction):
    t=transaction;generated=t.backend.generate(0,0)
    Path(generated.output['frames'][0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): t.backend.observe(generated)
    with pytest.raises(ValueError,match='changed'): t.backend.generate(0,0)


def test_production_execution_orders_preparation_supervision_and_audit(transaction,monkeypatch):
    t=transaction
    # Exercise the production executor itself; external tools are isolated.
    calls=[];output=t.backend.output/'production_stub'
    def run(command,**kwargs):
        calls.append(command)
        if command[1].endswith('feedback_bridge.py'):
            request=json.loads(Path(command[3]).read_text())
            assert request['parent']==t.backend.active_branch['run']
            assert request['gpu']==0 and request['timeout']==1800
            write_json(Path(command[5]),{'command':[t.backend.python,str(output/'supervisor.py'),str(output)]})
        elif command[1].endswith('supervisor.py'):
            output.mkdir();write_json(output/'status.json',{'status':'completed'})
        elif command[1].endswith('audit_candidate.py'):
            assert command[command.index('--parent')+1]==t.backend.active_branch['run']
        else: raise AssertionError(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('progressive_backend.subprocess.run',run)
    choice={'action':Action('edit_shot',{'shot':0,'prompt':'Iris','entry_sigma':1.,'mask_path':'fixture'},reason='fixture'),
            'actual':{'noise_seed':4243}}
    assert NativeReplayBackend._execute(t.backend,choice,t.backend.active_branch,output)==output
    assert [Path(c[1]).name for c in calls]==['feedback_bridge.py','supervisor.py','audit_candidate.py']


def test_resource_refusal_does_not_run_engineering_audit_or_register_branch(transaction,monkeypatch):
    t=transaction;output=t.backend.output/'refused';calls=[]
    def prepare(*args):
        output.mkdir();write_json(output/'status.json',{'status':'resource_refused'})
        return {'command':['fake','supervisor.py']}
    monkeypatch.setattr(t.backend,'prepare_action',prepare)
    monkeypatch.setattr('progressive_backend.subprocess.run',lambda *args,**kwargs:calls.append(args))
    with pytest.raises(RuntimeError,match='resource_refused'):
        NativeReplayBackend._execute(t.backend,{},t.backend.active_branch,output)
    assert len(calls)==1 and len(t.backend.branches)==1


def test_fixed_native_mode_also_handles_stale_mask_without_losing_partial(transaction):
    t=transaction;t.verifier.stale=True
    with t.session(feedback=False) as session: result=session.run_shot(0)
    assert len(t.requests)==1 and result.delivered['branch_id']=='native-0001'
    assert t.backend.active_branch['shot_outcomes'][0]['termination']=='execution_failed'


def test_native_preparation_never_overwrites_a_previous_attempt(transaction):
    t=transaction;output=t.backend.output/'previous'
    request=output.with_suffix('.request.json');request.write_text('original failed attempt')
    with pytest.raises(FileExistsError): t.backend.prepare_action({},t.backend.active_branch,output)
    assert request.read_text()=='original failed attempt'


def test_render_timeout_is_distinct_from_execution_failure(transaction):
    t=transaction
    def timeout(*args): raise TimeoutError('Synthetic frozen renderer deadline')
    t.backend._execute=timeout
    with t.session() as session: result=session.run_shot(0)
    assert result.delivered['branch_id']=='native-0000'
    assert t.backend.active_branch['shot_outcomes'][0]['termination']=='timeout'


def test_supervisor_timeout_is_propagated_without_audit(transaction,monkeypatch):
    t=transaction;output=t.backend.output/'timeout';calls=[]
    def prepare(*args):
        output.mkdir();write_json(output/'status.json',{'status':'timeout'})
        return {'command':['fake','supervisor.py']}
    monkeypatch.setattr(t.backend,'prepare_action',prepare)
    monkeypatch.setattr('progressive_backend.subprocess.run',lambda *args,**kwargs:calls.append(args))
    with pytest.raises(TimeoutError): NativeReplayBackend._execute(t.backend,{},t.backend.active_branch,output)
    assert len(calls)==1


def test_offline_window_checks_selected_shot_without_certifying_prefix(transaction):
    t=transaction
    with t.session(attempts=0) as session:
        with pytest.raises(RuntimeError,match='cannot start'):
            session.pre_shot(1)
        session.begin_offline_window(1)
        assert session.committed_shots==set()
        assert session.memory.partition_counts()=={'candidate':0,'persistent':0,'quarantine':0}
        assert t.verifier.state==[] and t.backend.delivered==[]
        with pytest.raises(RuntimeError,match='before the declared'):
            session.pre_shot(0)
        with pytest.raises(RuntimeError,match='cannot start'):
            session.pre_shot(2)
        result=session.run_shot(1)
        assert result.accepted and session.committed_shots=={1}
        with pytest.raises(RuntimeError,match='fresh session'):
            session.begin_offline_window(1)
    assert not t.requests
    assert {r['shot'] for r in t.backend.active_branch['shot_outcomes']}=={1}
    assert t.backend.fully_verified_branch_id is None
    assert t.backend.telemetry(1)['committed_shots']==1
    assert t.backend.telemetry(1)['next_shot_cursor']==2
    events=[r for r in session.ledger.records() if r['event']=='offline_window_start']
    assert len(events)==1 and events[0]['unassessed_prefix']==[0]
    assert events[0]['parent_binding']['artifacts']==t.backend.active_branch['artifacts']
    assert not events[0]['trusted_memory_added']


@pytest.mark.parametrize('start',[-1,0,2,True])
def test_offline_window_rejects_invalid_start(transaction,start):
    with transaction.session() as session:
        with pytest.raises(ValueError,match='valid nonzero'):
            session.begin_offline_window(start)


def test_offline_window_requires_unchanged_audited_parent(transaction):
    t=transaction
    Path(t.backend.active_branch['video']).write_bytes(b'changed parent')
    with t.session() as session:
        with pytest.raises(ValueError,match='artifact changed'):
            session.begin_offline_window(1)
        assert session._offline_window_start is None and not session.committed_shots


def test_offline_window_requires_backend_capability(transaction,monkeypatch):
    t=transaction
    monkeypatch.setattr(t.backend,'offline_window_context',None)
    with t.session() as session:
        with pytest.raises(RuntimeError,match='cannot bind'):
            session.begin_offline_window(1)
