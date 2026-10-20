#!/usr/bin/env python3
"""Replay a native controller transaction using previously recorded Iris evidence.

This is an integration audit, not a new generation experiment. Archived judgments
are reused only after source-video/frame/script binding; incompatible protection
scope is unknown and stops. No archived candidate is retroactively accepted.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
IMPLEMENTATION=ROOT/'implementation' if (ROOT/'implementation').is_dir() else ROOT
sys.path.insert(0,str(IMPLEMENTATION))
ADAPTER=ROOT/'adapters/official_longlive_rag'
sys.path.insert(0,str(ADAPTER))
from progressive_backend import NativeReplayBackend, read_json, only
from audit_candidate import audit
from utils.agent_bounded_feedback import BoundedFeedbackPolicy, actual_action
from utils.agent_constraints import Constraint, ConstraintAssessment, ConstraintEvidence
from utils.agent_progressive_repair import ProgressiveShotSession
from utils.agent_runtime import (Action, AgentRuntimeConfig, NormativeGraph, AbstainingController,
                                 Verdict, file_sha256)


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write_json(out/'status.json',{'status':'running','pid':__import__('os').getpid()})
    try:
        parent,candidate=args.parent.resolve(),args.candidate.resolve()
        diagnostics=args.diagnostics.resolve()
        config=yaml.safe_load((candidate/'config.yaml').read_text())
        request=config['repair_request']
        assert request['latent_range']==[81,121] and request['entry_sigma']==1.0
        assert request['noise_seed']==4243
        assert read_json(parent/'manifest.json')['source_benchmark_index']==90
        prompts=[item['prompts'][0] for item in config['timed_segments']]
        assert yaml.safe_load((parent/'config.yaml').read_text())['timed_segments']==config['timed_segments']
        # Re-run actual latent/cache/RNG/cursor checks without rendering.
        audit(candidate,parent,out/'candidate_engineering_audit.json')
        scene=out/'scene';scene.mkdir()
        write_json(scene/'global.json',{'caption':'','source':'Unmodified authored ns090 prompts'})
        for index,prompt in enumerate(prompts): write_json(scene/f'{index}.json',{'caption':prompt})
        backend=NativeReplayBackend(parent=parent,output=out/'backend',checkout=args.checkout,
            assets_manifest=args.assets_manifest,python=args.python,gpu=2,noise_seed=4242)
        bindings=[]
        def recorded_execute(choice,bound_parent,unused_output):
            old=Action('edit_shot',{'shot':2,'mask_path':request['mask_path'],
                'prompt':request['prompts'][0],'entry_sigma':1.0}).normalized(shot_index=2)
            actual=actual_action(old,parent_video=only(parent/'videos','*.mp4'),
                parent_latents=parent/'diagnostics/latents_0000.pt',seed=request['noise_seed'])
            if choice['actual']!=actual or Path(bound_parent['run'])!=parent:
                raise ValueError('No recorded candidate for this exact native action and parent')
            bindings.append({'actual_action':actual,'existing_candidate':str(candidate),
                'action_fingerprint':read_json(candidate/'manifest.json')['action_fingerprint'],
                'new_generation':False})
            return candidate
        backend._execute=recorded_execute
        before=read_json(args.temporal_fix/'result.json')
        baseline_provenance=read_json(args.baseline_attributes/'manifest.json')
        assert before['input_manifest_sha256']==file_sha256(args.baseline_attributes/'manifest.json')
        assert before['input_result_sha256']==file_sha256(args.baseline_attributes/'result.json')
        assert baseline_provenance['raw_script']==prompts
        after=read_json(diagnostics/'candidate_current/result.json')['result']
        after_provenance=read_json(diagnostics/'candidate_current/manifest.json')
        assert after_provenance['raw_script']==prompts
        current_guard=read_json(diagnostics/'current_guard/result.json')['review']
        guard_manifest=read_json(diagnostics/'current_guard/manifest.json')
        assert guard_manifest['fixed_script']['raw_prompts']==prompts[:3]
        sources=[parent/'manifest.json',candidate/'manifest.json',candidate/'plan.json',
            args.temporal_fix/'result.json',args.baseline_attributes/'manifest.json',args.baseline_attributes/'result.json',
            diagnostics/'candidate_current/result.json',diagnostics/'candidate_current/manifest.json',
            diagnostics/'current_guard/result.json',diagnostics/'current_guard/manifest.json']
        bound_sources={str(path):file_sha256(path) for path in sources}

        class RecordedVerifier:
            def snapshot_state(self): return {}
            def restore_state(self,state): assert state=={}
            def judge(self,observation,normative):
                shot=observation.shot_index
                if shot!=2:
                    constraint=Constraint(f'{shot}:not_evaluated','state','No archived current-shot judgment',
                        shot_indices=(shot,),expected='complete current-shot evaluation')
                    return Verdict(False,'not_evaluated_by_this_integration_audit',
                        metrics={'constraint_assessments':[ConstraintAssessment(constraint,'unknown').to_dict()]})
                source=Path(observation.telemetry['native_run'])
                if source==parent: rows,provenance=before['attribute_assessments'],baseline_provenance
                elif source==candidate: rows,provenance=after['attribute_assessments'],after_provenance
                else: raise ValueError('No archived judgment for this native branch')
                if file_sha256(observation.telemetry['shot_video'])!=provenance['video_sha256']:
                    raise ValueError('Archived judgment source video mismatch')
                if observation.telemetry['frame_indices']!=provenance['frame_indices']:
                    raise ValueError('Archived judgment time coordinates mismatch')
                expected=[provenance['input_sha256'][f'frame_{index:04d}.png'] for index in provenance['frame_indices']]
                if [file_sha256(p) for p in observation.frames]!=expected:
                    raise ValueError('Archived judgment frame pixels/encoding mismatch')
                assessments=[]
                for row in rows:
                    constraint=Constraint(row['id'],'attribute',row['expected'],subjects=('Iris',),shot_indices=(2,),
                        expected=row['expected'],parameters={'part':row['part'],'aspect':row['aspect'],'basis':'fixed_script_quote'})
                    assert row['expected'] in ' '.join(prompts[:3])
                    assessments.append(ConstraintAssessment(constraint,{'pass':'satisfied','fail':'violated','unknown':'unknown'}[row['status']],
                        evidence=(ConstraintEvidence('archived_attribute_diagnostic',observed=row),)).to_dict())
                options=[]
                if source==parent:
                    options=[{'id':'attribute:Iris:garment','mask_sha256':request['mask_sha256'],
                        'action':{'tool':'edit_shot','params':{'shot':2,'mask_path':request['mask_path'],
                            'prompt':request['prompts'][0],'entry_sigma':1.0}}}]
                return Verdict(all(row['status']=='satisfied' for row in assessments),'archived_attribute_evidence',
                    metrics={'constraint_assessments':assessments,'repair_options':options,
                        'repair_specs':{},'repair_references':{},'archived_evidence':True})

        class RecordedProtection:
            def __init__(self): self.calls=[]
            def review(self,parent_observation,candidate_observation,original,normative,**kwargs):
                for role,observation in [('parent',parent_observation),('candidate',candidate_observation)]:
                    assert file_sha256(observation.telemetry['shot_video'])==guard_manifest['video_sha256'][role]
                    assert observation.telemetry['frame_indices']==guard_manifest['frame_indices']
                    for index,path in zip(guard_manifest['frame_indices'],observation.frames):
                        assert file_sha256(path)==file_sha256(diagnostics/'current_guard'/f'{role}_{index:04d}.png')
                targets=sorted(row['id'] for row in kwargs['target_requirements'])
                # The old probe targeted color alone; shared fixed-action policy
                # targets both currently failing predicates in the same part.
                # Its guard cannot be presented as a judgment of this new packet.
                historical_targets=['4d74f1ec43a851fa']
                assert targets==['4d74f1ec43a851fa','7369e2f8e0c63452']
                result={'allow':False,'checks':{'identity':{'status':'unknown',
                    'observation':'No compatible protection judgment for the current target scope'}},
                    'error':'archived_guard_target_scope_mismatch', 'requested_target_ids':targets,
                    'archived_target_ids':historical_targets,
                    'archived_raw_guard':current_guard, 'archived_guard_adopted':False}
                self.calls.append(result)
                return result

        protection=RecordedProtection()
        runtime=AgentRuntimeConfig.from_config({'enabled':True,'max_edits_per_shot':1,
            'commit_lookahead_shots':1,'lock_path':str(out/'session.lock')})
        with ProgressiveShotSession(runtime,NormativeGraph.compile_scene(scene),backend,AbstainingController(),
                RecordedVerifier(),protection=protection,feedback_policy=BoundedFeedbackPolicy(),
                ledger_path=out/'ledger.jsonl',max_repair_seconds=600) as session:
            results=[session.run_shot(k) for k in range(3)]
            counts=session.memory.partition_counts()
            assert not session.memory.audit_invisible()
        assert len(bindings)==1 and len(protection.calls)==1
        assert results[-1].delivered['branch_id']=='native-0000'
        assert backend.active_branch['shot_outcomes'][-1]['termination']=='insufficient_evidence'
        assert backend.branches['native-0001']['quarantined']
        assert backend.fully_verified_branch_id is None and counts['persistent']==0
        assert all(file_sha256(path)==digest for path,digest in bound_sources.items())
        source_hashes={}
        for name in ['scripts/probe_native_progressive_transaction.py','utils/agent_progressive_repair.py',
            'utils/agent_runtime.py','utils/agent_bounded_feedback.py','utils/agent_constraints.py',
            'adapters/official_longlive_rag/progressive_backend.py','adapters/official_longlive_rag/feedback_bridge.py',
            'adapters/official_longlive_rag/native_region_preview.py','adapters/official_longlive_rag/audit_candidate.py']:
            src=(IMPLEMENTATION/name if name.startswith('utils/') else ROOT/name)
            dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
            source_hashes[name]=file_sha256(dst)
        report={'phase':'native_archived_transaction_integration','input_sha256':bound_sources,
            'source_sha256':source_hashes,'bindings':bindings,'current_guard':protection.calls[0],
            'selected_video_sha256':file_sha256(results[-1].delivered['video']),
            'original_video_sha256':file_sha256(only(parent/'videos','*.mp4')),
            'render_calls_consumed_by_transaction':1,'new_video_generations':0,'new_model_calls':0,
            'render_executor':'strict archived exact-action lookup; production subprocess executor not run',
            'retained_candidates':0,'future_guard_reached':False,'trusted_memory_counts':counts,
            'wall_seconds':time.monotonic()-started,'quality_claim':False,'held_out':False,
            'limits':['Reuses existing exposed Iris development evidence; not a new efficacy observation.',
                'First two shots explicitly unknown; no fabricated passing verdict.',
                'Current policy targets both garment failures; older one-target protection is incompatible.',
                'Unknown protection stops without reusing or adopting the known unreliable old verdict.',
                'Production generator invocation still requires a live integration trial and compatible verifier.']}
        write_json(out/'result.json',report)
        write_json(out/'status.json',{'status':'completed','wall_seconds':time.monotonic()-started})
        print(json.dumps({key:report[key] for key in ['wall_seconds','retained_candidates','new_video_generations','future_guard_reached']}))
    except Exception as error:
        write_json(out/'status.json',{'status':'execution_failed','error':f'{type(error).__name__}: {error}',
            'wall_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','candidate','diagnostics','baseline-attributes','temporal-fix','output','checkout','assets-manifest']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--python',required=True)
    torch.set_num_threads(4)
    run(p.parse_args())
