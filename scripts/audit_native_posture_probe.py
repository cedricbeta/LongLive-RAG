#!/usr/bin/env python3
"""Check the frozen posture probe's actual conditioning, parent and latent scope."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
import yaml

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def audit(root):
    started=time.monotonic();torch.set_num_threads(4)
    load=lambda p:json.loads(p.read_text())
    plan=load(root/'plan.json');assert plan['protocol']=='ns040_fixed_target_posture_and_same_boundary_prompt_v1'
    assert not (root/'pair_check.json').exists()
    parent=Path(plan['parent']);base=torch.load(parent/'diagnostics/latents_0000.pt',map_location='cpu',weights_only=True)
    assert sha(parent/'diagnostics/latents_0000.pt')==plan['parent_latents_sha256']
    rows=[]
    for arm in ['P_prompt']+(['R_group'] if (root/'R_group/status.json').exists() else []):
        folder=root/arm;status=load(folder/'status.json');assert status['status']=='completed'
        manifest=load(folder/'manifest.json');config=yaml.safe_load((folder/'config.yaml').read_text())
        assert manifest['study_plan_sha256']==sha(root/'plan.json')
        assert sha(folder/'config.yaml')==manifest['config_sha256']
        for name,digest in manifest['source_sha256'].items():assert sha(Path(manifest['cwd'])/name)==digest
        expected=plan['effective_prompt_control_segments'] if arm=='P_prompt' else plan['authored_segments']
        assert config['timed_segments']==expected
        assert config['seed']==0 and config['audit_prompt_switch'] is False
        if arm=='P_prompt':assert 'repair_request' not in config
        else:
            request=config['repair_request']
            assert request['prompts']==[plan['repair_prompt']] and request['latent_range']==[41,81]
            assert request['entry_sigma']==1.0 and request['noise_seed']==4245
            assert load(folder/'audit.json')['measured_engineering_checks_passed']
        switches=load(folder/'diagnostics/switches_0000.json');assert len(switches)==5
        for shot,row in enumerate(switches,1):
            assert row['start_latent_frame']==40*shot+1 and row['export_prefix_frames']==160*shot
            assert row['prompt_sha256']==[hashlib.sha256(x.encode()).hexdigest() for x in expected[shot]['prompts']]
            assert row['cache_positions_before']==row['cache_positions_after']
            assert row['retrieval_pool_counts_before']==row['retrieval_pool_counts_after']
            assert all(row['cross_attention_initialized']) and not row['history_content_audited']
        timing=load(folder/'diagnostics/timing_0000.json');assert timing['latent_frames']==241 and timing['export_frames']==960
        result=torch.load(folder/'diagnostics/latents_0000.pt',map_location='cpu',weights_only=True)
        assert result.shape==base.shape and result.dtype==base.dtype and torch.isfinite(result).all()
        assert torch.equal(result[:,:41],base[:,:41])
        video,=(folder/'videos').glob('*.mp4')
        row={'arm':arm,'prefix_latents_exact':41,'five_actual_prompt_hashes_match':True,
            'video_sha256':sha(video),'latent_sha256':sha(folder/'diagnostics/latents_0000.pt'),
            'generation_seconds':status['wall_seconds'],'new_private_forwards':0 if arm=='P_prompt' else load(folder/'audit.json')['additional_denoiser_calls'],
            'changed_latent_indices':[i for i in range(241) if not torch.equal(base[:,i],result[:,i])]}
        rows.append(row)
        if arm=='P_prompt':
            (folder/'audit.json').write_text(json.dumps({**row,'engineering_checks_passed':True,
                'scope':'Frozen source/config, actual switches, prefix/latent shape and finiteness. Full video decoding is recorded by the comparison exporter.',
                'history_content_hashes_audited':False,'resident_content_hashes_audited':False,'candidate_retained':False,'quality_claim':False},indent=2)+'\n')
    record={'status':'passed','arms':rows,'source_sha256':sha(Path(__file__)),'plan_sha256':sha(root/'plan.json'),
        'shared_nonempty_prefix_latents':41,'mask_and_replay_are_not_pixel_preservation_claims':True,
        'wall_seconds':time.monotonic()-started,'semantic_acceptance':False,'full_decode_record':'review/review.json'}
    (root/'pair_check.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root.resolve())
