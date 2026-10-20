#!/usr/bin/env python3
"""Freeze one exposed posture capability probe and a same-boundary prompt control."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import yaml

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(args):
    began=time.monotonic();root=args.root.resolve();parent=args.parent.resolve()
    assert not root.exists()
    read=lambda p:json.loads(p.read_text())
    manifest=read(parent/'manifest.json');audit=read(parent/'audit.json')
    assert read(parent/'status.json')['status']=='completed'
    assert manifest['source_benchmark_index']==40 and sha(parent/'config.yaml')==manifest['config_sha256']
    video,=(parent/'videos').glob('*.mp4')
    assert sha(video)==audit['video_sha256'] and sha(parent/'diagnostics/latents_0000.pt')==audit['latent_sha256']
    checkout=Path(manifest['cwd'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==manifest['official_revision']
    for name,digest in manifest['source_sha256'].items():assert sha(checkout/name)==digest
    for row in manifest['paired_assets']:assert sha(checkout/row['path'])==row['sha256']
    config=yaml.safe_load((parent/'config.yaml').read_text());authored=config['timed_segments']
    assert config['seed']==0 and 'repair_request' not in config and len(authored)==6
    assert all(s['seconds']==10 for s in authored)
    assert 'Nora and Ethan stand' in authored[1]['prompts'][0]
    cue='Nora and Ethan are both standing upright on their feet throughout this interaction.'
    private=authored[1]['prompts'][0]+' '+cue
    effective=[{'seconds':10,'prompts':[s['prompts'][0]+(' '+cue if i==1 else '')]} for i,s in enumerate(authored)]
    root.mkdir(parents=True)
    plan={'protocol':'ns040_fixed_target_posture_and_same_boundary_prompt_v1','frozen_unix':time.time(),
        'source_benchmark_index':40,'seed':0,'seconds':60,'fps':16,'resolution':[832,480],
        'selection':'Exposed ns040 selected after inspecting the original: visible sitting conflicts with the authored joint standing requirement. Not an automatic diagnostic admission or a held-out result.',
        'question':'Can the native masked replay change an authored posture error, and does it offer a visible benefit over ordinary conditioning with the identical strengthened prompt from the same boundary?',
        'arms':{'B_original':'Reused complete original.','P_prompt':'One ordinary generation; only the 10–20s prompt gets the appendix.','R_group':'At most one masked private replay of 10–20s using the same complete prompt; authored ordinary prompts and 20–60s remain unchanged.'},
        'new_generation_ceiling':2,'new_generation_per_arm':1,'timeout_per_generation_seconds':1800,
        'mask_generation_ceiling':1,'mask_timeout_seconds':600,'new_semantic_checker_calls':0,
        'parent':str(parent),'parent_video_sha256':audit['video_sha256'],'parent_latents_sha256':audit['latent_sha256'],
        'parent_manifest_sha256':sha(parent/'manifest.json'),'parent_timing_sha256':sha(parent/'diagnostics/timing_0000.json'),
        'parent_generation_seconds_previously_incurred':read(parent/'status.json')['wall_seconds'],
        'repair_segment':1,'repair_latent_range':[41,81],'repair_export_range':[160,320],
        'authored_segments':authored,'effective_prompt_control_segments':effective,'repair_prompt':private,'prompt_appendix':cue,
        'entry_sigma':1.0,'private_noise_seed':4245,'repair_checkout':str(args.repair_checkout.resolve()),
        'mask_recipe':{'phrase':'people','threshold':0.35,'model':str(args.clipseg.resolve()),
            'projection':'Segment all 160 actual frames. For each latent, union its four frame masks, form a program bounding rectangle, then pad 5% of its width horizontally and 15% of its height vertically.',
            'padding_fraction_xy':[0.05,0.15],'maximum_coverage_per_latent':0.85,'require_all_latents_nonempty':True,
            'face_exclusion':False,'reason':'Posture motion can move heads as well as bodies; identity preservation must be assessed from outcomes rather than assumed from face pinning.'},
        'mask_admission':'Before either result is inspected: require complete source-frame coverage, nonempty bounded program masks at every latent and visible coverage of the two actors in the three frozen overlays. If unsuitable, keep P and do not generate R. No mask/prompt/seed replacement.',
        'mask_preview_latents':[41,61,80],
        'review_frames':[f for s in range(6) for f in (s*160,s*160+80,s*160+159)],
        'pairing_rule':'P and R must each match all 41 original prefix latents. R must pin all edited cells outside the mask. Check all five actual prompt switches and decode all frames. Pixel equality is not inferred from latent pinning.',
        'paired_assets':manifest['paired_assets'],'generation_source_sha256':manifest['source_sha256'],
        'source_sha256':{'prepare':sha(Path(__file__)),'mask':sha(Path(__file__).with_name('prepare_native_group_mask.py'))},
        'failure_policy':'One attempt per arm, no additional prompt/seed/scope/sampler or case after outcomes. Preserve complete originals, both attempted outputs, failures, costs and all subsequent shots.',
        'limits':['Fixed-target capability probe; the previous two-call action diagnostic remains closed and unreliable.',
            'Program-derived group rectangles are a new coarse localization recipe, not validated identity tracking.',
            'P versus R changes ordinary conditioning versus masked private generation, not an isolated mask-only ablation.',
            'Actual segmentation and replay costs differ; this is not a budget-matched fixed-retry/Best-of-N comparison.',
            'No automatic, independent or human acceptance. The full twelve-script comparative pilot remains incomplete.'],
        'held_out':False,'automatic_acceptance':False,'quality_claim':False,'submission_ready':False}
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    write(root/'plan.json',plan);write(root/'assets_manifest.json',{'files':manifest['paired_assets']})
    shutil.copyfile(Path(__file__),root/'prepare_source.py')
    shutil.copyfile(Path(__file__).with_name('prepare_native_group_mask.py'),root/'mask_source.py')
    (root/'B_original').symlink_to(parent,target_is_directory=True)
    (root/'repair_prompt.txt').write_text(private+'\n')
    out=root/'P_prompt';out.mkdir()
    config.update(data_path=str(out/'initial_prompt.txt'),output_folder=str(out/'videos'),adapter_diagnostics_dir=str(out/'diagnostics'),timed_segments=effective)
    (out/'initial_prompt.txt').write_text(authored[0]['prompts'][0]+'\n')
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    child={**manifest,'phase':plan['protocol']+'_prompt_control','gpu':0,'command':[sys.executable,'-u','inference.py','--config_path',str(out/'config.yaml')],
        'timeout_seconds':1800,'config_sha256':sha(out/'config.yaml'),'study_plan_sha256':sha(root/'plan.json'),'correction_enabled':False,'quality_claim':False}
    write(out/'manifest.json',child);shutil.copyfile(root/'plan.json',out/'plan.json')
    shutil.copyfile(args.supervisor,out/'supervisor.py')
    write(root/'preparation.json',{'status':'prepared_unlaunched','wall_seconds':time.monotonic()-began,'assets_verified':len(manifest['paired_assets'])})
    print(json.dumps(read(root/'preparation.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','repair-checkout','clipseg','supervisor']:p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
