#!/usr/bin/env python3
"""Prepare the next two originals in the already-frozen twelve-script order."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import yaml

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare(args):
    began=time.monotonic();root=args.root.resolve();cohort=args.cohort.resolve();reference=args.reference.resolve()
    read=lambda p:json.loads(p.read_text())
    prior=read(cohort/'plan.json');assert prior['protocol']=='twelve_exposed_scripts_native_development_pilot_v1'
    assert read(cohort/'queue_status.json')['status']=='completed'
    previous=prior['prepared_baseline_indices'];assert previous==[40,130]
    chosen=[c for c in prior['cases'] if c['index'] not in previous][:2]
    assert [c['index'] for c in chosen]==[10,70]
    manifest=read(reference/'manifest.json');assert read(reference/'status.json')['status']=='completed'
    checkout=Path(manifest['cwd'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==manifest['official_revision']
    assert sha(reference/'config.yaml')==manifest['config_sha256']
    assert manifest['source_sha256']==prior['generation_source_sha256']
    for name,digest in manifest['source_sha256'].items():assert sha(checkout/name)==digest
    assert manifest['paired_assets']==prior['paired_assets']
    for row in prior['paired_assets']:assert sha(checkout/row['path'])==row['sha256']
    original=yaml.safe_load((reference/'config.yaml').read_text())
    assert original['seed']==0 and 'repair_request' not in original and original['audit_prompt_switch'] is False
    root.mkdir(parents=True,exist_ok=False)
    plan={'protocol':'twelve_script_pilot_second_original_batch_v1','frozen_unix':time.time(),
        'prior_cohort_plan_sha256':sha(cohort/'plan.json'),'prior_cohort_plan':str(cohort/'plan.json'),
        'selection':'Next two unrendered cases in the original frozen scene order, without using new generation outcomes. The actor exclusion applied only to the original first batch.',
        'cases':prior['cases'],'batch_indices':[c['index'] for c in chosen],'previously_completed_original_indices':previous,
        'planned_script_count':12,'new_original_attempt_ceiling':2,'new_repair_or_control_attempt_ceiling':0,
        'expected_completed_original_count_after_this_batch':4,'remaining_originals_after_success':8,
        'seed':0,'seconds':60,'fps':16,'resolution':[832,480],'per_case_timeout_seconds':1800,
        'generation_source_sha256':manifest['source_sha256'],'paired_assets':prior['paired_assets'],
        'reference_manifest_sha256':sha(reference/'manifest.json'),'preparer_sha256':sha(Path(__file__)),
        'review_frames':prior['review_frames'],'benchmark_sha256':prior['benchmark_sha256'],
        'failure_policy':'One attempt per original, no restart, seed search, replacement or added appearance. Keep ns070 despite finite appearance-extractor coverage gaps. No automatic repair is scheduled by this batch.',
        'comparative_pilot_status':'Four planned/completed originals after success only. Full multi-strategy action/budget/evaluator protocol remains pending; no Ours comparison or efficacy claim.',
        'hidden_cast_metadata_to_generator':False,'held_out':False,'automatic_acceptance':False,'quality_claim':False,'human_labels':0}
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    write(root/'plan.json',plan);shutil.copyfile(Path(__file__),root/'prepare_source.py')
    for case,gpu in zip(chosen,[0,2]):
        folder=root/case['id']/'baseline';folder.mkdir(parents=True)
        config={**original,'data_path':str(folder/'initial_prompt.txt'),'output_folder':str(folder/'videos'),
            'adapter_diagnostics_dir':str(folder/'diagnostics'),
            'timed_segments':[{'seconds':10,'prompts':[text]} for text in case['prompts']]}
        (folder/'initial_prompt.txt').write_text(case['prompts'][0]+'\n')
        (folder/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        child={**manifest,'phase':plan['protocol'],'gpu':gpu,'source_benchmark_index':case['index'],
            'command':[sys.executable,'-u','inference.py','--config_path',str(folder/'config.yaml')],
            'timeout_seconds':1800,'config_sha256':sha(folder/'config.yaml'),'source_case_plan_sha256':sha(root/'plan.json'),
            'purpose':'Next two original videos from the frozen twelve-script development cohort, with no repair/control generation.'}
        write(folder/'manifest.json',child);shutil.copyfile(root/'plan.json',folder/'plan.json');shutil.copyfile(args.supervisor,folder/'supervisor.py')
    write(root/'preparation.json',{'status':'prepared_unlaunched','wall_seconds':time.monotonic()-began,'assets_verified':len(prior['paired_assets']),'new_original_indices':plan['batch_indices']})
    print(json.dumps(read(root/'preparation.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','cohort','reference','supervisor']:p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
