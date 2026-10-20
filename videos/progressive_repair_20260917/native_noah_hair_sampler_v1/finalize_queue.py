import hashlib, json, subprocess, time
from pathlib import Path
import yaml
S=Path('/home/chendong/video-gen/LongLive-RAG')
F=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(F/'plan.json');parent=Path(plan['parent']);pair=F/'sampler_pair';began=time.monotonic()
state={'phase':'waiting_existing_candidates','new_generation_launches':0,'started_unix':time.time()}
def write(): (F/'finalize_queue_status.json').write_text(json.dumps(state,indent=2)+'\n')
write()
remaining=['native','euler']
while remaining:
    for name in remaining[:]:
        c=pair/name
        if not (c/'status.json').exists():continue
        s=read(c/'status.json')
        if s['status'] in ('running','preflight'):continue
        state[name+'_generation_status']=s['status'];write()
        if s['status']!='completed':raise RuntimeError(name+' generation did not complete; no restart')
        with (F/(name+'_audit.log.txt')).open('x') as log:
            subprocess.run([str(S/'venv/bin/python'),str(S/'scripts/audit_native_fast_repair.py'),'--folder',str(c),'--parent',str(parent)],stdout=log,stderr=subprocess.STDOUT,check=True)
        expected={'native':'native_x0_renoise_v1','euler':'fixed_noise_euler_v1'}[name]
        config=yaml.safe_load((c/'config.yaml').read_text());manifest=read(c/'manifest.json')
        assert config['repair_request']['repair_sampler']==manifest['action']['repair_sampler']==expected
        assert hashlib.sha256(json.dumps(manifest['action'],sort_keys=True).encode()).hexdigest()==manifest['action_fingerprint']
        repairs=[r for r in read(c/'diagnostics/repair_0000.json') if r['phase']=='edit']
        assert len(repairs)==14 and all(r['repair_sampler']==expected and r['denoising_steps']==4 for r in repairs)
        if name=='euler':
            assert all(len(r['sampler_trace'])==4 for r in repairs)
            for row in repairs:
                trace=row['sampler_trace'];assert abs(trace[0]['sigma']-1)<1e-6 and trace[-1]['next_sigma']==0
                assert all(t['background_noise']=='same_initial_private_noise' for t in trace)
        else:assert all('sampler_trace' not in r for r in repairs)
        record={'status':'passed','mode':expected,'actual_private_calls':56,'euler_step_traces':sum(len(r.get('sampler_trace',[])) for r in repairs),'base_audit_sha256':sha(c/'audit.json'),'manifest_sha256':sha(c/'manifest.json'),'quality_claim':False}
        (c/'sampler_audit.json').write_text(json.dumps(record,indent=2)+'\n')
        remaining.remove(name);state[name+'_audit_status']='passed';write()
    if remaining:
        if time.monotonic()-began>2000:raise TimeoutError('Bounded wait; candidates not restarted')
        time.sleep(5)
a,b=[read(pair/n/'manifest.json') for n in ['native','euler']]
assert a['native_base_action_fingerprint']==b['native_base_action_fingerprint']
assert {k:v for k,v in a['action'].items() if k!='repair_sampler'}=={k:v for k,v in b['action'].items() if k!='repair_sampler'}
configs=[yaml.safe_load((pair/n/'config.yaml').read_text()) for n in ['native','euler']]
for c in configs:
    for k in ['data_path','output_folder','adapter_diagnostics_dir']:c.pop(k)
    c['repair_request'].pop('repair_sampler')
assert configs[0]==configs[1]
state['phase']='exporting_complete_three_way_video';write()
with (F/'review.log.txt').open('x') as log:
    subprocess.run([str(S/'venv/bin/python'),str(S/'scripts/review_native_sampler_pair.py'),'--root',str(F)],stdout=log,stderr=subprocess.STDOUT,check=True)
state.update(phase='completed',pair_inputs_match_except_sampler=True,queue_seconds_including_overlap_wait=time.monotonic()-began);write();print(json.dumps(state))
