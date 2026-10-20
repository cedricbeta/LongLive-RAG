from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
s=Path('/home/chendong/video-gen/LongLive-RAG');r=s/'videos/progressive_repair_20260917';local=s.parent/'ICLR27-Longlive-agent';remote=r/'remote_verified_checkout'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
git=lambda root,*args:subprocess.check_output(['git',*args],cwd=root,text=True).strip()
commit=git(local,'rev-parse','HEAD');assert git(remote,'rev-parse','HEAD')==commit
paths=subprocess.check_output(['git','ls-files','-z'],cwd=local).decode().split('\0');paths=[p for p in paths if p]
assert set(paths)==set(p for p in subprocess.check_output(['git','ls-files','-z'],cwd=remote).decode().split('\0') if p)
files={}
for p in paths:
 files[p]=sha(remote/p);assert files[p]==sha(local/p),p
assert not any('private_assignment_key' in p for p in paths)
exports=[]
for name in ['typed_script_compiler_v6_development','native_iris_names_v6','native_iris_names_v6_window']:
 folder=remote/'results'/name;m=json.loads((folder/'export_manifest.json').read_text())
 for p,h in m['files_sha256'].items():assert sha(folder/p)==h and str((folder/p).relative_to(remote)) in files,p
 manifest=json.loads((folder/'run/manifest.json').read_text())
 for p,h in manifest['source_sha256'].items():assert sha(folder/'run/source'/p)==h,p
 if 'plan_sha256' in manifest:assert sha(folder/'plan.json')==manifest['plan_sha256']
 exports.append({'run':name,'files':len(m['files_sha256']),'frozen_sources':len(manifest['source_sha256'])})
for path in ['results/name_candidate_contracts/development_summary.json','results/native_iris_names_v6/summary.json','results/native_iris_names_v6_window/summary.json']:
 d=json.loads((remote/path).read_text())
 for p,h in d['inputs_sha256'].items():assert sha(remote/p)==h,p
record=json.loads((remote/'results/name_candidate_contracts/exported_tests.json').read_text())
for p,h in record['source_sha256'].items():assert sha(remote/p)==h,p
paper=json.loads((remote/'paper/tables/provenance.json').read_text())
for p,h in paper.items():assert sha(remote/p)==h,p
folder=remote/'results/native_iris_names_v6_window';audit=json.loads((folder/'audit/audit.json').read_text());native_cells=0
for row in audit['masks']:
 path=folder/'audit'/row['array_file'];assert sha(path)==row['array_sha256'];a=np.load(path);meta=json.loads(path.with_name(path.stem+'_metadata.json').read_text());indices={int(x):i for i,x in enumerate(a['export_frame_indices'])}
 for i,mapping in enumerate(meta['source_frame_mapping']):
  ix=[indices[k] for k in mapping['export_frames']];mask=(a['frame_masks'][ix,0]>=.35).any(0);protect=(a['protected'][ix,0]>=.35).any(0);assert np.array_equal(a['native_mask'][i],mask & ~protect);native_cells+=1
manifest=json.loads((folder/'run/manifest.json').read_text());origin=Path(manifest['arguments']['output']).parent;images=set()
for path in (folder/'run/vlm_cache').rglob('*.json'):
 d=json.loads(path.read_text())
 for p,h in zip(d['ordered_image_paths'],d['ordered_image_hashes']):
  rel=Path(p).relative_to(origin);assert sha(folder/rel)==h;images.add(str(rel))
a=remote/'artifacts/native_iris_names_v6_window';binding=json.loads((a/'artifact_binding.json').read_text())
for p,h in binding['artifact_files_sha256'].items():assert sha(a/p)==h,p
assert sha(remote/binding['source_script'])==binding['source_script_sha256']
assert sha(a/'candidate.mp4')==json.loads((folder/'run/backend/candidate_0001_shot02_attempt1/audit.json').read_text())['video_sha256']
summary=json.loads((folder/'summary.json').read_text());assert summary['recorded_model_calls']==21 and summary['completed_native_videos']==1 and not summary['selected_parent_changed'] and not summary['future_shots_actually_checked'] and not summary['identity_references_promoted']
result={'repository':'https://github.com/cedricbeta/ICLR27-Longlive-agent','branch':'research/progressive-repair-20260917','commit':commit,'verified_unix':time.time(),'tracked_files_verified':len(files),'method':'Nonforce SSH push, independent remote fetch, all tracked file SHA256 compared to local delivery; export and frozen source hashes, real masks, actual request images, summaries and paper bindings checked.','exports':exports,'test_source_hashes':len(record['source_sha256']),'paper_inputs':len(paper),'native_mask_projection_slices_checked':native_cells,'actual_request_images_checked':len(images),'paper_sha256':sha(remote/'paper/build/main.pdf'),'private_human_key_absent':True,'submission_ready':False,'files_sha256':files}
for p in [r/f'remote_delivery_check_{commit[:7]}.json',r/'remote_delivery_check_latest.json']:p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}))
