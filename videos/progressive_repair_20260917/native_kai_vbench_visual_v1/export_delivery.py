#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--target',type=Path,required=True)
a=p.parse_args()
f=Path(__file__).resolve().parent
source=f.parents[2]
target=a.target.resolve()
result=target/'results'/f.name
artifact=target/'artifacts'/f.name
result.mkdir(parents=True,exist_ok=True)
files={}
paths=[]
refs={
'B_original/input/moviegen000-0.mp4':'artifacts/native_sampler_cases_v1/ns280_baseline.mp4',
'P20_prompt/input/moviegen000-0.mp4':'artifacts/native_kai_same_boundary_control_v1/P20_prompt.mp4',
'RP_local_cue/input/moviegen000-0.mp4':'artifacts/native_kai_retention_factorial_v1/RP_local_cue.mp4',
}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path in sorted(f.rglob('*')):
 if not path.is_file() or '__pycache__' in path.parts: continue
 relative=str(path.relative_to(f))
 if relative in refs:
  destination=target/refs[relative]
  assert sha(path)==sha(destination),relative
 else:
  destination=(artifact if path.suffix=='.mp4' else result)/relative
  destination.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(path,destination)
  paths.append(str(destination.relative_to(target)))
 files[relative]={'repository_path':str(destination.relative_to(target)),'sha256':sha(destination),'bytes':destination.stat().st_size}
for name in ['prepare_kai_vbench_comparison.py','audit_vbench_custom_video.py','summarize_kai_vbench.py','verify_kai_vbench_export.py','prepare_kai_vbench_export.py']:
 destination=target/'implementation/scripts'/name
 shutil.copyfile(source/'scripts'/name,destination)
 relative=str(destination.relative_to(target))
 paths.append(relative)
 files['implementation/'+name]={'repository_path':relative,'sha256':sha(destination),'bytes':destination.stat().st_size}
manifest={'runtime_root':str(f),'files':files,'original_videos_reused':refs,
 'scope':'One exposed ns280 seed and three frozen conditions; six official custom visual metrics; no entity acceptance or human labels.',
 'source_snapshot_manifest':'reproduction/vbench_long/source_manifest.json',
 'model_manifest':'results/official_vbench_provision_v1/model_assets_manifest.json'}
(result/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths.append(str((result/'export_manifest.json').relative_to(target)))
(f.parent/'native_kai_vbench_export_paths.json').write_text(json.dumps(sorted(paths),indent=2)+'\n')
print(json.dumps({'exported_paths':len(paths),'manifest_files':len(files),'bytes':sum(x['bytes'] for x in files.values())}))
