#!/usr/bin/env python3
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);a=p.parse_args()
f=Path(__file__).resolve().parent;source=f.parents[2];target=a.target.resolve();files={};paths=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path in sorted(f.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pt':continue
    rel=path.relative_to(f)
    dest=target/('artifacts' if path.suffix in ('.mp4','.png') else 'results')/f.name/rel
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    name=str(dest.relative_to(target));paths.append(name)
    files[str(rel)]={'repository_path':name,'sha256':sha(dest),'bytes':dest.stat().st_size}
for name in ['prepare_native_development_pilot.py']:
    dest=target/'implementation/scripts'/name;shutil.copyfile(source/'scripts'/name,dest)
    rel=str(dest.relative_to(target));paths.append(rel)
    files['implementation/'+name]={'repository_path':rel,'sha256':sha(dest),'bytes':dest.stat().st_size}
manifest={'runtime_root':str(f),'files':files,'raw_latents_not_in_git':True,
    'latent_provenance':'Actual tensors were checked at runtime; hashes and dimensions are retained. Reproduce generation to recover tensors.',
    'scope':'First two original videos from twelve preselected exposed scripts. No repair/strategy comparison or quality acceptance.'}
dest=target/'results'/f.name/'export_manifest.json';dest.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(str(dest.relative_to(target)))
(f.parent/'native_development_pilot12_export_paths.json').write_text(json.dumps(sorted(paths),indent=2)+'\n')
print(json.dumps({'files':len(paths),'manifest_files':len(files),'bytes':sum(v['bytes'] for v in files.values())}))
