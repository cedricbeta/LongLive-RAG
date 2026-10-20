from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys,tempfile

def verify(root):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 read=lambda p:json.loads(p.read_text())
 def module(p,name):
  spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 folders=['native_iris_molmo_observations_v1','native_iris_molmo_observations_v1_telemetry','native_iris_natural_expression_observations_v1','native_iris_natural_expression_observations_v1_telemetry','native_iris_natural_expression_v1','native_iris_natural_expression_v1_telemetry']
 counts={}
 for name in folders:
  f=root/'results'/name;manifest=read(f/'export_manifest.json')['files_sha256']
  actual={str(p.relative_to(root)) for p in f.rglob('*') if p.is_file() and p.name!='export_manifest.json' and '__pycache__' not in p.parts}
  assert actual==set(manifest),(name,'incomplete export')
  for path,expected in manifest.items():assert sha(root/path)==expected,path
  counts[name]=len(manifest)
 for name,field in [('native_iris_molmo_observations_v1','result_sha256'),('native_iris_natural_expression_v1','outputs_sha256')]:
  provenance=read(root/'results'/name/'provenance.json')
  for key in ('source_sha256',field):
   for path,expected in provenance[key].items():assert sha(root/path)==expected,path
 records={};idhashes=set()
 with tempfile.TemporaryDirectory() as temp:
  tmp=Path(temp)
  for name in ['native_iris_molmo_observations_v1','native_iris_natural_expression_observations_v1']:
   f=root/'results'/name;out=tmp/(name+'.json')
   subprocess.run([sys.executable,str(root/'implementation/scripts/audit_native_observations.py'),'--root',str(root),'--folder',str(f),'--telemetry',str(root/'results'/(name+'_telemetry')),'--output',str(out)],check=True,capture_output=True)
   assert out.read_bytes()==(f/'audit.json').read_bytes(),name
   data=read(out);records[name]={k:data[k] for k in ('actual_calls','completed','actual_frames_reproduced','supervisor_seconds','controls')}
   for row in read(f/'partial_results.json'):idhashes.add(row['input_tensors']['input_ids']['sha256'])
  assert len(idhashes)==1
  old='native_iris_molmo_observations_v1'
  oldmod=module(root/'implementation/scripts/summarize_native_observations.py','oldsum')
  assert json.dumps(oldmod.summarize(root/'results'/old/'audit.json'),indent=2)+'\n' == (root/'results'/old/'description_inventory.json').read_text()
  # Preserve the same relative layout without modifying the independent checkout.
  (tmp/'results').mkdir();(tmp/'results'/old).symlink_to(root/'results'/old,target_is_directory=True)
  gallery=tmp/'artifacts'/old/'index.html';gallery.parent.mkdir(parents=True)
  oldmod.gallery(tmp/'results'/old,gallery)
  assert sha(gallery)==sha(root/'artifacts'/old/'index.html')
  new='native_iris_natural_expression_v1'
  newmod=module(root/'implementation/scripts/summarize_native_expression_control.py','newsum')
  summary=newmod.summarize(root)
  assert json.dumps(summary,indent=2)+'\n' == (root/'results'/new/'summary.json').read_text()
  gallery=tmp/'natural.html';newmod.gallery(root,summary,gallery)
  assert sha(gallery)==sha(root/'artifacts'/new/'index.html')
  out=tmp/'expression.json'
  subprocess.run([sys.executable,str(root/'implementation/scripts/verify_native_expression_control.py'),'--root',str(root),'--output',str(out)],check=True,capture_output=True)
  assert out.read_bytes()==(root/'results'/new/'export_audit.json').read_bytes()
  expression=read(out)
  # Re-encode the complete display using its published script and compare actual bytes.
  out=tmp/'four_way_comparison.mp4'
  subprocess.run([sys.executable,str(root/'implementation/scripts/plot_native_expression_control.py'),'--root',str(root),'--output',str(out)],check=True,capture_output=True)
  assert sha(out)==sha(root/'artifacts'/new/out.name)
  assert sha(out.with_suffix('.json'))==sha(root/'artifacts'/new/'four_way_comparison.json')
 return {'export_files':counts,'observations':records,'all_45_actual_tokenized_text_inputs_identical':True,'audits_summaries_and_galleries_byte_exact':True,'full_60_second_comparison_reencoded_byte_exact':True,'natural_expression_verification':expression,'new_model_calls':0,'new_generated_videos':0,'quality_claim':False}

if __name__=='__main__':
 root=Path(sys.argv[1]).resolve();result=verify(root)
 Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
