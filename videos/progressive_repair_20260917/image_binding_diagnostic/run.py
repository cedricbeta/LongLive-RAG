from pathlib import Path
import json,sys,time
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parents[2]))
from utils.agent_vlm import FingerprintedVLMClient
from utils.agent_runtime import TemporalInput,file_sha256
m=json.loads((root/'manifest.json').read_text());client=FingerprintedVLMClient(base_url=sys.argv[1],model=m['model'],cache_dir=root/'vlm_cache',max_tokens=768,max_parse_retries=0)
rows=[];start=time.monotonic()
for case in m['cases']:
 schema={'type':'object','additionalProperties':False,'properties':{f'image_{i+1}':{'type':'string','minLength':1,'maxLength':450} for i in range(len(case['images']))},'required':[f'image_{i+1}' for i in range(len(case['images']))]}
 reply=client.call(prompt='Describe each supplied image independently in its numbered field, in order. Describe the visible person, face, hair and clothes. Do not compare identity and do not assume any two images are the same. Report only what is visible.',image_paths=case['images'],schema=schema,call_kind='image_binding_diagnostic_v1',shot_index=0,temporal_inputs=tuple(TemporalInput(kind='diagnostic_crop',identifier=p,sha256=file_sha256(p),available_after_shot=0) for p in case['images']))
 rows.append({'id':case['id'],'input_sha256':[file_sha256(p) for p in case['images']],'response':reply});(root/'partial_results.json').write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],reply['parsed'],flush=True)
(root/'result.json').write_text(json.dumps({'rows':rows,'wall_seconds':time.monotonic()-start,'acceptance_rule_changed':False},indent=2)+'\n')
