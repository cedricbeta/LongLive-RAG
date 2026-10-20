#!/usr/bin/env python3
import hashlib,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForImageTextToText

root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_text());inputs_plan=json.loads((root/'inputs.json').read_text())
started=time.monotonic();torch.manual_seed(plan['seed'])
model=AutoModelForImageTextToText.from_pretrained(inputs_plan['snapshot'],trust_remote_code=True,local_files_only=True,
    dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
processor=AutoProcessor.from_pretrained(inputs_plan['snapshot'],trust_remote_code=True,local_files_only=True,use_fast=False)
loaded=time.monotonic();responses=[]
for case in inputs_plan['cases']:
    before=time.monotonic();row={'case':case['id'],'attempts':1}
    try:
        images=[Image.open(root/f['path']).convert('RGB') for f in case['frames']]
        content=[{'type':'text','text':plan['question']}]+[{'type':'image','image':image} for image in images]
        inputs=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,
            add_generation_prompt=True,return_tensors='pt',return_dict=True)
        row['input_shapes']={k:list(v.shape) for k,v in inputs.items()}
        inputs={k:v.to('cuda:0') for k,v in inputs.items()}
        with torch.inference_mode():
            output=model.generate(**inputs,do_sample=False,max_new_tokens=plan['max_new_tokens'])
        torch.cuda.synchronize();tokens=output[0,inputs['input_ids'].shape[1]:].tolist()
        row.update(status='completed',output_tokens=tokens,response=processor.tokenizer.decode(tokens,skip_special_tokens=True),
            input_tokens=inputs['input_ids'].shape[1],reached_token_ceiling=len(tokens)==plan['max_new_tokens'])
    except Exception as error:row.update(status='failed',error=type(error).__name__+': '+str(error))
    row['wall_seconds']=time.monotonic()-before;responses.append(row)
    (root/'responses.json').write_text(json.dumps(responses,indent=2)+'\n');print(case['id'],row['status'],flush=True)
result={'status':'completed' if all(r['status']=='completed' for r in responses) else 'failed',
    'requests':len(responses),'model_load_seconds':loaded-started,'worker_seconds':time.monotonic()-started,
    'peak_gpu_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'semantic_acceptance':False,
    'official_metric_runs':0,'candidate_selection_calls':0,'human_labels':0,'limits':plan['limits']}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='completed':raise SystemExit(1)
