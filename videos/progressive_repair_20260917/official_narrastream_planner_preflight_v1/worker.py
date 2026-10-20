#!/usr/bin/env python3
import hashlib,json,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_text())
sys.path.insert(0,plan['official_checkout'])
from narrastream_bench.utils.entity_extraction import _extractor_prompt,_extract_json,_normalize_payload
started=time.monotonic();torch.manual_seed(0)
model=Qwen3_5ForConditionalGeneration.from_pretrained(plan['checkpoint'],local_files_only=True,
    dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
tokenizer=AutoTokenizer.from_pretrained(plan['checkpoint'],local_files_only=True)
loaded=time.monotonic();rows=[]
for case in plan['cases']:
    text=_extractor_prompt(case['prompt'],max_entities=4,max_attributes_per_entity=5)
    messages=[{'role':'user','content':text}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(rendered,return_tensors='pt').to('cuda:0')
    before=time.monotonic()
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=256)
    torch.cuda.synchronize()
    tokens=output[0,inputs['input_ids'].shape[1]:].tolist()
    response=tokenizer.decode(tokens,skip_special_tokens=True)
    row={'case':case['id'],'shot_index':case['shot_index'],'official_user_message':text,
        'rendered_input_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'response':response,
        'output_tokens':tokens,'input_tokens':inputs['input_ids'].shape[1],'wall_seconds':time.monotonic()-before}
    try:
        row['normalized_payload']=_normalize_payload(_extract_json(response),max_entities=4,max_attributes_per_entity=5)
        row['official_parser_accepted']=row['normalized_payload'] is not None
    except Exception as error:
        row.update(official_parser_accepted=False,parse_error=type(error).__name__+': '+str(error))
    rows.append(row)
    (root/'responses.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'case':case['id'],'tokens':len(tokens),'parser_accepted':row['official_parser_accepted']}),flush=True)
result={'phase':'local_public_planner_two_request_preflight','model_load_seconds':loaded-started,
    'worker_seconds':time.monotonic()-started,'peak_gpu_memory_allocated_bytes':torch.cuda.max_memory_allocated(),
    'requests':len(rows),'parser_accepted':sum(r['official_parser_accepted'] for r in rows),'video_scores':0,
    'official_metric_run':False,'limits':['Official extractor prompt and parser; local greedy BF16 inference, not the original remote provider.',
    'The official API requests JSON response_format; this preflight uses unconstrained local decoding. It is not a complete official scorer reproduction.',
    'Thinking is disabled exactly as requested by the official entity extraction helper. No changed prompt, retry or response repair.',
    'Syntactic extraction success does not establish complete or correct semantic labels.']}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
