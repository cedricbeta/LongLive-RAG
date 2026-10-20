import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.run_identity_contract_probe import NativeMolmo
from PIL import Image
import torch

folder = Path(__file__).resolve().parent
manifest = json.loads((folder / 'manifest.json').read_text())
assert not (folder / 'status.json').exists()
uuid = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert not any(line.split(',')[0].strip() == uuid for line in apps.splitlines())
for path, digest in manifest['input_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
started = time.monotonic()
status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time()}
(folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
rows = []
try:
    client = NativeMolmo(manifest['model_path'], folder)
    ready = time.monotonic()
    for case in manifest['cases']:
        paths = case['frames_a'] + case['frames_b']
        content = [{'type': 'text', 'text': manifest['prompt']}] + [
            {'type': 'image', 'image': Image.open(path).convert('RGB')} for path in paths]
        call_start = time.monotonic()
        inputs = client.processor.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
        inputs = {key: value.to(client.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            ids = client.model.generate(**inputs, max_new_tokens=384, do_sample=False)
        tokens = ids[0, inputs['input_ids'].size(1):]
        text = client.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        torch.cuda.synchronize()
        rows.append({'case': case['id'], 'images': paths, 'prompt': manifest['prompt'],
                     'raw_reply': text, 'output_tokens': len(tokens), 'wall_seconds': time.monotonic() - call_start})
        (folder / 'partial_results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(case['id'], text, flush=True)
    (folder / 'result.json').write_text(json.dumps({'rows': rows, 'model_load_seconds': ready - started,
        'wall_seconds': time.monotonic() - started, 'acceptance_rule_defined': False,
        'independent_final_evaluation': False, 'branch_promoted': False}, indent=2) + '\n')
    status.update(status='completed', wall_seconds=time.monotonic() - started)
except Exception as error:
    status.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
