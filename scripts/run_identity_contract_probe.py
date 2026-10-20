#!/usr/bin/env python3
"""Bounded development controls with the output contract visible to the model.

No candidate selection or repair acceptance. The native Molmo backend is a
different-family diagnostic, not an independent final efficacy evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.agent_identity_comparison import compare, EVIDENCE_SCHEMA
from utils.agent_runtime import file_sha256
from utils.agent_vlm import FingerprintedVLMClient, _validate_json

CONTRACT = (
    '\nReturn only a JSON object matching the following output contract. '
    'actor_correspondence asks whether the intended subject in EACH image slot '
    'is unambiguous, not whether the people in A and B are the same. '
    'Use one_unambiguous_subject_per_slot when each slot clearly depicts one '
    'intended subject, including when those subjects are different people. '
    'Use ambiguous when multiple possible subjects prevent assigning the requested '
    'facial observations to one subject. face_visibility describes whether the '
    'actual requested facial structures can be seen: clear, partial, or unobservable. '
    'Observe each slot separately before comparing. Missing facial detail must '
    'remain unobservable; do not invent a matching feature. Schema: '
    + json.dumps(EVIDENCE_SCHEMA, sort_keys=True)
)


class ExplicitContract:
    def __init__(self, client):
        self.client = client

    def call(self, **kwargs):
        kwargs['prompt'] += CONTRACT
        return self.client.call(**kwargs)


class NativeMolmo:
    def __init__(self, model_path, folder):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch, self.folder = torch, folder
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True,
                                                      local_files_only=True, use_fast=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
        self.model.eval()

    def call(self, *, prompt, image_paths, schema, **unused):
        from PIL import Image
        torch = self.torch
        images = [Image.open(p).convert('RGB') for p in image_paths]
        content = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image', 'image': image} for image in images]
        request = {'prompt': prompt, 'schema': schema,
                   'images': [{'path': str(p), 'sha256': file_sha256(p)} for p in image_paths],
                   'generation': {'max_new_tokens': 2048, 'do_sample': False},
                   'grammar_constrained_decoding': False, 'dtype': 'bfloat16'}
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.folder / 'transcripts' / (digest + '.json')
        path.parent.mkdir(exist_ok=True)
        started = time.monotonic()
        inputs = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt', return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        request['input_shapes'] = {k: list(v.shape) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        tokens = ids[0, inputs['input_ids'].size(1):]
        reply = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        torch.cuda.synchronize()
        wall = time.monotonic() - started
        record = {'request': request, 'reply': reply, 'output_tokens': len(tokens),
                  'wall_seconds': wall, 'call_id': digest}
        try:
            parsed = json.loads(reply)
            _validate_json(parsed, schema)
            record['parsed'] = parsed
        except Exception as error:
            record['parse_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            path.write_text(json.dumps(record, indent=2) + '\n')
        return {'parsed': parsed, 'call_id': digest, 'transcript': str(path),
                'wall_seconds': wall, 'cached': False}


def run(folder, base_url=None):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (folder / 'status.json').exists():
        raise FileExistsError('Existing attempt must not be silently repeated')
    for rel, digest in manifest['input_sha256'].items():
        if file_sha256(folder / rel) != digest:
            raise ValueError('Frozen image mismatch: ' + rel)
    source = folder / 'source'
    frozen = {}
    for rel in ['scripts/run_identity_contract_probe.py', 'utils/agent_identity_comparison.py',
                'utils/agent_runtime.py', 'utils/agent_vlm.py', 'utils/agent_edit.py']:
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, path)
        frozen[rel] = file_sha256(path)
    (folder / 'source_manifest.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (folder / 'explicit_contract.txt').write_text(CONTRACT)
    status = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time()}
    (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    started, rows = time.monotonic(), []
    try:
        if manifest['backend'] == 'qwen_sglang':
            client = FingerprintedVLMClient(base_url=base_url, model=manifest['model'],
                cache_dir=folder / 'vlm_cache', max_tokens=2048, seed=0, max_parse_retries=0)
        elif manifest['backend'] == 'molmo_native':
            client = NativeMolmo(manifest['model_path'], folder)
        else:
            raise ValueError('Unknown backend')
        ready = time.monotonic()
        client = ExplicitContract(client)
        for case in manifest['cases']:
            expected = case['expected_control_status']
            row = {'id': case['id'], 'expected_control_status': expected}
            try:
                result = compare(client, case['frames_a'], case['frames_b'])
                row.update(result)
                row['execution_status'] = 'completed'
            except Exception as error:
                row.update(execution_status='execution_failed', error=f'{type(error).__name__}: {error}',
                           decision={'status': 'unknown', 'reason': 'Execution or parsing failed'})
            row['control_matches_expectation'] = None if expected is None else (
                row['execution_status'] == 'completed' and row['decision']['status'] == expected)
            rows.append(row)
            (folder / 'partial_results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(case['id'], row['execution_status'], row['decision'], flush=True)
        result = {'phase': 'development_identity_explicit_contract', 'rows': rows,
                  'constructed_controls': sum(r['expected_control_status'] is not None for r in rows),
                  'controls_matching_expectation': sum(r['control_matches_expectation'] is True for r in rows),
                  'model_load_seconds': ready - started, 'wall_seconds': time.monotonic() - started,
                  'integrated_into_repair_acceptance': False, 'live_branch_promoted': False,
                  'independent_final_evaluation': False, 'limitations': manifest['limits']}
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(status='completed', wall_seconds=time.monotonic() - started)
    except Exception as error:
        status.update(status='execution_failed', wall_seconds=time.monotonic() - started,
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (folder / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--base-url')
    args = parser.parse_args()
    run(args.folder.resolve(), args.base_url)
