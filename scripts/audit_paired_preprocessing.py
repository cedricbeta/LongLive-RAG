#!/usr/bin/env python3
"""Replay actual paired image preprocessing on CPU, without loading model weights."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    folders = [args.root.resolve() / 'results' / name for name in
               ('native_iris_paired_review_v1', 'native_iris_short_pair_descriptions_v1')]
    read = lambda p: json.loads(p.read_text())
    model = read(folders[0] / 'model_manifest.json')
    files = {name: row['sha256'] for name, row in model['files'].items()
             if not name.endswith('.safetensors')}
    for name, expected in files.items():
        assert sha(args.model_path / name) == expected, name
    source = output / 'runner.py'
    shutil.copy2(Path(__file__), source)
    plan = {'protocol': 'actual_paired_preprocessing_replay_v1', 'frozen_unix': time.time(),
        'model_path': str(args.model_path.resolve()), 'nonweight_model_files_sha256': files,
        'source_sha256': sha(source), 'arguments': {k: str(v) for k, v in vars(args).items()},
        'package_versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'Pillow', 'numpy')},
        'result_inputs_sha256': {str(p.relative_to(args.root.resolve())): sha(p)
            for folder in folders for p in (folder / 'plan.json', folder / 'partial_results.json')},
        'scope': 'CPU replay of all 39 actual requests using the pinned processor. Exact tensor comparisons and same-image crop controls; zero language-model calls and no semantic accuracy claim.'}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True,
        local_files_only=True, use_fast=False)
    rows = []
    for folder in folders:
        for original in read(folder / 'partial_results.json'):
            content = [{'type': 'text', 'text': original['prompt']}]
            image_hashes = []
            for item in original['images']:
                path = folder / item['image']
                assert sha(path) == item['sha256']
                image_hashes.append(item['sha256'])
                with Image.open(path) as im:
                    image = im.convert('RGB')
                content += [{'type': 'text', 'text': f"Slot {item['slot']}, frame {item['frame']}"},
                            {'type': 'image', 'image': image}]
            tensors = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
                add_generation_prompt=True, return_tensors='pt', return_dict=True)
            actual = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                for k, v in tensors.items()}
            assert actual == original['input_tensors'], (folder.name, original['id'], 'tensor mismatch')
            record = {'run': folder.name, 'id': original['id'], 'actual_input_tensors_reproduced_exactly': True,
                'tensors': actual, 'png_sha256': image_hashes}
            # The pinned processor records crop counts for each input image.
            if len(image_hashes) == 2 and len(set(image_hashes)) == 1:
                images = tensors['pixel_values']
                a, b = tensors['image_num_crops'].tolist()
                assert a == b and images.shape[0] == a + b
                record['aligned_normalized_crop_stacks_exact'] = torch.equal(images[:a], images[a:])
                assert record['aligned_normalized_crop_stacks_exact']
            rows.append(record)
            (output / 'partial_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    result = {'status': 'completed', 'requests_reproduced': len(rows), 'rows': rows,
        'source_sha256': sha(source), 'plan_sha256': sha(output / 'plan.json'),
        'wall_seconds': time.monotonic() - started, 'actual_language_model_calls': 0,
        'model_weights_loaded': False, 'device': 'cpu', 'scope': plan['scope']}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'model-path', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
