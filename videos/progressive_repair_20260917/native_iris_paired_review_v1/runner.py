#!/usr/bin/env python3
"""Freeze bounded, order-swapped paired reviews of an existing development replay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import time

LABELS = ('pass', 'fail', 'unknown')
DOMAINS = ('other_entities', 'required_actions', 'scene', 'visual_quality')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def support(path):
    spec = importlib.util.spec_from_file_location('paired_runtime_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(value, frames):
    if not isinstance(value, dict) or set(value) != {'appearance_A', 'appearance_B', 'identity', *DOMAINS}:
        raise ValueError('Incorrect top-level fields')
    for slot in ('A', 'B'):
        rows = value['appearance_' + slot]
        if not isinstance(rows, list) or len(rows) != len(frames):
            raise ValueError('Every supplied frame requires an appearance row')
        for row, frame in zip(rows, frames):
            if not isinstance(row, dict) or set(row) != {'frame', 'clothing', 'garment_color', 'garment_type', 'hair_color', 'hair_style'}:
                raise ValueError('Incorrect appearance fields')
            if type(row['frame']) is not int or row['frame'] != frame:
                raise ValueError('Frame order or identity changed')
            if not isinstance(row['clothing'], str) or not 1 <= len(row['clothing']) <= 300:
                raise ValueError('Missing or excessive visible clothing description')
            if any(row[k] not in LABELS for k in ('garment_color', 'garment_type', 'hair_color', 'hair_style')):
                raise ValueError('Invalid appearance status')
    for name in ('identity', *DOMAINS):
        row = value[name]
        field = 'status' if name == 'identity' else 'worse_slot'
        allowed = ('same', 'different', 'unknown') if name == 'identity' else ('A', 'B', 'neither', 'unknown')
        if not isinstance(row, dict) or set(row) != {field, 'observation'}:
            raise ValueError('Incorrect comparison fields')
        if row[field] not in allowed or not isinstance(row['observation'], str) or not 1 <= len(row['observation']) <= 450:
            raise ValueError('Invalid comparison response')
    return value


def prompt(compiled, shot, frames):
    entity, = compiled['entities']
    appearance = [r for r in entity['appearance'] if r['source']['shot'] <= shot <= r['last_active_shot']]
    expected = {}
    for part, aspect in [('garment', 'color'), ('garment', 'type'), ('hair', 'color'), ('hair', 'style')]:
        row, = [r for r in appearance if r['part'] == part and r['aspect'] == aspect]
        assert row['source']['quote'] in compiled['prompts'][row['source']['shot']]
        expected[f'{part}_{aspect}'] = row['expected']
    packet = {
        'task': 'Compare two anonymous versions of the same sampled time interval. Neither version is asserted correct. Do not infer which system produced a slot.',
        'fixed_authored_current_prompt': compiled['prompts'][shot],
        'entity': entity['name'], 'fixed_appearance_requirements': expected,
        'intended_target_fields': ['garment_color', 'garment_type'],
        'non_target_appearance_fields': ['hair_color', 'hair_style'],
        'frame_indices_in_each_slot': frames,
        'instructions': [
            'Images are labelled by slot A or B and frame number. Within each slot they are in chronological order at 16 original frames per second. Compare aligned times.',
            'For every supplied image, first describe the visible clothing, then label each fixed appearance requirement pass, fail or unknown. Pass means visibly consistent, fail means visible contradiction, unknown means insufficient or ambiguous evidence. Missing visual evidence is never pass. Do not copy the expected garment name into the description unless actually visible.',
            'Clothing color and type are separate requirements. An apron, shirt or uncertain outer layer must not be called the specified coat merely because its color agrees. Do not require a hidden length or other detail not supplied by the authored requirement.',
            'Identity: same only if visible facial/body cues support correspondence; different only if those cues show a different person; otherwise unknown. Intended changes to garment color/type alone are not identity changes. Hair is assessed separately.',
            'For other_entities, required_actions, scene and visual_quality, report which slot has visibly worse content: A, B, neither, or unknown. Neither requires sufficient evidence of no visible difference in that domain. Unknown is required when evidence is insufficient. Existing defects shared by both are not a new comparative defect.',
            'Other_entities concerns required/extra/lost people. Required_actions concerns the authored action in the shown sequence; sparse images do not certify motion between samples. Scene includes relevant props, layout and relationships. Visual_quality concerns visible distortion, duplication, seams, blur or artifacts.',
            'Do not balance better clothing against worse identity, action or quality. Do not return a total score, preference or acceptance decision.',
            'Return only strict JSON with exactly appearance_A, appearance_B, identity, other_entities, required_actions, scene, visual_quality. Each appearance list has one row per supplied frame, in the stated order: frame (integer), clothing (short visible description), garment_color, garment_type, hair_color, hair_style (each pass/fail/unknown). Identity has status (same/different/unknown) and observation (short string). Each other comparison has worse_slot (A/B/neither/unknown) and observation (short string). Keep descriptions factual and short.']}
    return json.dumps(packet, ensure_ascii=False, sort_keys=True)


def prepare(args):
    import cv2
    from PIL import Image
    root, folder = args.root.resolve(), args.folder.resolve()
    assert args.gpu in (0, 1, 2)
    prior = root / 'results/native_iris_molmo_observations_v1'
    previous = read(prior / 'plan.json')
    compiled_path = root / 'results/native_iris_natural_expression_v1/compiled_script.json'
    compiled = read(compiled_path)
    for name, row in read(prior / 'model_manifest.json')['files'].items():
        assert sha(args.model_path / name) == row['sha256'], name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'inputs').mkdir()
    for name in ('model_manifest.json',):
        shutil.copy2(prior / name, folder / name)
    shutil.copytree(prior / 'model_source', folder / 'model_source')
    shutil.copy2(compiled_path, folder / 'compiled_script.json')
    shutil.copy2(Path(__file__), folder / 'runner.py')
    shutil.copy2(Path(__file__).with_name('probe_native_observations.py'), folder / 'support.py')
    videos = {'original': 'artifacts/official_iris_60s_baseline_prepared/videos/rank0-0-0_lora.mp4',
        'fixed_second': 'artifacts/native_iris_fixed_noise_v1/candidate.mp4',
        'natural_expression': 'artifacts/native_iris_natural_expression_v1/candidate.mp4'}
    windows = {'current': {'shot': 2, 'frames': [320, 352, 400, 448, 479]},
               'future': {'shot': 4, 'frames': [672, 720, 768]}}
    images, sources = {}, []
    for branch, relative in videos.items():
        video = root / relative
        sources.append({'branch': branch, 'repository_path': relative, 'sha256': sha(video)})
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 960 and cap.get(cv2.CAP_PROP_FPS) == 16
        for window in windows.values():
            for frame in window['frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                assert ok and image.shape == (480, 832, 3)
                path = folder / 'inputs' / f'{branch}_{frame}.png'
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
                images[branch, frame] = {'image': str(path.relative_to(folder)), 'sha256': sha(path)}
        cap.release()
    gray = folder / 'inputs/gray.png'
    Image.new('RGB', (832, 480), (127, 127, 127)).save(gray)
    for window in windows.values():
        for frame in window['frames']:
            images['gray', frame] = {'image': 'inputs/gray.png', 'sha256': sha(gray)}
    cases = []
    def add(left, right, scope, control, group):
        w = windows[scope]
        packets = [{'slot': slot, 'frame': frame, **images[branch, frame]}
                   for slot, branch in [('A', left), ('B', right)] for frame in w['frames']]
        cases.append({'branches': {'A': left, 'B': right}, 'scope': scope, 'frames': w['frames'],
            'control': control, 'pair_group': group, 'images': packets,
            'image': packets[0]['image'], 'image_sha256': packets[0]['sha256'],
            'prompt': prompt(compiled, w['shot'], w['frames'])})
    for comparator in ('original', 'fixed_second'):
        for scope in windows:
            group = comparator + '_vs_natural_' + scope
            add(comparator, 'natural_expression', scope, None, group)
            add('natural_expression', comparator, scope, None, group)
    add('original', 'original', 'current', 'same_sequence', 'original_self_current')
    add('natural_expression', 'natural_expression', 'current', 'same_sequence', 'natural_self_current')
    add('natural_expression', 'natural_expression', 'future', 'same_sequence', 'natural_self_future')
    add('gray', 'gray', 'current', 'no_visual_evidence', 'gray_self_current')
    random.Random(17345).shuffle(cases)
    for index, case in enumerate(cases):
        case['id'] = f'pair_{index:03d}'
    plan = {k: previous[k] for k in ('model', 'model_revision', 'package_versions', 'dtype', 'attention')}
    plan.update(protocol='native_iris_paired_molmo_review_v1', frozen_unix=time.time(),
        model_path=str(args.model_path.resolve()), python=str(args.python.absolute()),
        source_sha256=sha(folder / 'runner.py'), support_source_sha256=sha(folder / 'support.py'),
        model_manifest_sha256=sha(folder / 'model_manifest.json'), compiled_script_sha256=sha(folder / 'compiled_script.json'),
        videos=sources, windows=windows, cases=cases, gpu=args.gpu, timeout_seconds=900, max_calls=12,
        generation={'max_new_tokens': 2048, 'do_sample': False}, seed_per_call=0,
        grammar_constrained_decoding=False, held_out=False, actual_human_labels=0,
        reference_observation_plan_sha256=sha(prior / 'plan.json'),
        analysis='Normalize A/B outputs to branch identities. Report every per-frame label and protection outcome, exact same-sequence controls and order-swap disagreements. Do not majority-vote conflicting orders, collapse unknown to pass, or treat correlated frames as independent samples.',
        adoption_rule='No automatic adoption from this diagnostic. Any order disagreement, new observed defect or unresolved identity/action evidence remains a limitation. This is a development assessment with targets supplied, not the earlier target-blind task or a calibrated independent final evaluator.',
        failure_policy='One bounded run, no parse or model retries. Stop after three consecutive malformed replies or any execution failure. All attempts and unattempted case IDs are retained; never rerun or change the frozen prompt after inspecting outputs.',
        scope='Two archived comparators versus one already exposed candidate, two windows, both slot orders, three exact-sequence controls and one gray control. Full-sized sampled frames only; no full-motion or whole-suffix guarantee, no new video generation.')
    write(folder / 'plan.json', plan)
    write(folder / 'manifest.json', {'gpu': args.gpu, 'protocol': plan['protocol']})
    print(json.dumps({'folder': str(folder), 'cases': len(cases), 'plan_sha256': sha(folder / 'plan.json')}))


def worker(folder):
    started = time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    plan = read(folder / 'plan.json')
    assert sha(folder / 'runner.py') == plan['source_sha256']
    assert sha(folder / 'support.py') == plan['support_source_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(plan['model_path'], trust_remote_code=True, local_files_only=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(plan['model_path'], trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    model.eval()
    torch.cuda.synchronize()
    loaded = time.monotonic()
    (folder / 'transcripts').mkdir()
    rows, malformed, stop = [], 0, 'all_cases_attempted'
    for case in plan['cases']:
        tick = time.monotonic()
        row = {'id': case['id'], 'prompt': case['prompt'], 'images': case['images'],
               'generation': plan['generation'], 'seed': plan['seed_per_call']}
        try:
            content = [{'type': 'text', 'text': case['prompt']}]
            for item in case['images']:
                path = folder / item['image']
                assert sha(path) == item['sha256']
                with Image.open(path) as im:
                    image = im.convert('RGB')
                content += [{'type': 'text', 'text': f"Slot {item['slot']}, frame {item['frame']}"},
                            {'type': 'image', 'image': image}]
            inputs = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
                add_generation_prompt=True, return_tensors='pt', return_dict=True)
            row['input_tensors'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                for k, v in inputs.items()}
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            torch.manual_seed(plan['seed_per_call'])
            row['model_call_started'] = True
            with torch.inference_mode():
                ids = model.generate(**inputs, **plan['generation'])
            tokens = ids[0, inputs['input_ids'].size(1):]
            row['output_token_ids'] = tokens.tolist()
            row['reply'] = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            torch.cuda.synchronize()
            try:
                row['parsed'] = validate(json.loads(row['reply']), case['frames'])
                row['status'] = 'completed'
                malformed = 0
            except Exception as error:
                row.update(status='parse_failed', error=f'{type(error).__name__}: {error}')
                malformed += 1
        except Exception as error:
            row.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
        row['wall_seconds'] = time.monotonic() - tick
        write(folder / 'transcripts' / (case['id'] + '.json'), row)
        rows.append(row)
        write(folder / 'partial_results.json', rows)
        print(json.dumps({'id': case['id'], 'status': row['status'], 'seconds': row['wall_seconds']}), flush=True)
        if row['status'] == 'execution_failed' or malformed >= 3:
            stop = 'execution_failure' if row['status'] == 'execution_failed' else 'three_consecutive_malformed_replies'
            break
    write(folder / 'result.json', {'rows': rows, 'model_load_seconds': loaded - started,
        'worker_seconds': time.monotonic() - started, 'actual_calls': sum(r.get('model_call_started', False) for r in rows),
        'completed': sum(r['status'] == 'completed' for r in rows), 'stop_reason': stop,
        'unattempted_case_ids': [c['id'] for c in plan['cases'][len(rows):]],
        'actual_human_labels': 0, 'new_generated_videos': 0, 'candidate_adopted': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--root', type=Path)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--python', type=Path)
    p.add_argument('--gpu', type=int, default=1)
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    elif args.worker:
        worker(args.folder.resolve())
    else:
        folder = args.folder.resolve()
        plan = read(folder / 'plan.json')
        assert sha(folder / 'support.py') == plan['support_source_sha256']
        support(folder / 'support.py').supervise(folder)
