#!/usr/bin/env python3
"""Bounded two-image clothing descriptions after an unsuccessful joint review."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import time


PROMPT = ('Compare only the visible clothing of the central person in the two images labelled Slot A and Slot B. '
          'Describe clothing in A and B separately, including type, color, sleeves, layers and fastenings when visible. '
          'Then state the visible differences. The images may show the same clothing; do not assume there must be an edit. '
          'Do not infer a desired outfit, identity or earlier images. If no person is visible, say so. '
          'Return only a JSON object with exactly three string fields: clothing_A, clothing_B, visible_differences. '
          'Keep the combined descriptions under 100 words.')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(value, frames):
    if not isinstance(value, dict) or set(value) != {'clothing_A', 'clothing_B', 'visible_differences'}:
        raise ValueError('Expected exactly three description fields')
    if any(not isinstance(x, str) or not 1 <= len(x) <= 1000 for x in value.values()):
        raise ValueError('Expected three nonempty short strings')
    return value


def prepare(args):
    source, folder = args.reference.resolve(), args.folder.resolve()
    previous = json.loads((source / 'plan.json').read_text())
    result = json.loads((source / 'result.json').read_text())
    assert result['stop_reason'] == 'three_consecutive_malformed_replies'
    assert result['actual_calls'] == 3 and result['completed'] == 0
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'model_source'):
        shutil.copytree(source / name, folder / name)
    for name in ('model_manifest.json', 'compiled_script.json', 'support.py'):
        shutil.copy2(source / name, folder / name)
    shutil.copy2(source / 'plan.json', folder / 'reference_plan.json')
    shutil.copy2(source / 'result.json', folder / 'reference_result.json')
    shutil.copy2(Path(__file__), folder / 'runner.py')
    shutil.copy2(Path(__file__).with_name('probe_native_paired_review.py'), folder / 'paired_engine.py')
    cases = []
    def add(a, b, frame, scope, control, group):
        images = []
        for slot, branch in [('A', a), ('B', b)]:
            name = 'gray.png' if branch == 'gray' else f'{branch}_{frame}.png'
            path = folder / 'inputs' / name
            images.append({'slot': slot, 'frame': frame, 'image': 'inputs/' + name, 'sha256': sha(path)})
        cases.append({'branches': {'A': a, 'B': b}, 'frames': [frame], 'scope': scope, 'control': control,
            'pair_group': group, 'images': images, 'image': images[0]['image'],
            'image_sha256': images[0]['sha256'], 'prompt': PROMPT})
    for comparator in ('original', 'fixed_second'):
        for scope, window in previous['windows'].items():
            for frame in window['frames']:
                group = f'{comparator}_vs_natural_{frame}'
                add(comparator, 'natural_expression', frame, scope, None, group)
                add('natural_expression', comparator, frame, scope, None, group)
    for branch, frame in [('original', 400), ('natural_expression', 400), ('natural_expression', 720), ('gray', 400)]:
        control = 'no_visual_evidence' if branch == 'gray' else 'same_image'
        add(branch, branch, frame, 'control', control, f'{branch}_self_{frame}')
    random.Random(17346).shuffle(cases)
    for index, case in enumerate(cases):
        case['id'] = f'pair_{index:03d}'
    plan = copy.deepcopy(previous)
    plan.update(protocol='native_iris_short_pair_descriptions_v1', frozen_unix=time.time(),
        source_sha256=sha(folder / 'runner.py'), paired_engine_sha256=sha(folder / 'paired_engine.py'),
        reference_plan_sha256=sha(folder / 'reference_plan.json'), reference_result_sha256=sha(folder / 'reference_result.json'),
        cases=cases, max_calls=36, timeout_seconds=900,
        generation={'max_new_tokens': 512, 'do_sample': False},
        analysis='Report raw A/B descriptions and differences for all pre-fixed frames and both orders. Exact-self controls test spurious differences. Do not convert word counts into accuracy or select a candidate from this probe.',
        adoption_rule='No candidate adoption. A simpler different task changes image count, prompt, output schema and token limit together; no isolated causal attribution or final-evaluator independence is claimed.',
        failure_policy='One run with no parsing/model retry. Same three-consecutive-malformed or execution-failure stop as the frozen engine; all unattempted cases retained. No quality-based stopping or prompt changes after outputs.',
        scope='Same eight fixed frame times and archived videos as the previous joint review, now only two aligned images and three description strings per call. No script targets, branch names or controller verdicts sent to the model. 32 ordered comparison calls and four constructed controls; one exposed script, no human labels or motion/identity acceptance.')
    (folder / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (folder / 'manifest.json').write_text(json.dumps({'gpu': plan['gpu'], 'protocol': plan['protocol']}, indent=2) + '\n')
    print(json.dumps({'folder': str(folder), 'cases': len(cases), 'plan_sha256': sha(folder / 'plan.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    if args.prepare:
        prepare(args)
    else:
        folder = args.folder.resolve()
        plan = json.loads((folder / 'plan.json').read_text())
        assert sha(folder / 'runner.py') == plan['source_sha256']
        assert sha(folder / 'paired_engine.py') == plan['paired_engine_sha256']
        assert sha(folder / 'support.py') == plan['support_source_sha256']
        if args.worker:
            load(folder / 'paired_engine.py', 'paired_engine').worker(folder, validator=validate)
        else:
            load(folder / 'support.py', 'paired_runtime_support').supervise(folder)
