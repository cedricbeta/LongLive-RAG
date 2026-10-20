#!/usr/bin/env python3
"""Apply a previously frozen target-blind observation protocol to one new render."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import cv2
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    assert args.gpu in (0, 1, 2)
    reference, control, output = args.reference.resolve(), args.control.resolve(), args.output.resolve()
    previous = json.loads((reference / 'plan.json').read_text())
    frozen = json.loads((control / 'plan.json').read_text())
    assert sha(reference / 'plan.json') == frozen['observation_reference_plan_sha256']
    engineering = json.loads((control / 'candidate/audit.json').read_text())
    assert engineering['all_engineering_checks_passed'] and engineering['status']['status'] == 'completed'
    video = args.root.resolve() / args.video_relative
    assert sha(video) == engineering['video_sha256']
    assert sha(reference / 'runner.py') == previous['source_sha256']
    assert sha(reference / 'model_manifest.json') == previous['model_manifest_sha256']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'inputs').mkdir()
    for name in ('runner.py', 'model_manifest.json'):
        shutil.copy2(reference / name, output / name)
    shutil.copytree(reference / 'model_source', output / 'model_source')
    shutil.copy2(control / 'plan.json', output / 'render_plan.json')
    shutil.copy2(control / 'candidate/audit.json', output / 'render_audit.json')
    shutil.copy2(reference / 'plan.json', output / 'reference_plan.json')
    cases = []
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 960 and cap.get(cv2.CAP_PROP_FPS) == 16
    for index in frozen['review_frames']:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok and frame.shape == (480, 832, 3)
        path = output / 'inputs' / f'frame_{index}.png'
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
        cases.append({'branch': 'natural_expression', 'frame': index, 'seconds': index / 16,
            'image': str(path.relative_to(output)), 'image_sha256': sha(path),
            'scope': 'current' if index < 480 else 'future', 'control': None})
    cap.release()
    gray, = [c for c in previous['cases'] if c['control'] == 'no_person']
    shutil.copy2(reference / gray['image'], output / 'inputs/gray.png')
    cases.append({**gray, 'image': 'inputs/gray.png'})
    repeat, = [c for c in cases if c['frame'] == 400]
    cases.append({**repeat, 'control': 'same_image_repeat'})
    random.Random(17344).shuffle(cases)
    for index, case in enumerate(cases):
        case['id'] = f'image_{index:03d}'
    plan = copy.deepcopy(previous)
    plan.update(protocol='native_iris_natural_expression_observations_v1', frozen_unix=time.time(),
        cases=cases, videos=[{'branch': 'natural_expression', 'repository_path': args.video_relative,
                             'sha256': engineering['video_sha256']}], gpu=args.gpu, max_calls=10,
        timeout_seconds=600,
        observation_reference_plan_sha256=sha(output / 'reference_plan.json'),
        render_plan_sha256=sha(output / 'render_plan.json'), render_audit_sha256=sha(output / 'render_audit.json'),
        hypothesis='Apply the same already frozen target-blind observation prompt and model to the new wording-only render; no quality benefit assumed.',
        scope='One new exposed development candidate, at the eight frame indices frozen before generation. Prompt/model/decoding are inherited exactly. One gray control and one same-image repeat are procedural controls, not added semantic samples. No candidate adoption.')
    for key in ('prompt', 'generation', 'seed_per_call', 'dtype', 'attention', 'package_versions',
                'model', 'model_revision', 'model_path', 'source_sha256', 'model_manifest_sha256'):
        assert plan[key] == previous[key]
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'gpu': args.gpu, 'protocol': plan['protocol']}, indent=2) + '\n')
    print(json.dumps({'folder': str(output), 'calls': len(cases), 'video_sha256': engineering['video_sha256']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'control', 'output', 'root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--video-relative', required=True)
    p.add_argument('--gpu', type=int, default=1)
    prepare(p.parse_args())
