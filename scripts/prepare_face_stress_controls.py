#!/usr/bin/env python3
"""Construct bounded nuisance/visibility controls; these are not human labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out = args.output.resolve()
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    observations = json.loads((args.observations / 'observations.json').read_text())
    source = {v['source_filename']: (key, v) for key, v in observations.items()}
    names = ['iris_parent_1.png', 'iris_candidate_1.png', 'actor_left_200.png', 'actor_right_200.png']
    cases, transforms, originals = [], [], []
    for name in names:
        digest, observation = source[name]
        original = args.observations / 'evidence' / (digest + '.png')
        assert sha(original) == digest and observation['detected_faces'] == 1
        image = cv2.imread(str(original))
        originals.append(image)
        dest = out / 'inputs' / name
        shutil.copy2(original, dest)
        height, width = image.shape[:2]
        encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 35])[1]
        occluded = image.copy()
        x, y, w, h = observation['detections_xywh_landmarks_score'][0][:4]
        x0, y0 = max(0, int(np.floor(x)) - 5), max(0, int(np.floor(y)) - 5)
        x1, y1 = min(width, int(np.ceil(x + w)) + 5), min(height, int(np.ceil(y + h)) + 5)
        occluded[y0:y1, x0:x1] = 128
        variants = [('jpeg_q35', cv2.imdecode(encoded, cv2.IMREAD_COLOR), 'pass'),
                    ('brightness_075', np.round(image.astype(float) * 0.75).astype('uint8'), 'pass'),
                    ('border_32', cv2.copyMakeBorder(image, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=(128,128,128)), 'pass'),
                    ('detected_face_occluded', occluded, 'unknown'),
                    ('pixelated_12', cv2.resize(cv2.resize(image, (12,12), interpolation=cv2.INTER_AREA),
                                              (width,height), interpolation=cv2.INTER_NEAREST), 'unknown')]
        for label, array, expected in variants:
            path = out / 'inputs' / (Path(name).stem + '_' + label + '.png')
            cv2.imwrite(str(path), array)
            cases.append({'id': Path(name).stem + '_' + label, 'frames_a': [str(dest)],
                          'frames_b': [str(path)], 'expected_control_status': expected})
            transforms.append({'output': path.name, 'input_sha256': digest, 'operation': label,
                               'occlusion_box_xyxy': [x0,y0,x1,y1] if label == 'detected_face_occluded' else None})
    # Two complete source portraits in the same image, with no subject selected.
    left, right = originals[2:]
    both = np.full((max(left.shape[0],right.shape[0]), left.shape[1] + right.shape[1], 3), 128, np.uint8)
    both[:left.shape[0], :left.shape[1]] = left
    both[:right.shape[0], left.shape[1]:] = right
    rng = np.random.default_rng(0)
    for label, image in [('two_faces', both), ('noise_seed0', rng.integers(0,256,(256,256,3),dtype=np.uint8)),
                         ('black', np.zeros((256,256,3),np.uint8)), ('white', np.full((256,256,3),255,np.uint8))]:
        path = out / 'inputs' / (label + '.png')
        cv2.imwrite(str(path), image)
        cases.append({'id': label, 'frames_a': [str(path)], 'frames_b': [str(path)], 'expected_control_status': 'unknown'})
    manifest = {'phase': 'constructed_face_nuisance_and_visibility_controls', 'cases': cases,
        'input_sha256': {str(p.relative_to(out)): sha(p) for p in (out / 'inputs').glob('*.png')},
        'source_observation_sha256': sha(args.observations / 'observations.json'),
        'transforms': transforms, 'source_sha256': sha(Path(__file__)),
        'limits': ['Program-constructed expectations, not participant judgments or population calibration.',
                   'No positive example of a new generated pose/lighting or hard same-gender negative is labeled here.',
                   'Occlusion boxes are detector-derived test fixtures, never method repair masks.',
                   'Parameters and expectation rules fixed before running this expanded suite.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(Path(__file__), out / 'source.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
