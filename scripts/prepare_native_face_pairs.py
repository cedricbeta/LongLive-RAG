#!/usr/bin/env python3
"""Prepare unlabeled, fixed-time native-video pairs for CPU face diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2


def main(args):
    out = args.output.resolve()
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frames, videos = {}, {}
    for role, folder in [('parent', args.parent), ('candidate', args.candidate)]:
        assert json.loads((folder / 'status.json').read_text())['status'] == 'completed'
        assert json.loads((folder / 'manifest.json').read_text())['source_benchmark_index'] == 90
        video, = (folder / 'videos').glob('*.mp4')
        videos[role] = sha(video)
        cap = cv2.VideoCapture(str(video))
        assert cap.get(cv2.CAP_PROP_FPS) == 16 and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 960
        try:
            for segment in (2, 4):
                frames[role, segment] = []
                for frame in [segment * 160 + x for x in (32, 80, 128)]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, pixels = cap.read()
                    assert ok
                    path = out / 'inputs' / f'{role}_{frame:04d}.png'
                    cv2.imwrite(str(path), pixels)
                    frames[role, segment].append(str(path))
        finally:
            cap.release()
    manifest = {'video_sha256': videos, 'frame_offsets': [32, 80, 128],
        'cases': [{'id': name, 'expected_control_status': None,
                   'frames_a': frames['parent', segment], 'frames_b': frames['candidate', segment]}
                  for name, segment in [('native_iris_current', 2), ('native_iris_later', 4)]],
        'input_sha256': {str(p.relative_to(out)): sha(p) for p in (out / 'inputs').glob('*.png')},
        'source_sha256': sha(__file__), 'labels': 'Unlabeled actual generated pairs, not synthetic known-answer controls',
        'candidate_promoted': False, 'independent_final_evaluation': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(__file__, out / 'prepare_native_face_pairs.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'candidate', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
