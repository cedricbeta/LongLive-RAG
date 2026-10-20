#!/usr/bin/env python3
"""Plot actual review times for the original and both second candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root, output):
    if output.exists():
        raise FileExistsError('Use a fresh figure path')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(root))] = sha(path)
        return json.loads(path.read_text())
    baseline = read(root / 'results/official_iris_60s_baseline_prepared/audit.json')
    original, = (root / 'artifacts/official_iris_60s_baseline_prepared/videos').glob('*.mp4')
    videos = [(original, baseline['video_sha256'], 'Original shared parent')]
    for name, label in [('native_iris_mixed_evidence_v1', 'Feedback: changed expression'),
                        ('native_iris_fixed_noise_v1', 'Control: fixed expression')]:
        summary = read(root / 'results' / name / 'summary.json')
        assert summary['worker_status']['status'] == 'completed'
        second = summary['native_render_jobs'][1]
        assert second['status']['status'] == 'completed'
        label += '\nCandidate 2, seed 4244'
        videos.append((root / 'artifacts' / name / 'candidate.mp4', second['engineering_audit']['video_sha256'], label))
    frames = [320, 400, 479]
    figure, axes = plt.subplots(3, 3, figsize=(13.5, 8.5))
    for column, (video, expected, label) in enumerate(videos):
        assert sha(video) == expected
        inputs[str(video.relative_to(root))] = expected
        capture = cv2.VideoCapture(str(video))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 960
            assert capture.get(cv2.CAP_PROP_FPS) == 16
            for row, frame in enumerate(frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                assert ok, (video.name, frame)
                axes[row, column].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
                if row == 0:
                    axes[row, column].set_title(label, fontsize=11)
                if column == 0:
                    axes[row, column].set_ylabel(f'Frame {frame}\n{frame / 16:.4g} s', fontsize=10)
                for spine in axes[row, column].spines.values():
                    spine.set_visible(False)
        finally:
            capture.release()
    figure.suptitle('Exposed Iris development pair: source frames, not independent quality ratings', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    report = {'stage': 'paired_actual_frame_illustration', 'export_frames': frames,
        'frame_scope': 'Same export times as the current attribute-review samples; complete frames shown, while the checker also uses actor crops.',
        'inputs_sha256': inputs, 'figure_sha256': sha(output),
        'source_script_sha256': sha(Path(__file__)),
        'environment': {'matplotlib': matplotlib.__version__, 'opencv': cv2.__version__},
        'new_model_calls': 0, 'new_generated_videos': 0, 'actual_human_labels': 0,
        'limits': ['Original shared parent and both completed second candidates, including failures.',
                   'Plot resizing is presentation only; source MP4 files remain separately published.',
                   'A frame illustration is not an accuracy label, motion score or independent evaluation.']}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'figure': str(output), 'frames': frames, 'video_columns': len(videos)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plot(args.root.resolve(), args.output.resolve())
