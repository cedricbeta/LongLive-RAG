#!/usr/bin/env python3
"""Create a full-duration four-way comparison of archived development outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def plot(root, output):
    if output.exists():
        raise FileExistsError('Use a new comparison path')
    videos = [
        ('Original', 'artifacts/official_iris_60s_baseline_prepared/videos/rank0-0-0_lora.mp4'),
        ('Feedback second', 'artifacts/native_iris_mixed_evidence_v1/candidate.mp4'),
        ('Fixed-noise second', 'artifacts/native_iris_fixed_noise_v1/candidate.mp4'),
        ('Natural expression', 'artifacts/native_iris_natural_expression_v1/candidate.mp4')]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    def probe(path):
        return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,duration,nb_read_frames',
            '-of', 'json', str(path)], text=True))['streams'][0]
    font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    assert font.is_file()
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n']
    filters = []
    sources = {}
    for index, (label, relative) in enumerate(videos):
        path = root / relative
        assert probe(path) == {'width': 832, 'height': 480, 'avg_frame_rate': '16/1',
                               'duration': '60.000000', 'nb_read_frames': '960'}
        sources[relative] = sha(path)
        command += ['-threads', '1', '-i', str(path)]
        filters.append(f'[{index}:v]scale=416:240:flags=lanczos,pad=416:268:0:28:color=black,'
            f"drawtext=fontfile={font}:text='{label}':fontcolor=white:fontsize=17:x=10:y=4[v{index}]")
    filters.append('[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|416_0|0_268|416_268:shortest=1[out]')
    output.parent.mkdir(parents=True, exist_ok=True)
    command += ['-filter_complex_threads', '1', '-filter_complex', ';'.join(filters), '-map', '[out]',
        '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)
    measured = probe(output)
    assert measured == {'width': 832, 'height': 536, 'avg_frame_rate': '16/1',
                        'duration': '60.000000', 'nb_read_frames': '960'}
    report = {'phase': 'four_way_full_duration_development_comparison', 'inputs_sha256': sources,
        'output_sha256': sha(output), 'output_probe': measured, 'font_sha256': sha(font),
        'source_sha256': sha(Path(__file__)), 'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'labels_in_reading_order': [label for label, _ in videos], 'actual_human_labels': 0,
        'candidate_adopted': False,
        'scope': 'Display-only downsampling and titles, all 960 source frames preserved at corresponding timestamps. Full-resolution source MP4s are separate. This figure is not a generation arm or quality rating.'}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'probe': measured}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plot(args.root.resolve(), args.output.resolve())
