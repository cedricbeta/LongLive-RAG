#!/usr/bin/env python3
"""Show all three complete videos and the frozen frame grid, without selection."""
import hashlib, json, subprocess, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan, queue = read(root / 'plan.json'), read(root / 'queue_status.json')
assert queue['status'] == 'completed'
assert all(x['generation_status'] == 'completed' and x['audit_returncode'] == 0 for x in queue['arms'])
parent = Path(plan['parent'])
videos = [('B_original', parent, 'B: original'),
          ('P0_prompt', root / 'P0_prompt/baseline', 'P0: prompt only'),
          ('R0_local', root / 'R0_local/baseline', 'R0: fixed-target local')]
out = root / 'comparison'
out.mkdir(exist_ok=False)
frames_out = out / 'fixed_views'; frames_out.mkdir()
started = time.monotonic(); video_records = []; frames = {}
for name, folder, label in videos:
    video, = (folder / 'videos').glob('*.mp4')
    audit = read(folder / 'audit.json')
    assert sha(video) == audit['video_sha256']
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 960 and capture.get(cv2.CAP_PROP_FPS) == 16
    row = {'arm': name, 'path': str(video), 'sha256': sha(video), 'label': label, 'frames': []}
    try:
        for frame in plan['review_frames']:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read(); assert ok, (name, frame)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            assert rgb.shape == (480, 832, 3)
            frames[(name, frame)] = Image.fromarray(rgb)
            row['frames'].append({'index': frame, 'seconds': frame / 16,
                'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
    finally:
        capture.release()
    video_records.append(row)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font = ImageFont.truetype(font_path, 20)
grid_records = []
for shot in range(6):
    indices = [shot * 160, shot * 160 + 80, shot * 160 + 159]
    canvas = Image.new('RGB', (1872, 1200), 'white'); draw = ImageDraw.Draw(canvas)
    for column, (name, _, label) in enumerate(videos):
        for row, frame in enumerate(indices):
            origin = (column * 624, row * 400)
            draw.text((origin[0] + 10, origin[1] + 10), f'{label} | {frame / 16:.4g} s', font=font, fill='black')
            canvas.paste(frames[(name, frame)].resize((624, 360), Image.Resampling.LANCZOS), (origin[0], origin[1] + 40))
    path = frames_out / f'shot_{shot + 1:02d}.jpg'
    canvas.save(path, quality=94)
    grid_records.append({'shot': shot, 'frames': indices, 'file': str(path.relative_to(root)), 'sha256': sha(path)})
command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error']
for row in video_records:
    command += ['-threads', '4', '-i', row['path']]
filters = []
for i, row in enumerate(video_records):
    # Static labels are controlled strings; no user text is inserted into the filter.
    label = row['label'].replace(':', r'\:')
    filters.append(f"[{i}:v]scale=624:360:flags=lanczos,pad=624:400:0:40:white,drawtext=fontfile={font_path}:text='{label}':x=10:y=10:fontsize=20:fontcolor=black[v{i}]")
filters.append('[v0][v1][v2]hstack=inputs=3,format=yuv420p[v]')
output = out / 'full_60s_comparison.mp4'
command += ['-filter_complex_threads', '2', '-filter_complex', ';'.join(filters), '-map', '[v]',
    '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '18', '-movflags', '+faststart',
    '-frames:v', '960', str(output)]
with (out / 'ffmpeg.log.txt').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
cap = cv2.VideoCapture(str(output)); count = 0
assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 16
while True:
    ok, frame = cap.read()
    if not ok: break
    assert frame.shape == (400, 1872, 3)
    count += 1
cap.release(); assert count == 960
record = {'status': 'completed', 'source_script_sha256': sha(Path(__file__)), 'inputs': video_records,
    'fixed_views': grid_records, 'ffmpeg_command': command, 'output_sha256': sha(output),
    'all_output_frames_decoded': count, 'fps': 16, 'seconds': 60, 'wall_seconds': time.monotonic() - started,
    'opencv_version': cv2.__version__, 'pixel_operations': 'Whole frames resized for layout and text labels added above. No crop, recoloring, retiming, interpolation or hidden segment removal.',
    'new_model_calls': 0, 'additional_generated_candidates': 0, 'human_labels': 0, 'automatic_acceptance': False,
    'limits': ['All complete videos remain available separately at original resolution.',
               'The 54 frozen stills cover all six windows but do not certify continuous motion, identity or action preservation.',
               'R0 is a developer fixed-target probe that bypasses automatic diagnosis and acceptance.']}
(out / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'status': 'completed', 'comparison': str(output), 'frames': count, 'frozen_views': 54}))
