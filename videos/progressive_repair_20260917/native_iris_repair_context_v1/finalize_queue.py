"""Finish this one already-launched ON attempt; never launch another render."""
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time


folder = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
began = time.monotonic()
record = {'status': 'waiting_existing_on', 'started_unix': time.time(),
    'source_sha256': sha(Path(__file__)), 'new_render_launches': 0,
    'audit_source_sha256': sha(folder / 'audit_source.py'),
    'review_source_sha256': sha(folder / 'review_source.py'),
    'off_audit_sha256': sha(folder / 'off/context_audit.json')}
def write():
    temp = folder / 'finalize_status.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(folder / 'finalize_status.json')
assert not (folder / 'finalize_status.json').exists()
assert read(folder / 'off/context_audit.json')['off_conformance_passed']
write()
try:
    while time.monotonic() - began < 1920:
        status_path = folder / 'on/status.json'
        if status_path.exists():
            status = read(status_path)
            if status['status'] not in ('preflight', 'running'):
                break
        time.sleep(2)
    else:
        raise TimeoutError('Existing ON supervisor did not finish within finalizer wait budget')
    if status['status'] != 'completed':
        raise RuntimeError('ON did not complete: ' + status['status'])
    for name, key in [('audit_source.py', 'audit_source_sha256'), ('review_source.py', 'review_source_sha256')]:
        assert sha(folder / name) == record[key]
    record['status'] = 'auditing_on'
    write()
    with (folder / 'on/context_audit.log.txt').open('x') as log:
        subprocess.run([sys.executable, '-u', str(folder / 'audit_source.py'), '--folder', str(folder), '--case', 'on'],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    record['status'] = 'building_complete_video'
    write()
    with (folder / 'review.log.txt').open('x') as log:
        subprocess.run([sys.executable, '-u', str(folder / 'review_source.py'), '--folder', str(folder)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    review = read(folder / 'frame_review/review.json')
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>局部重画上下文对照</title><style>body{font:17px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#17232e}video{width:100%}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%}section{margin:30px 0}@media(max-width:750px){.row{grid-template-columns:1fr}}</style>',
        '<h1>局部重画的历史上下文对照</h1><p>官方 LongLive-RAG 1.3B；同一父版本、自动 mask、提示词、种子 4244 和强度 1.0。目标：Iris 的苔绿色外套。</p>',
        '<p>从左至右：原始父视频、完整上下文的 C 分支、仅在局部重画时使用当前块 K/V 的实验候选。关闭模式已完整复现 C。包含全部 960 帧、60 秒。</p>',
        '<p><strong>这是一次固定动作的开发机制探针，未运行可信语义验收，未自动保留，不代表最新 progressive 的成功。</strong>改变包括旧 sink、检索帧与局部历史，不能单独归因为 RAG。</p>',
        '<video controls preload="metadata" src="three_way_comparison.mp4"></video><p><a href="three_way_comparison.mp4">下载完整三栏对照</a> · <a href="review.json">来源和逐帧记录</a></p>']
    labels = {'original': '原始父视频', 'full_context': '完整上下文（C = OFF）', 'current_block': '当前块上下文（ON 候选）'}
    for frame in review['review_frames']:
        parts.append(f'<section><h2>第 {frame} 帧 · {frame / 16:g} 秒</h2><div class="row">')
        for row in review['images']:
            if row['frame'] == frame:
                label = labels[row['branch']]
                parts.append(f'<div><p>{label}</p><img loading="lazy" width="832" height="480" src="{html.escape(row["path"])}" alt="{label} {frame}"></div>')
        parts.append('</div></section>')
    parts.append('</html>')
    (folder / 'frame_review/index.html').write_text('\n'.join(parts) + '\n')
    record.update(status='completed', review_images=len(review['images']),
                  video_frames=review['actual_frames_per_branch'],
                  video_sha256=review['comparison_video']['sha256'],
                  candidate_retained=False, semantic_quality_claim=False)
except Exception as error:
    record.update(status='execution_failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    record['wall_seconds'] = time.monotonic() - began
    write()
