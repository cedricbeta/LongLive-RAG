#!/usr/bin/env python3
"""Publish the closed automatic no-op and its contradictory color decisions."""
import argparse, hashlib, html, json, re, shutil
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('--target', type=Path, required=True); args = p.parse_args()
root, target = Path(__file__).resolve().parent, args.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = read(root / 'summary.json')
assert summary['status'] == 'closed_both_arms_pass_original_with_color_judgment_conflict'
assert read(root / 'queue_status.json')['status'] == 'completed'
required_images = set()
for p in root.glob('*/run/vlm_cache/*/*.json'):
    required_images.update(Path(x).resolve() for x in read(p)['ordered_image_paths'])
actual_inputs = len(required_images)
required_images.update(p.resolve() for p in root.glob('*/run/backend/frames/**/*.png'))
required_images.add((root / 'actual_color_inputs.jpg').resolve())
files, paths, omitted = {}, [], []
def bind(dest):
    key = str(dest.relative_to(target)); assert key not in files
    files[key] = {'sha256': sha(dest), 'bytes': dest.stat().st_size}; paths.append(key)
for source in sorted(root.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts or '.call_locks' in source.parts or source.suffix == '.lock':
        continue
    if source.suffix in ['.png', '.jpg'] and source.resolve() not in required_images:
        omitted.append({'path': str(source.relative_to(root)), 'sha256': sha(source), 'bytes': source.stat().st_size}); continue
    assert source.suffix in ['.py', '.json', '.jsonl', '.yaml', '.txt', '.log', '.pid', '.png', '.jpg'] or source.name in ['STOP', 'EXITED'], source
    if source.suffix in ['.json', '.jsonl', '.txt', '.log']:
        assert not re.search(r'\bhf_[A-Za-z0-9]{20,}|X-Amz-(?:Signature|Credential)=|Authorization: Bearer ', source.read_text()), source
    dest = target / 'results' / root.name / source.relative_to(root)
    assert not dest.exists(), dest
    dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest); bind(dest)
dest = target / 'results' / root.name / 'omitted_derived_frames.json'
dest.write_text(json.dumps({'reason': 'Redundant decoded frames remain local and are recoverable from the published original. Every actual VLM input and backend evidence frame is included.', 'files': omitted}, indent=2) + '\n'); bind(dest)
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ns100 自动对照：颜色误判，零修复</title><style>body{font:17px/1.65 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}video,img{width:100%}pre{white-space:pre-wrap;background:#f3f4f6;padding:18px}</style>
<h1>ns100 自动对照：两组误判原片通过，零修复</h1>
<p>反馈与固定噪声两组各做 11 次模型调用，均停止在原片诊断。没有生成候选，也没有执行两种策略之间的动作差异。原始记录中的 fully_verified_current_shot 不能作为可信语义验收。</p>
<p><a href="../../reproduction/native_ns100_conditional_pair/README.md">协议与复现</a> · <a href="../../results/native_ns100_conditional_pair_v1/summary.json">原始判断与成本</a></p>
<h2>完整原片 · 官方 LongLive-RAG 1.3B · 60 秒</h2>
<video controls preload="metadata" src="../native_development_pilot12_batch4_v1/ns100/baseline/videos/rank0-0-0_lora.mp4"></video>
<p>原文要求 dark green work shirt。开发者观察及下列实际输入显示深蓝衬衫，检查器却将三张图全部判为深绿通过。两组实际颜色输入和回复相同。</p>
<h2>检查器实际收到的裁剪图</h2>
<img loading="lazy" alt="实际三张颜色检查输入，蓝色衬衫" src="../../results/native_ns100_conditional_pair_v1/actual_color_inputs.jpg">
'''
for row in sorted(summary['arms'][0]['color_replies'], key=lambda x: Path(x['actual_input_paths'][0]).name):
    page += '<h3>' + html.escape(Path(row['actual_input_paths'][0]).name) + '</h3><pre>' + html.escape(json.dumps(row['color_reply'], indent=2, ensure_ascii=False)) + '</pre>'
page += '<p>其中一条回复明确说观察到 dark blue，却认为与 dark green 足够接近。完整回复保持原样，不改成失败再冒充自动修复。两个自动预算均已关闭；另行固定目标生成探针必须单独解释。零候选、零保留、零人评和独立分数。</p></html>\n'
dest = target / 'artifacts' / root.name / 'index.html'; dest.parent.mkdir(parents=True, exist_ok=False); dest.write_text(page); bind(dest)
bind(target / 'reproduction/native_ns100_conditional_pair/README.md')
parent = target / 'artifacts/native_development_pilot12_batch4_v1/ns100/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == summary['arms'][0]['selected_original_sha256']
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps({'files': files, 'actual_vlm_input_count': actual_inputs, 'exported_image_count': len(required_images),
    'omitted_redundant_frames': len(omitted), 'reused_original': {'repository_path': str(parent.relative_to(target)), 'sha256': sha(parent)},
    'new_video_candidates': 0, 'scope': 'Automatic original false pass, not repair efficacy.'}, indent=2) + '\n')
paths.append(str(dest.relative_to(target)))
(root.parent / 'native_ns100_automatic_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'paths': len(paths), 'bytes': sum(x['bytes'] for x in files.values()), 'actual_vlm_inputs': actual_inputs}))
