#!/usr/bin/env python3
"""Export the completed pair only after all fixed views have been inspected."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary, plan = read(root / 'summary.json'), read(root / 'plan.json')
views = read(root / 'developer_observations.json')
assert summary['new_generation_attempts'] == 2 and summary['remaining_generation_budget'] == 0
assert summary['all_generation_audits_passed'] and views['all_54_predeclared_views_seen']
assert summary['plan_sha256'] == sha(root / 'plan.json')
files, paths, omitted = {}, [], []


def bind(path):
    key = str(path.relative_to(target))
    assert key not in files
    files[key] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    paths.append(key)


for source in sorted(root.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts:
        continue
    rel = source.relative_to(root)
    if source.name == 'latents_0000.pt' or 'sample_frames' in rel.parts:
        omitted.append({'path': str(rel), 'bytes': source.stat().st_size, 'sha256': sha(source)})
        continue
    assert source.suffix in ['.py', '.json', '.jsonl', '.yaml', '.txt', '.log', '.jpg', '.png', '.mp4', '.pt'], source
    if source.suffix in ['.json', '.jsonl', '.txt', '.log']:
        assert not re.search(r'\bhf_[A-Za-z0-9]{20,}|X-Amz-(?:Signature|Credential)=|Authorization: Bearer ', source.read_text()), source
    media = source.suffix == '.mp4' or ('fixed_views' in rel.parts and source.suffix == '.jpg')
    dest = target / ('artifacts' if media else 'results') / root.name / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    shutil.copyfile(source, dest)
    bind(dest)
for name in ['prepare.py', 'README.md']:
    bind(target / 'reproduction/native_ns010_fixed_target_pair' / name)
parent = target / 'artifacts/native_development_pilot12_batch2_v1/ns010/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == plan['parent_video_sha256']
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ns010 首段红发要求：普通提示与局部修复</title>
<style>body{font:17px/1.7 system-ui;max-width:1400px;margin:28px auto;padding:0 20px}video,img{width:100%}</style>
<h1>ns010：普通提示与局部修复的完整对照</h1>
<p>官方 LongLive-RAG 1.3B；开发者选择的固定目标能力探针，绕过自动诊断和验收。两组各生成一次，预算关闭。不是最新 progressive 自动成功、独立评测或人工验收。</p>
<p><a href="../../reproduction/native_ns010_fixed_target_pair/README.md">协议、成本与复现</a> · <a href="../../results/native_ns010_fixed_target_pair_v1/summary.json">实际结果</a> · <a href="../native_ns110_fixed_target_pair_v1/index.html">保留的 ns110 错人定位失败</a></p>
<h2>完整 60 秒三栏对照</h2>
<p>左：原片 B；中：普通提示 P0；右：局部修复 R0。两组只在首段加强同一句 curly red hair 要求，后五段原文不变。完整保留 960 帧；展示仅缩放、加标签和编码。</p>
<video controls preload="metadata" src="comparison/full_60s_comparison.mp4"></video>
'''
page += '<p>' + html.escape(views['conclusion']) + '</p>'
page += '''<p>完整单片：<a href="../native_development_pilot12_batch2_v1/ns010/baseline/videos/rank0-0-0_lora.mp4">原片 B</a> · <a href="P0_prompt/baseline/videos/rank0-0-0_lora.mp4">普通提示 P0</a> · <a href="R0_local/baseline/videos/rank0-0-0_lora.mp4">局部修复 R0</a></p><h2>全部预定抽帧</h2>'''
assert len(views['windows']) == 6
for i, row in enumerate(views['windows'], 1):
    page += f'<h3>{(i-1)*10}–{i*10} 秒</h3><img loading="lazy" alt="第 {i} 段三个固定时间的完整三栏画面" src="comparison/fixed_views/shot_{i:02d}.jpg">'
    page += '<ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in row['observations']) + '</ul>'
page += '<h2>实际程序区域</h2><p>CPU CLIPSeg；无手绘或事后修改。区域覆盖侧面和下部头发，中央发顶和部分近脸发丝未覆盖。程序区域不等于完整目标定位或像素保护。</p>'
for index in [0, 20, 40]:
    page += f'<img loading="lazy" alt="latent {index} 程序区域" src="../../results/native_ns010_fixed_target_pair_v1/garment_mask/mask_latent_{index:04d}.png">'
page += '<h2>实际成本</h2><ul>'
for row in summary['arms']:
    page += '<li>' + html.escape(row['arm']) + f'：生成 {row["generation_seconds"]:.3f} 秒，工程检查 {row["audit_seconds"]:.3f} 秒。</li>'
page += f'<li>顺序队列 {summary["queue_seconds"]:.3f} 秒；CPU 区域监督 {summary["mask_supervised_seconds"]:.3f} 秒；准备 {summary["preparation_seconds"]:.3f} 秒。</li></ul>'
page += '<p>不是等计算预算对照。私有去噪时间已包含在生成时间中，不可重复相加。没有人评、独立身份／动作验收或连续运动结论；所有原片缺陷与候选副作用均保留。</p></html>\n'
dest = target / 'artifacts' / root.name / 'index.html'
assert not dest.exists()
dest.write_text(page)
bind(dest)
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps({'files': files, 'omitted_files': omitted,
    'omission_reason': 'Generated latents require regeneration; redundant decoded frames are recoverable from full published videos. All six complete comparison contact sheets are included.',
    'reused_parent': {'repository_path': str(parent.relative_to(target)), 'sha256': sha(parent)},
    'scope': 'Two fixed-target developer probes; no automatic retained improvement or independent quality acceptance.'}, indent=2) + '\n')
paths.append(str(dest.relative_to(target)))
(root.parent / 'native_ns010_fixed_target_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'paths': len(paths), 'bound_bytes': sum(x['bytes'] for x in files.values()), 'omitted_files': len(omitted)}))
