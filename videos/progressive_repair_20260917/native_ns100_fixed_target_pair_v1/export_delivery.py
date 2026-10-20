#!/usr/bin/env python3
"""Export both closed probe arms, complete videos and bounded visual findings."""
import argparse, hashlib, html, json, re, shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary, plan = read(root / 'summary.json'), read(root / 'plan.json')
assert summary['new_generation_attempts'] == 2 and summary['remaining_generation_budget'] == 0
assert summary['plan_sha256'] == sha(root / 'plan.json')
assert summary['initial_prompt_audit_failure_retained']
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
    if dest.exists():
        assert sha(dest) == sha(source), dest
    else:
        shutil.copyfile(source, dest)
    bind(dest)
for name in ['prepare.py', 'README.md']:
    bind(target / 'reproduction/native_ns100_fixed_target_pair' / name)
parent = target / 'artifacts/native_development_pilot12_batch4_v1/ns100/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == plan['parent_video_sha256']
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ns100 固定目标：衬衫变绿，但普通提示也能做到</title>
<style>body{font:17px/1.7 system-ui;max-width:1400px;margin:28px auto;padding:0 20px}video,img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:9px}</style>
<h1>ns100：局部修复与普通提示都能把衬衫改绿</h1>
<p>官方 LongLive-RAG 1.3B；固定目标能力探针，绕过自动诊断。两组各生成一次，预算关闭。不是最新 progressive 自动成功、独立评测或人工验收。</p>
<p><a href="../../reproduction/native_ns100_fixed_target_pair/README.md">协议、成本与复现</a> · <a href="../../results/native_ns100_fixed_target_pair_v1/summary.json">机器记录</a> · <a href="../native_ns100_conditional_pair_v1/index.html">先前自动检查器误判原片通过</a></p>
<h2>完整 60 秒三栏对照</h2>
<p>左：原片 B；中：普通提示 P0；右：局部修复 R0。只在首段加强同一句服装要求，后五段原文不变。保留全部 960 帧；展示仅缩放、加标签和编码。</p>
<video controls preload="metadata" src="comparison/full_60s_comparison.mp4"></video>
<p>开发者看过全部 54 个预定视图：两组的绿色在各自 18 个抽帧中均保持到 59.94 秒。普通提示已有同类颜色收益，不能据此证明局部修复更优。</p>
<p>R0 开头座椅布局和头部姿态较接近原片，但手持笔记本变化；后段出现类似重复人物，结尾袖长改变并有碎片和形变。P0 改变构图、物件与姿态，后段多出蓝衣人物。原片也有额外人物和画面缺陷。身份、动作和连续运动保持均未获验证。</p>
<p>完整单片：<a href="../native_development_pilot12_batch4_v1/ns100/baseline/videos/rank0-0-0_lora.mp4">原片 B</a> · <a href="P0_prompt/baseline/videos/rank0-0-0_lora.mp4">普通提示 P0</a> · <a href="R0_local/baseline/videos/rank0-0-0_lora.mp4">局部修复 R0</a></p>
<h2>全部预定抽帧</h2>
'''
observations = read(root / 'developer_observations.json')
for i, row in enumerate(observations['windows'], 1):
    page += f'<h3>{(i-1)*10}–{i*10} 秒</h3><img loading="lazy" alt="第 {i} 段三个固定时间的完整三栏画面" src="comparison/fixed_views/shot_{i:02d}.jpg">'
    page += '<ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in row['observations']) + '</ul>'
page += '<h2>预先生成的程序掩码</h2><p>CPU CLIPSeg，未手绘或事后调整。区域覆盖衬衫，也包含手、笔记本和少量头发；这不是语义保护证明。</p>'
for index in [0, 20, 40]:
    page += f'<img loading="lazy" alt="latent {index} 程序掩码" src="../../results/native_ns100_fixed_target_pair_v1/garment_mask/mask_latent_{index:04d}.png">'
page += '<p>两个生成分别耗时 559.07 / 558.87 秒，并行队列 566.98 秒。掩码监督耗时 165.44 秒；准备 21.59 秒。R0 的 56 次额外私有去噪调用耗时 20.62 秒，已包含在生成时间中。不是等计算预算对照。原始 P0 审计入口错误及随后不重生成的元数据审计恢复均已保留。</p></html>\n'
dest = target / 'artifacts' / root.name / 'index.html'
if dest.exists():
    assert dest.read_text() == page
else:
    dest.write_text(page)
bind(dest)
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps({'files': files, 'omitted_files': omitted,
    'omission_reason': 'Generated latents remain local and require regeneration; redundant decoded frames are recoverable from full published videos. All six comparison contact sheets are included.',
    'reused_parent': {'repository_path': str(parent.relative_to(target)), 'sha256': sha(parent)},
    'scope': 'Two fixed-target developer probes; no automatic retained improvement or independent quality acceptance.'}, indent=2) + '\n')
paths.append(str(dest.relative_to(target)))
(root.parent / 'native_ns100_fixed_target_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'paths': len(paths), 'bound_bytes': sum(x['bytes'] for x in files.values()), 'omitted_files': len(omitted)}))
