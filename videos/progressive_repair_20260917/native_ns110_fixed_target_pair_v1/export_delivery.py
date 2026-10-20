#!/usr/bin/env python3
"""Export the closed wrong-person localization result, without video generation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True)
args = p.parse_args()
root, target = Path(__file__).resolve().parent, args.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s = read(root / 'summary.json')
assert s['status'] == 'closed_before_generation_wrong_actor_mask'
assert not (root / 'P20_prompt').exists() and not (root / 'R20_local').exists()
for kind in ['artifacts', 'results']:
    assert not (target / kind / root.name).exists()
files, paths = {}, []


def bind(path):
    rel = str(path.relative_to(target))
    files[rel] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    paths.append(rel)


for source in sorted(root.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts:
        continue
    assert source.suffix in ['.py', '.json', '.txt', '.png', '.pt']
    assert source.stat().st_size < 10 * 1024**2
    kind = 'artifacts' if source.suffix == '.png' else 'results'
    dest = target / kind / root.name / source.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    bind(dest)
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ns110：定位选错人，关闭生成名额</title>
<style>body{font:17px/1.7 system-ui;max-width:1000px;margin:30px auto;padding:0 20px}img,video{max-width:100%}</style>
<h1>ns110：定位选错人，本例零生成</h1>
<p>开发者从已看过的官方 1.3B 原片选择 Luna 的黑色高领衣要求，计划检查 20–30 秒。一次现成 CLIPSeg 定位得到非空区域，但三张固定预览均覆盖其他女性的衣服，中间浅色高领衣没有被覆盖。因此按事先规则关闭两次生成名额，不修改短语、阈值或区域重试。</p>
<p>这是固定目标定位能力探针，不是完整自动流程，也没有提示控制／局部修复视频可比较。CPU 定位进程 200.072 秒，含启动的监督时间 202.422 秒，零 GPU 生成、零语义检查器调用、零保留改进。</p>
<p><a href="../../reproduction/native_ns110_fixed_target_pair/README.md">来源与复现</a> · <a href="../../results/native_ns110_fixed_target_pair_v1/summary.json">实际结果</a> · <a href="../native_development_pilot12_batch4_v1/ns110/baseline/videos/rank0-0-0_lora.mp4">完整原片</a></p>
<h2>20 秒：覆盖左侧浅棕外套及书架</h2><img src="garment_mask/mask_latent_0081.png" alt="20 秒定位预览，紫色为区域">
<h2>25 秒：主要覆盖后来出现的左侧蓝衣女性</h2><img src="garment_mask/mask_latent_0101.png" alt="25 秒定位预览，紫色为区域">
<h2>29.75 秒：覆盖左侧两人，中间高领衣仍未覆盖</h2><img src="garment_mask/mask_latent_0120.png" alt="29.75 秒定位预览，紫色为区域">
<p>紫色为程序生成的区域叠加，不是生成后的视频。没有人工像素坐标或手改 mask；开发者只作预设的可执行性筛查。三张预览不等于对全部 160 帧的分割质量测量。所有实际区域张量、逐帧输入哈希、参数、源码及耗时均保留；准备好的生成入口没有执行。</p></html>\n'''
dest = target / 'artifacts' / root.name / 'index.html'
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(page)
bind(dest)
bind(target / 'reproduction/native_ns110_fixed_target_pair/README.md')
parent = target / 'artifacts/native_development_pilot12_batch4_v1/ns110/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == read(root / 'plan.json')['parent_video_sha256']
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps({'files': files, 'reused_original': {'path': str(parent.relative_to(target)), 'sha256': sha(parent)},
    'new_generation_attempts': 0, 'all_actual_localization_outputs_included': True,
    'scope': 'Closed exposed developer-targeted mask admission failure; no repair efficacy result.'}, indent=2) + '\n')
paths.append(str(dest.relative_to(target)))
(root.parent / 'native_ns110_mask_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'paths': len(paths), 'bound_bytes': sum(x['bytes'] for x in files.values()), 'new_generations': 0}))
