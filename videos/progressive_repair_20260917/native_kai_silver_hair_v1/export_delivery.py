"""Export one completed Kai probe and preserve all earlier deliveries."""
import hashlib
import json
import shutil
import time
from html import escape
from pathlib import Path

S = Path('/home/chendong/video-gen/LongLive-RAG')
T = Path('/home/chendong/video-gen/ICLR27-Longlive-agent')
R = S / 'videos/progressive_repair_20260917'
F = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(F / 'finalize_queue_status.json')['phase'] == 'completed'
obs = read(F / 'developer_observations.json')
assert read(F / 'summary.json')['repair_attempts'] == 1
assert not (T / 'results/native_kai_silver_hair_v1').exists()
records = []


def copy(src, rel):
    dst = T / rel
    assert not dst.exists(), dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = sha(src)
    assert sha(dst) == digest
    records.append({'path': str(rel), 'source': str(src.relative_to(S)),
                    'sha256': digest, 'bytes': dst.stat().st_size})


for src in sorted(F.rglob('*')):
    if not src.is_file() or '__pycache__' in src.parts:
        continue
    rel = src.relative_to(F)
    if src.suffix in ('.json', '.yaml', '.txt', '.py', '.log', '.npz'):
        copy(src, Path('results/native_kai_silver_hair_v1') / rel)
    elif src.suffix == '.png':
        copy(src, Path('artifacts/native_kai_silver_hair_v1') / rel)
copy(next((F / 'candidate/videos').glob('*.mp4')),
     Path('artifacts/native_kai_silver_hair_v1/candidate.mp4'))
copy(F / 'frame_review/full_60s_comparison.mp4',
     Path('artifacts/native_kai_silver_hair_v1/full_60s_comparison.mp4'))
copy(F / 'hair_mask/mask.pt',
     Path('reproduction/native_kai_silver_hair/automatic_hair_mask.tensor'))
telemetry = R / 'native_kai_silver_hair_v1_telemetry'
assert read(telemetry / 'status.json')['status'] == 'completed'
for src in sorted(telemetry.iterdir()):
    if src.is_file():
        copy(src, Path('results') / telemetry.name / src.name)
copy(S / 'scripts/review_native_fixed_target.py',
     Path('implementation/scripts/review_native_fixed_target.py'))
# A distinct export name preserves the old published auditor and its hashes.
# The only change is a limitation sentence conditioned on actual prefix length.
copy(S / 'scripts/audit_native_fast_repair.py',
     Path('implementation/scripts/audit_native_fixed_target.py'))
copy(F / 'REPRODUCTION.md', Path('reproduction/native_kai_silver_hair/README.md'))
page = T / 'artifacts/native_kai_silver_hair_v1/index.html'
page.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kai：银发目标的一次修复探针</title>
<style>body{font:17px/1.65 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f6f8fa;color:#19303d}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e5}video,img{width:100%;height:auto}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style>
<h1>Kai：银发目标，一次封顶探针</h1>
<p>官方 LongLive-RAG 1.3B，完整 60 秒，960 帧，16 fps。左：原片；右：固定目标候选。目标为第三镜头 20–30 秒的 Kai 头发颜色；完整前段及后续均保留。</p>
<p class="note">''' + escape(obs['summary_zh']) + '''</p>
<article><video controls preload="metadata" src="full_60s_comparison.mp4"></video>
<p>快速查看：<a href="full_60s_comparison.mp4#t=20">20 秒起点</a> · <a href="full_60s_comparison.mp4#t=25">25 秒局部改善</a> · <a href="full_60s_comparison.mp4#t=35">35 秒再次变色</a></p>
<p><a href="../native_sampler_cases_v1/ns280_baseline.mp4">完整原片</a> · <a href="candidate.mp4">候选原分辨率视频</a></p>
<p>看过原片后选择的开发探针。年龄漂移、后续 Ava 未离场等原有错误单独保留，不把修发色当成修好身份或完整脚本。原始第三镜头提示词加“Kai has silver hair.”；强度 1.0，种子 4245，原生 x0/re-noise 采样器。没有自动验收或独立人工标注。</p></article>
<article><h2>生成前的自动区域</h2><img loading="lazy" src="hair_mask/mask_latent_0101.png" alt="紫色为程序选中的 Kai 头发区域">
<p>普通 man 定位在一张预览中为空。唯一允许的 man in a brown coat 方案在三张预览中覆盖 Kai 头发，未明显选中 Ava 头发；未手工画区域。粗网格边界和人物定位不是准确标注。</p></article>
<article><h2>全部 18 个预定时刻</h2>''' + ''.join(
    f'<img loading="lazy" src="frame_review/fixed_views_{i}.png" alt="成对固定样本 {i}">'
    for i in (1, 2, 3)) + '''</article>
<p><a href="../../results/native_kai_silver_hair_v1/developer_observations.json">逐项观察与限制</a> ·
<a href="../../results/native_kai_silver_hair_v1/summary.json">次数和成本</a> ·
<a href="../../reproduction/native_kai_silver_hair/README.md">复现</a> ·
<a href="../historical_repair_showcase_20260917/index.html">历史 5B 实例（不同版本）</a></p></html>''')
records.append({'path': str(page.relative_to(T)), 'source': None,
                'sha256': sha(page), 'bytes': page.stat().st_size})
manifest = T / 'results/native_kai_silver_hair_v1/export_manifest.json'
manifest.write_text(json.dumps({'phase': 'completed_bounded_exposed_kai_hair_probe',
    'exported_unix': time.time(), 'files': records, 'human_labels': 0,
    'raw_latents_and_weights_in_git': False}, indent=2) + '\n')
paths = [r['path'] for r in records] + [str(manifest.relative_to(T)), 'README.md']
(R / 'native_kai_hair_export_paths.json').write_text(json.dumps(paths, indent=2) + '\n')
print(json.dumps({'bound_files': len(records), 'bytes': sum(r['bytes'] for r in records)}))
