"""Export this completed bounded study without touching prior result folders."""
from pathlib import Path
from html import escape
import hashlib
import json
import shutil
import time

S = Path('/home/chendong/video-gen/LongLive-RAG')
T = Path('/home/chendong/video-gen/ICLR27-Longlive-agent')
R = S / 'videos/progressive_repair_20260917'
F = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(F / 'finalize_queue_status.json')['phase'] == 'completed'
obs = read(F / 'developer_observations.json')
summary = read(F / 'summary.json')
assert summary['repair_attempts'] == 2 and not summary['automatic_acceptance_performed']
assert not (T / 'results/native_noah_hair_sampler_v1').exists()
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
        copy(src, Path('results/native_noah_hair_sampler_v1') / rel)
    elif src.suffix == '.png':
        copy(src, Path('artifacts/native_noah_hair_sampler_v1') / rel)
for name in ['native', 'euler']:
    folder = F / 'sampler_pair' / name
    assert read(folder / 'sampler_audit.json')['status'] == 'passed'
    copy(next((folder / 'videos').glob('*.mp4')),
         Path('artifacts/native_noah_hair_sampler_v1') / (name + '.mp4'))
    telemetry = R / ('native_noah_hair_' + name + '_v1_telemetry')
    assert read(telemetry / 'status.json')['status'] == 'completed'
    for src in sorted(telemetry.iterdir()):
        if src.is_file():
            copy(src, Path('results') / telemetry.name / src.name)
copy(F / 'frame_review/full_60s_three_way.mp4',
     Path('artifacts/native_noah_hair_sampler_v1/full_60s_three_way.mp4'))
copy(F / 'hair_mask/mask.pt',
     Path('reproduction/native_noah_hair_sampler/automatic_hair_mask.tensor'))
for name in ['preview_native_hair_mask.py', 'prepare_noah_hair_sampler_pair.py',
             'review_native_sampler_pair.py']:
    copy(S / 'scripts' / name, Path('implementation/scripts') / name)
copy(F / 'REPRODUCTION.md', Path('reproduction/native_noah_hair_sampler/README.md'))

page = T / 'artifacts/native_noah_hair_sampler_v1/index.html'
page.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Noah：官方 1.3B 发型修复两次对照</title>
<style>body{font:17px/1.65 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;background:#f6f8fa;color:#19303d}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e5}video,img{width:100%;height:auto}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style>
<h1>Noah：剃光头目标，两次封顶对照</h1>
<p>官方 LongLive-RAG 1.3B，完整 60 秒，960 帧，16 fps。左：原片；中：原生 x0/re-noise 修复；右：固定噪声 Euler 修复。每栏保留原分辨率 832×480，全部后续镜头保留。</p>
<p class="note">''' + escape(obs['summary_zh']) + '''</p>
<article><video controls preload="metadata" src="full_60s_three_way.mp4"></video>
<p><a href="../native_sampler_cases_v1/ns000_baseline.mp4">原片</a> · <a href="native.mp4">原生采样候选</a> · <a href="euler.mp4">Euler 候选</a></p>
<p>这是看过原片后选择真实发型缺陷的开发探针，不是留出测试或自动控制器验收。两种候选共用父 latent、自动 mask、完整原始提示词加“Noah has a shaved head.”、种子 4243、强度 1.0、四个原生时间步。比较同时改变采样更新与私有噪声刷新，不能单独归因于其中一项。人工标注数为 0。</p></article>
<article><h2>生成前的自动区域</h2><img loading="lazy" src="hair_mask/mask_latent_0020.png" alt="紫色为自动头发区域">
<p>CLIPSeg person ∩ hair − face，阈值 0.35。最初 man 定位在三张预览中两张为空，改用一次事先记录的 person 名词后才生成候选。未手工画 mask，也未看候选后调 mask；粗网格边缘并非准确标注。</p></article>
<article><h2>全部 18 个预定采样时刻</h2>''' + ''.join(
    f'<img loading="lazy" src="frame_review/fixed_views_{i}.png" alt="三路固定样本 {i}">'
    for i in (1, 2, 3)) + '''</article>
<p>观察来自开发者对固定图像的检查，不是独立评分或连续运动验收。
<a href="../../results/native_noah_hair_sampler_v1/developer_observations.json">逐项观察</a> ·
<a href="../../results/native_noah_hair_sampler_v1/summary.json">次数、成本和限制</a> ·
<a href="../../reproduction/native_noah_hair_sampler/README.md">复现</a> ·
<a href="../historical_repair_showcase_20260917/index.html">历史 5B 四例（不同版本）</a></p></html>''')
records.append({'path': str(page.relative_to(T)), 'source': None,
                'sha256': sha(page), 'bytes': page.stat().st_size})
manifest_path = T / 'results/native_noah_hair_sampler_v1/export_manifest.json'
manifest_path.write_text(json.dumps({'phase': 'completed_bounded_exposed_sampler_pair',
    'exported_unix': time.time(), 'files': records, 'raw_latents_and_weights_in_git': False,
    'human_labels': 0, 'automatic_acceptance_performed': False}, indent=2) + '\n')
paths = [r['path'] for r in records] + [str(manifest_path.relative_to(T)), 'README.md']
(R / 'native_noah_hair_export_paths.json').write_text(json.dumps(paths, indent=2) + '\n')
print(json.dumps({'bound_files': len(records), 'bytes': sum(r['bytes'] for r in records)}))
