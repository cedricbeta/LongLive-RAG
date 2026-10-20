"""Export the closed ns235 cue control without changing earlier result bundles."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from html import escape

S = Path('/home/chendong/video-gen/LongLive-RAG')
T = Path('/home/chendong/video-gen/ICLR27-Longlive-agent')
R = S / 'videos/progressive_repair_20260917'
F = Path(__file__).resolve().parent
NAME = 'native_ns235_continuation_cue_v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(F / 'summary.json')['remaining_generation_budget'] == 0
assert read(F / 'repair_admission.json')['decision'] == 'close_without_local_pair'
assert not (T / 'results' / NAME).exists()
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
    if src.suffix in ('.json', '.yaml', '.txt', '.py', '.log'):
        copy(src, Path('results') / NAME / rel)
    elif src.suffix == '.png':
        copy(src, Path('artifacts') / NAME / rel)
for arm in ('B_original', 'P_cue'):
    copy(next((F / arm / 'videos').glob('*.mp4')), Path('artifacts') / NAME / (arm + '.mp4'))
    telemetry = R / f'native_ns235_{arm}_v1_telemetry'
    assert read(telemetry / 'status.json')['status'] == 'completed'
    for src in sorted(telemetry.iterdir()):
        if src.is_file():
            copy(src, Path('results') / telemetry.name / src.name)
copy(F / 'baseline_review/full_60s_comparison.mp4', Path('artifacts') / NAME / 'full_60s_comparison.mp4')
for name in ('continuation_cue.py', 'test_continuation_cue.py', 'install_continuation_cue.py'):
    copy(S / 'adapters/official_longlive_rag' / name, Path('adapters/official_longlive_rag') / name)
for name in ('prepare_native_cue_study.py', 'prepare_native_cue_pair.py',
             'audit_native_cue_case.py', 'review_native_cue_study.py'):
    copy(S / 'scripts' / name, Path('implementation/scripts') / name)
copy(F / 'portable_prepare.py', Path('reproduction/native_ns235_continuation_cue/prepare.py'))
copy(F / 'REPRODUCTION.md', Path('reproduction/native_ns235_continuation_cue/README.md'))
obs = read(F / 'developer_observations.json')
page = T / 'artifacts' / NAME / 'index.html'
page.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kai 摄影棚：原版与仅后续银发提醒</title>
<style>body{font:17px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f6f8fa;color:#19303d}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e5}video,img{width:100%;height:auto}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style>
<h1>Kai 摄影棚：一句原有属性提醒的实际效果</h1>
<p>官方 LongLive-RAG 1.3B，完整 60 秒，960 帧，16 fps。左 B_original：原始六段提示词；右 P_cue：仅在 30 秒后的三个提示词边界追加“Kai has silver hair.”。两版前 121 个 latent 完全一致。</p>
<p class="note">''' + escape(obs['summary_zh']) + '''</p>
<article><video controls preload="metadata" src="full_60s_comparison.mp4"></video>
<p>快速查看：<a href="full_60s_comparison.mp4#t=20">20 秒：未达局部修复条件</a> · <a href="full_60s_comparison.mp4#t=35">35 秒：开始可见差异</a> · <a href="full_60s_comparison.mp4#t=45">45 秒：银发保持与动作差异</a></p>
<p><a href="B_original.mp4">完整原版</a> · <a href="P_cue.mp4">完整仅提醒版</a></p>
<p>这是脚本属性提醒对照，未执行局部修复，也没有反馈决策或自动验收。共两次生成，预定的 R/RP 条件未满足后关闭预算；不另换场景或种子。全部 18 个预定时刻均保留，逐帧静态抽样不等于连续运动评测。旧年龄漂移和 Ava 未按要求离场等问题仍然存在。</p></article>
<article><h2>全部固定采样画面</h2>''' + ''.join(
    f'<img loading="lazy" src="baseline_review/fixed_views_{i}.png" alt="原版与提醒版固定画面 {i}">'
    for i in (1, 2, 3)) + '''</article>
<p><a href="../../results/native_ns235_continuation_cue_v1/developer_observations.json">观察与限制</a> ·
<a href="../../results/native_ns235_continuation_cue_v1/summary.json">次数和成本</a> ·
<a href="../../reproduction/native_ns235_continuation_cue/README.md">方法与复现</a> ·
<a href="../native_kai_silver_hair_v1/index.html">另一场景：固定目标局部修复及后续复发</a> ·
<a href="../historical_repair_showcase_20260917/index.html">历史 5B 实例（不同版本）</a></p></html>''')
records.append({'path': str(page.relative_to(T)), 'source': None,
                'sha256': sha(page), 'bytes': page.stat().st_size})
manifest = T / 'results' / NAME / 'export_manifest.json'
manifest.write_text(json.dumps({'phase': 'closed_cue_control_only_no_local_repairs',
    'exported_unix': time.time(), 'files': records, 'human_labels': 0,
    'raw_latents_and_weights_in_git': False}, indent=2) + '\n')
paths = [row['path'] for row in records] + [str(manifest.relative_to(T)), 'README.md']
(R / 'native_ns235_cue_export_paths.json').write_text(json.dumps(paths, indent=2) + '\n')
print(json.dumps({'bound_files': len(records), 'bytes': sum(row['bytes'] for row in records)}))
