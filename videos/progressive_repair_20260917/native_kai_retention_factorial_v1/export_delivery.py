"""Deliver the bounded retention factorial; earlier source bundles remain intact."""
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import time

S = Path('/home/chendong/video-gen/LongLive-RAG')
T = S.parent / 'ICLR27-Longlive-agent'
R = S / 'videos/progressive_repair_20260917'
F = Path(__file__).resolve().parent
NAME = 'native_kai_retention_factorial_v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(F / 'finalize_queue_status.json')['phase'] == 'completed'
assert read(F / 'summary.json')['remaining_generation_budget'] == 0
assert read(F / 'pair_check.json')['status'] == 'passed'
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
    # rglob does not traverse directory symlinks; archived B/R are referenced
    # below and must never be rewritten, rebound or silently duplicated.
    rel = src.relative_to(F)
    if rel.parts[0] in ('B_original', 'R_local'):
        continue
    if not src.is_file() or '__pycache__' in src.parts:
        continue
    if src.suffix in ('.json', '.yaml', '.txt', '.py', '.log'):
        copy(src, Path('results') / NAME / rel)
    elif src.suffix == '.png':
        copy(src, Path('artifacts') / NAME / rel)
for arm in ('P_cue', 'RP_local_cue'):
    copy(next((F / arm / 'videos').glob('*.mp4')), Path('artifacts') / NAME / (arm + '.mp4'))
    telemetry = R / f'native_kai_retention_{arm}_v1_telemetry'
    assert read(telemetry / 'status.json')['status'] == 'completed'
    for src in sorted(telemetry.iterdir()):
        if src.is_file():
            copy(src, Path('results') / telemetry.name / src.name)
copy(F / 'four_way_review/full_60s_comparison.mp4', Path('artifacts') / NAME / 'full_60s_comparison.mp4')
for name in ('prepare_native_retention_pair.py', 'audit_native_retention_cue.py',
             'audit_native_retention_pair.py', 'review_native_retention_fourway.py'):
    copy(S / 'scripts' / name, Path('implementation/scripts') / name)
copy(F / 'METHOD_AND_REPRODUCTION.md', Path('reproduction/native_kai_retention_factorial/README.md'))
obs = read(F / 'developer_observations.json')
page = T / 'artifacts' / NAME / 'index.html'
page.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kai 教室：局部修复 × 后续属性提醒</title>
<style>body{font:17px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f6f8fa;color:#19303d}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e5}video,img{width:100%;height:auto}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style>
<h1>Kai 教室：局部银发修复能否延续</h1>
<p>官方 LongLive-RAG 1.3B，完整 60 秒，960 帧。上排：原版 B／仅后续提醒 P；下排：已有局部修复 R／相同修复＋提醒 RP。B、R 复用既有原片；仅新生成 P、RP 各一次。</p>
<p class="note">''' + escape(obs['summary_zh']) + '''</p>
<article><video controls preload="metadata" src="full_60s_comparison.mp4"></video>
<p>快速查看：<a href="full_60s_comparison.mp4#t=20">20 秒：修复开始</a> · <a href="full_60s_comparison.mp4#t=25">25 秒：局部变化</a> · <a href="full_60s_comparison.mp4#t=35">35 秒：提醒后的变化</a> · <a href="full_60s_comparison.mp4#t=45">45 秒：后续保持与副作用</a></p>
<p>完整单片：<a href="../native_sampler_cases_v1/ns280_baseline.mp4">B 原版</a> · <a href="P_cue.mp4">P 仅提醒</a> · <a href="../native_kai_silver_hair_v1/candidate.mp4">R 仅局部修复</a> · <a href="RP_local_cue.mp4">RP 修复＋提醒</a></p>
<p>提醒仅在 30／40／50 秒边界追加原脚本已有的“Kai has silver hair.”，没有增加新属性。20–30 秒的局部 mask、提示词、噪声、原生采样器均沿用已有 R。两组成对前 121 个 latent 完全一致。该案例是在看过既有结果后选定的开发对照；不是自动反馈成功、留出评测、同预算 best-of-N 或独立人工验收。</p>
<p>P 的提醒从 30 秒开始，因此不能据此比较“局部修复”与“20 秒时直接改写整段提示词”的优劣。年龄、身份、其他人物、动作和画质问题必须单独保留；发色改善不等于全片正确。</p></article>
<article><h2>全部 18 个预定时刻</h2>''' + ''.join(
    f'<img loading="lazy" src="four_way_review/fixed_views_{i}.png" alt="四种条件固定样本 {i}">'
    for i in range(1, 7)) + '''</article>
<p><a href="../../results/native_kai_retention_factorial_v1/developer_observations.json">观察与限制</a> ·
<a href="../../results/native_kai_retention_factorial_v1/summary.json">全部次数和成本</a> ·
<a href="../../results/native_kai_retention_factorial_v1/pair_check.json">实际条件和前段核对</a> ·
<a href="../../reproduction/native_kai_retention_factorial/README.md">方法与复现</a> ·
<a href="../native_ns235_continuation_cue_v1/index.html">另一场景的仅提醒对照</a> ·
<a href="../historical_repair_showcase_20260917/index.html">历史 5B 实例（不同版本）</a></p></html>''')
records.append({'path': str(page.relative_to(T)), 'source': None,
                'sha256': sha(page), 'bytes': page.stat().st_size})
manifest = T / 'results' / NAME / 'export_manifest.json'
manifest.write_text(json.dumps({'phase': 'completed_closed_exposed_retention_factorial',
    'exported_unix': time.time(), 'files': records, 'human_labels': 0,
    'raw_latents_and_weights_in_git': False}, indent=2) + '\n')
paths = [row['path'] for row in records] + [str(manifest.relative_to(T)), 'README.md']
(R / 'native_kai_retention_export_paths.json').write_text(json.dumps(paths, indent=2) + '\n')
print(json.dumps({'bound_files': len(records), 'bytes': sum(row['bytes'] for row in records)}))
