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
NAME = 'native_kai_same_boundary_control_v1'
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
    # rglob does not traverse directory symlinks; archived B/P30/RP are referenced
    # below and must never be rewritten, rebound or silently duplicated.
    rel = src.relative_to(F)
    if rel.parts[0] in ('B_original', 'P30_prompt', 'RP_local_cue'):
        continue
    if not src.is_file() or '__pycache__' in src.parts:
        continue
    if src.suffix in ('.json', '.yaml', '.txt', '.py', '.log'):
        copy(src, Path('results') / NAME / rel)
    elif src.suffix == '.png':
        copy(src, Path('artifacts') / NAME / rel)
for arm in ('P20_prompt',):
    copy(next((F / arm / 'videos').glob('*.mp4')), Path('artifacts') / NAME / (arm + '.mp4'))
    telemetry = R / 'native_kai_same_boundary_P20_v1_telemetry'
    assert read(telemetry / 'status.json')['status'] == 'completed'
    for src in sorted(telemetry.iterdir()):
        if src.is_file():
            copy(src, Path('results') / telemetry.name / src.name)
copy(F / 'four_way_review/full_60s_comparison.mp4', Path('artifacts') / NAME / 'full_60s_comparison.mp4')
for name in ('prepare_native_prompt_start_control.py', 'audit_native_prompt_start_control.py',
             'review_native_prompt_start_control.py'):
    copy(S / 'scripts' / name, Path('implementation/scripts') / name)
copy(F / 'REPRODUCTION.md', Path('reproduction/native_kai_same_boundary_control/README.md'))
obs = read(F / 'developer_observations.json')
page = T / 'artifacts' / NAME / 'index.html'
page.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kai：同为20秒起点的普通提示词对照</title>
<style>body{font:17px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f6f8fa;color:#19303d}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e5}video,img{width:100%;height:auto}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style>
<h1>Kai：普通提示词提醒是否也能及时修正</h1>
<p>官方 LongLive-RAG 1.3B，完整60秒。上排：原版 B／从30秒开始提醒的 P30；下排：新增从20秒开始提醒的 P20／已有局部修复＋后续提醒 RP。只新生成 P20 一次，其余原片复用且未改写。</p>
<p class="note">''' + escape(obs['summary_zh']) + '''</p>
<article><video controls preload="metadata" src="full_60s_comparison.mp4"></video>
<p>快速查看：<a href="full_60s_comparison.mp4#t=20">20秒：同一起点</a> · <a href="full_60s_comparison.mp4#t=25">25秒：发色与非目标变化</a> · <a href="full_60s_comparison.mp4#t=35">35秒：后续保持</a> · <a href="full_60s_comparison.mp4#t=45">45秒：动作与画质</a></p>
<p>完整单片：<a href="../native_sampler_cases_v1/ns280_baseline.mp4">B 原版</a> · <a href="../native_kai_retention_factorial_v1/P_cue.mp4">P30</a> · <a href="P20_prompt.mp4">P20</a> · <a href="../native_kai_retention_factorial_v1/RP_local_cue.mp4">RP</a></p>
<p>P20 从20秒开始在原始提示词后追加“Kai has silver hair.”，该段文本与 RP 的私有修复提示词相同；30／40／50秒的实际提醒相同。P20 无 mask、无额外私有去噪。四种条件前81个 latent 完全一致。</p>
<p>这是看过旧结果后冻结的一次开发对照，不是自动诊断、反馈策略或独立人工验收。原有年龄漂移、Ava未离场等错误单独保留。发色修正不能直接当作身份、动作或全片质量通过；完整片段和全部预定采样均保留。</p></article>
<article><h2>全部18个预定时刻</h2>''' + ''.join(
    f'<img loading="lazy" src="four_way_review/fixed_views_{i}.png" alt="同起点对照固定样本 {i}">'
    for i in range(1, 7)) + '''</article>
<p><a href="../../results/native_kai_same_boundary_control_v1/developer_observations.json">逐项观察与限制</a> ·
<a href="../../results/native_kai_same_boundary_control_v1/summary.json">次数与成本</a> ·
<a href="../../results/native_kai_same_boundary_control_v1/pair_check.json">实际条件核对</a> ·
<a href="../../reproduction/native_kai_same_boundary_control/README.md">方法与复现</a> ·
<a href="../native_kai_retention_factorial_v1/index.html">前一组局部修复与后续提醒对照</a> ·
<a href="../historical_repair_showcase_20260917/index.html">历史5B实例（不同版本）</a></p></html>''')
records.append({'path': str(page.relative_to(T)), 'source': None,
                'sha256': sha(page), 'bytes': page.stat().st_size})
manifest = T / 'results' / NAME / 'export_manifest.json'
manifest.write_text(json.dumps({'phase': 'completed_closed_same_boundary_prompt_control',
    'exported_unix': time.time(), 'files': records, 'human_labels': 0,
    'raw_latents_and_weights_in_git': False}, indent=2) + '\n')
paths = [row['path'] for row in records] + [str(manifest.relative_to(T)), 'README.md']
(R / 'native_kai_same_boundary_export_paths.json').write_text(json.dumps(paths, indent=2) + '\n')
print(json.dumps({'bound_files': len(records), 'bytes': sum(row['bytes'] for row in records)}))
