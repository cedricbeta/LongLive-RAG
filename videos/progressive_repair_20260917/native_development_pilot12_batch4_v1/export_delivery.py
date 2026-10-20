#!/usr/bin/env python3
"""Export this completed batch once to fresh repository paths."""
import argparse, hashlib, html, json, shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
source, target = root.parents[2], a.target.resolve()
files, paths = {}, []
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for kind in ['artifacts', 'results']:
    assert not (target / kind / root.name).exists(), 'Refuse to overwrite a prior export'
for path in sorted(root.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pt':
        continue
    rel = path.relative_to(root)
    dest = target / ('artifacts' if path.suffix in ('.mp4', '.png') else 'results') / root.name / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    name = str(dest.relative_to(target))
    paths.append(name)
    files[str(rel)] = {'repository_path': name, 'sha256': sha(dest), 'bytes': dest.stat().st_size}
existing_preparer = target / 'implementation/scripts/prepare_native_pilot_followup.py'
assert sha(existing_preparer) == sha(source / 'scripts/prepare_native_pilot_followup.py')
plan = json.loads((root / 'plan.json').read_text())
observations = json.loads((root / 'developer_observations.json').read_text())
blurbs = {key: value['gallery_blurb'] for key, value in observations['cases'].items()}

page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>开发 pilot 第四批原版</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#172735;background:#f7f8fa}video,img{width:100%;height:auto}article{margin:28px 0;padding:20px;background:white;border:1px solid #ddd;border-radius:10px}</style><h1>12 案例开发 pilot：第四批原版</h1>',
    '<p>官方 LongLive-RAG 1.3B，同一套时序适配、原始脚本与 seed 0。每条 60 秒。冻结的 12 条开发脚本现已完成 8 条原版，另 4 条尚未启动。本批修复、策略对照和质量验收均为 0。</p>',
    '<p>下一批选择在生成前沿用原冻结场景顺序，未用输出质量替换案例；人物注册表只用于选样。下文是开发者固定静帧观察，不是独立评估或人工标签。</p>',
    '<p><a href="../../results/' + root.name + '/summary.json">记录与成本</a> · <a href="../../reproduction/native_development_pilot12_batch4/README.md">复现</a> · <a href="../native_development_pilot12_v1/index.html">首批两条原版</a></p>']
for name in ['ns100', 'ns110']:
    case = next(c for c in plan['cases'] if c['id'] == name)
    page.append('<article><h2>' + html.escape(name + ' · 原版 · ' + case['scene']) + '</h2><video controls preload="metadata" src="' + name + '/baseline/videos/rank0-0-0_lora.mp4"></video><p>' + blurbs[name] + '</p><details><summary>完整六段原文</summary><ol>')
    page.extend('<li>' + html.escape(text) + '</li>' for text in case['prompts'])
    page.append('</ol></details><details><summary>全部 18 个固定时间点</summary><img loading="lazy" src="' + name + '/baseline/sample_frames/all_fixed_views.png"></details></article>')
page.append('<p>这些视频不证明纠错收益。最新官方自动 progressive 仍为零保留；历史 5B 的局部收益另列。</p></html>')
dest = target / 'artifacts' / root.name / 'index.html'
dest.write_text('\n'.join(page) + '\n')
relative = str(dest.relative_to(target)); paths.append(relative)
files['gallery'] = {'repository_path': relative, 'sha256': sha(dest), 'bytes': dest.stat().st_size}
for name in ['prepare.py', 'README.md']:
    dest = target / 'reproduction/native_development_pilot12_batch4' / name
    assert dest.exists()
    relative = str(dest.relative_to(target)); paths.append(relative)
    files['reproduction/' + name] = {'repository_path': relative, 'sha256': sha(dest), 'bytes': dest.stat().st_size}
manifest = {'runtime_root': str(root), 'files': files, 'raw_latents_not_in_git': True, 'reused_preparer': {'repository_path': str(existing_preparer.relative_to(target)), 'sha256': sha(existing_preparer)},
    'latent_provenance': 'Full tensors checked at runtime with recorded hashes and dimensions. Regenerate to recover tensors.',
    'scope': 'Fourth two originals from twelve preselected exposed scripts: eight originals total, no repair efficacy or completed strategy comparison.'}
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps(manifest, indent=2) + '\n'); paths.append(str(dest.relative_to(target)))
(root.parent / 'native_pilot_batch4_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'files': len(paths), 'manifest_files': len(files), 'bytes': sum(v['bytes'] for v in files.values())}))
