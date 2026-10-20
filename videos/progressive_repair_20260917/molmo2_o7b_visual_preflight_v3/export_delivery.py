#!/usr/bin/env python3
import argparse, hashlib, html, json, shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
files, paths = {}, []
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = [f'molmo2_o7b_provision_v{i}' for i in [1, 2, 3]] + [f'molmo2_o7b_visual_preflight_v{i}' for i in [1, 2, 3]]
for name in names:
    for kind in ['results', 'artifacts']:
        assert not (target / kind / name).exists(), name
for name in names:
    folder = root.parent / name
    sources = [p for p in folder.iterdir() if p.is_file()]
    if name == root.name:
        for part in ['frames', 'telemetry']:
            sources += [p for p in (folder / part).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for source in sorted(sources):
        assert source.suffix in ['.json', '.jsonl', '.py', '.txt', '.png']
        # Keep logs but reject publication if an unexpected credential-bearing URL appears.
        if source.suffix in ['.txt', '.json', '.jsonl']:
            content = source.read_text()
            assert not any(marker in content for marker in ['X-Amz-Credential=', 'X-Amz-Signature=', 'Authorization: Bearer', 'hf_']), source
        rel = source.relative_to(folder)
        dest = target / ('artifacts' if source.suffix == '.png' else 'results') / name / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        key = str(dest.relative_to(target)); paths.append(key)
        files[name + '/' + str(rel)] = {'repository_path': key, 'bytes': dest.stat().st_size, 'sha256': sha(dest)}
responses = json.loads((root / 'responses.json').read_text())
page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Molmo2 两次视觉预检</title><style>body{font:17px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:0 20px}img{width:100%;height:auto}pre{white-space:pre-wrap;background:#f3f4f6;padding:18px}</style><h1>两次 Molmo2-O-7B 看图预检</h1>',
    '<p>一次模型加载，两次固定问题，28.716 秒。只证明本地调用可用，尚未成为可靠验收器。无脚本要求、修复意图、候选选择、正式评分或人工标签。</p>',
    '<p><a href="../../results/' + root.name + '/summary.json">成本和局限</a> · <a href="../../reproduction/molmo2_o7b_preflight/README.md">模型版本与复现</a> · <a href="../native_development_pilot12_batch2_v1/index.html">完整原版视频</a></p>',
    '<h2>全部十张实际输入</h2><img src="actual_inputs_montage.png"><p>每条视频的 0 / 2.5 / 5 / 7.5 / 9.9375 秒；未提供后续镜头。</p>',
    '<h2>固定问题</h2><pre>' + html.escape(json.loads((root / 'plan.json').read_text())['question']) + '</pre>']
for row in responses:
    page.append('<h2>' + row['case'] + ' · 原始回复</h2><pre>' + html.escape(row['response']) + '</pre>')
page.append('<p>人数和姿态与这些输入大致一致；杯子可见但并非手持，衣服颜色描述较粗，静帧也不能证明间隔内始终没有变化。没有以这些回复挑选修复候选或计算成功率。本轮两次调用预算已关闭。</p></html>')
dest = target / 'artifacts' / root.name / 'index.html'
dest.write_text('\n'.join(page) + '\n')
key = str(dest.relative_to(target)); paths.append(key)
files['gallery'] = {'repository_path': key, 'bytes': dest.stat().st_size, 'sha256': sha(dest)}
dest = target / 'reproduction/molmo2_o7b_preflight/README.md'
key = str(dest.relative_to(target)); paths.append(key)
files['reproduction'] = {'repository_path': key, 'bytes': dest.stat().st_size, 'sha256': sha(dest)}
manifest = {'files': files, 'weights_and_caches_in_git': False,
    'scope': 'One successful model load, two fixed observations. Two earlier dependency aborts made zero calls. No scoring, candidate selection, human labels or repair efficacy.'}
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps(manifest, indent=2) + '\n'); paths.append(str(dest.relative_to(target)))
(root.parent / 'molmo2_o7b_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'files': len(paths), 'manifest_files': len(files), 'bytes': sum(row['bytes'] for row in files.values())}))
