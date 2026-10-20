#!/usr/bin/env python3
import argparse, hashlib, html, json, re, shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files, paths = {}, []
for name in ['native_ns200_progressive_pair_v1', root.name]:
    folder = root.parent / name
    for source in sorted(folder.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts or '.call_locks' in source.parts:
            continue
        assert source.suffix in ['.py', '.json', '.jsonl', '.yaml', '.txt', '.log', '.pid'] or source.name in ['STOP', 'EXITED'], source
        if source.suffix in ['.json', '.jsonl', '.txt', '.log']:
            text = source.read_text()
            assert not re.search(r'\bhf_[A-Za-z0-9]{20,}|X-Amz-(?:Signature|Credential)=|Authorization: Bearer ', text), source
        rel = source.relative_to(folder)
        dest = target / 'results' / name / rel
        if dest.exists() and sha(dest) != sha(source):
            assert name == root.name and str(rel) == 'export_delivery.py'
            assert sha(dest) == sha(root / 'export_first_attempt.py')
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
        key = str(dest.relative_to(target)); paths.append(key)
        files[name + '/' + str(rel)] = {'repository_path': key, 'sha256': sha(dest), 'bytes': dest.stat().st_size}
for name in ['implementation/scripts/run_native_progressive_shared_gpu.py',
             'implementation/scripts/run_native_fixed_noise_shared_gpu.py',
             'implementation/utils/agent_gpu_ownership.py', 'implementation/tests/test_agent_gpu_ownership.py',
             'reproduction/native_ns200_progressive_pair/README.md']:
    dest = target / name; assert dest.exists(); paths.append(name)
    files[name] = {'repository_path': name, 'sha256': sha(dest), 'bytes': dest.stat().st_size}
summary = json.loads((root / 'summary.json').read_text())
page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ns200 自动对照：编译失败</title><style>body{font:17px/1.6 system-ui;max-width:1050px;margin:30px auto;padding:0 20px}video{width:100%}pre{white-space:pre-wrap;background:#f3f4f6;padding:18px}td,th{padding:10px;border-bottom:1px solid #ddd}</style><h1>ns200 自动对照：两边均在文字编译阶段失败</h1>',
    '<p>两次真实文字调用，零看图调用，零候选视频。反馈策略和固定局部重试尚未执行到差异所在，因此不能比较纠错效果；原版作为回退引用保留。</p>',
    '<p><a href="../../results/' + root.name + '/summary.json">成本和完整记录</a> · <a href="../../reproduction/native_ns200_progressive_pair/README.md">协议与复现</a></p>',
    '<h2>完整原版 · 官方 LongLive-RAG · 60 秒</h2><video controls preload="metadata" src="../native_development_pilot12_batch3_v1/ns200/baseline/videos/rank0-0-0_lora.mp4"></video>',
    '<p>首段外观基本符合原文，后段有服装与人物变化。该原版的开发者观察另见<a href="../native_development_pilot12_batch3_v1/index.html">第三批原版索引</a>；这些观察没有改变事前固定的案例、首段检查窗口或尝试上限。</p>',
    '<table><tr><th>预定策略</th><th>文字调用</th><th>候选</th><th>完整生命周期</th></tr>']
for row in summary['arms']:
    page.append(f"<tr><td>{html.escape(row['arm'])}</td><td>1</td><td>0</td><td>{row['arm_lifecycle_seconds']:.3f} 秒</td></tr>")
page.append('</table><p>两边原始回复逐字相同：将 They/She 判为非角色，同时又赋予它们必须出现的段落。现有编译器拒绝这一矛盾，分割、生成、成对保护及后续验证都未发生。</p>')
for arm in ['feedback', 'fixed_noise']:
    call, = (root / arm / 'run/vlm_cache').rglob('*.json')
    parsed = json.loads(call.read_text())['parsed']
    page.append('<details><summary>' + arm + ' · 完整解析回复</summary><pre>' + html.escape(json.dumps(parsed, indent=2, ensure_ascii=False)) + '</pre></details>')
page.append('<p>首次监测启动失败也保留；当时模型未加载。第二次仅修复监测准备，实验计划保持原字节。本案例已经关闭，不补造对照或成功率；最新官方自动 progressive 仍为零保留。</p></html>')
dest = target / 'artifacts' / root.name / 'index.html'
dest.parent.mkdir(parents=True, exist_ok=False); dest.write_text('\n'.join(page) + '\n')
key = str(dest.relative_to(target)); paths.append(key)
files['gallery'] = {'repository_path': key, 'sha256': sha(dest), 'bytes': dest.stat().st_size}
parent = target / 'artifacts/native_development_pilot12_batch3_v1/ns200/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == summary['arms'][0]['fallback_reference_video_sha256']
manifest = {'files': files, 'reused_parent': {'repository_path': str(parent.relative_to(target)), 'sha256': sha(parent)},
    'model_weights_or_latents_in_git': False, 'scope': 'Two compiler failures, zero visual calls, candidates, independent scores or retained improvements. Actual policy difference was not exercised.'}
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps(manifest, indent=2) + '\n'); paths.append(str(dest.relative_to(target)))
(root.parent / 'native_ns200_pair_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'files': len(paths), 'manifest_files': len(files), 'bytes': sum(row['bytes'] for row in files.values())}))
