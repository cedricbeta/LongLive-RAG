#!/usr/bin/env python3
"""Export the closed preflight and zero-candidate attempt with actual visual inputs."""
import argparse, hashlib, html, json, re, shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--target', type=Path, required=True)
a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(root / 'summary.json')['status'] == 'closed_no_candidates_control_not_started'
assert read(root / 'queue_status.json')['status'] == 'completed'
paths, files, omitted = [], {}, []
required_images = set()
for record in (root / 'feedback/run/vlm_cache').rglob('*.json'):
    required_images.update(Path(x).resolve() for x in read(record)['ordered_image_paths'])
actual_vlm_image_count = len(required_images)
required_images.update(x.resolve() for x in (root / 'feedback/run/backend/frames').rglob('*.png'))
required_images.add((root / 'actual_assignment_inputs.jpg').resolve())

def bind(dest, source=None):
    key = str(dest.relative_to(target))
    assert key not in paths
    paths.append(key)
    files[key] = {'sha256': sha(dest), 'bytes': dest.stat().st_size}
    if source is not None:
        files[key]['source_relative_to_experiment_root'] = str(source.relative_to(root.parent))

for name in ['native_conditional_names_v7_preflight_v1', 'native_ns300_conditional_pair_v1', root.name]:
    folder = root.parent / name
    for source in sorted(folder.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts or '.call_locks' in source.parts or source.suffix == '.lock':
            continue
        if source.suffix in ['.jpg', '.png'] and source.resolve() not in required_images:
            omitted.append({'path': str(source.relative_to(root.parent)), 'sha256': sha(source), 'bytes': source.stat().st_size})
            continue
        assert source.suffix in ['.py', '.json', '.jsonl', '.yaml', '.txt', '.log', '.pid', '.md', '.png', '.jpg'] or source.name in ['STOP', 'EXITED'], source
        if source.suffix in ['.json', '.jsonl', '.txt', '.log', '.md']:
            assert not re.search(r'\bhf_[A-Za-z0-9]{20,}|X-Amz-(?:Signature|Credential)=|Authorization: Bearer ', source.read_text()), source
        dest = target / 'results' / name / source.relative_to(folder)
        assert not dest.exists(), dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest); bind(dest, source)

implementation_paths = read(root.parent / 'conditional_names_v7_implementation_paths.json')
for name in implementation_paths + ['reproduction/conditional_names_v7/README.md', 'reproduction/native_ns300_conditional_pair/README.md', 'README.md']:
    bind(target / name)

dest = target / 'results' / root.name / 'omitted_derived_frames.json'
dest.write_text(json.dumps({'reason': 'Redundant decoded PNG/JPEG frames remain local and are recoverable from the linked original video. All actual ordered VLM image inputs and the three backend evidence frames are included.', 'files': omitted}, indent=2) + '\n')
bind(dest)
summary = read(root / 'summary.json')
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ns300 自动尝试：零候选</title><style>body{font:17px/1.65 system-ui;max-width:1050px;margin:30px auto;padding:0 20px}video,img{max-width:100%}pre{white-space:pre-wrap;background:#f3f4f6;padding:18px}td,th{padding:10px;border-bottom:1px solid #ddd}</style>
<h1>ns300 自动尝试：人物对应不确定，零候选</h1>
<p>反馈组完成两次文字和六次看图调用，人物对应全部不确定，交付原片。固定噪声组在 GPU 空闲检查阶段失败，未加载模型。因此没有有效成对效果比较，也没有修复成功案例。</p>
<p><a href="../../reproduction/native_ns300_conditional_pair/README.md">协议、失败与复现</a> · <a href="../../results/native_ns300_conditional_pair_v2/summary.json">全部成本</a> · <a href="../../reproduction/conditional_names_v7/README.md">四次文字预检</a></p>
<h2>完整原片 · 官方 LongLive-RAG · 60 秒</h2>
<video controls preload="metadata" src="../native_development_pilot12_batch3_v1/ns300/baseline/videos/rank0-0-0_lora.mp4"></video>
<p>本次只请求诊断 10–20 秒。原文要求 Luna 加入，Clara 将钥匙交给 Miles；实际后两张诊断输入中可见两名女性。未将外观直接当作身份真值，也未手改模型结果。</p>
<h2>检查器实际看到的三张编号图</h2>
<img loading="lazy" alt="三张实际人物分配输入，保留程序编号和边框" src="../../results/native_ns300_conditional_pair_v2/actual_assignment_inputs.jpg">
<p>这些是实际诊断输入的排版展示，没有修复候选。各调用原始 PNG、JSON 和哈希均保留。下列 uncertain 不代表原片正确。</p>
'''
for row in summary['arms'][0]['assignment_replies']:
    page += '<details><summary>完整人物分配回复</summary><pre>' + html.escape(json.dumps(row['reply'], indent=2, ensure_ascii=False)) + '</pre></details>'
page += '<p>这轮已经关闭：零新视频、零保留、零人评、零独立分数。GPU 同机加载已执行，实际分割与跟踪未执行。历史 5B 演示不用于证明此官方版本有效。</p></html>\n'
dest = target / 'artifacts' / root.name / 'index.html'
dest.parent.mkdir(parents=True, exist_ok=False); dest.write_text(page); bind(dest)
parent = target / 'artifacts/native_development_pilot12_batch3_v1/ns300/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == summary['arms'][0]['delivered_original_sha256']
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps({'files': files, 'reused_original': {'repository_path': str(parent.relative_to(target)), 'sha256': sha(parent)}, 'omitted_derived_frame_count': len(omitted), 'actual_vlm_image_inputs_included': actual_vlm_image_count, 'exported_image_count': len(required_images), 'model_weights_or_latents_in_git': False}, indent=2) + '\n')
paths.append(str(dest.relative_to(target)))
(root.parent / 'conditional_names_ns300_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'paths': len(paths), 'bound_files': len(files), 'bytes': sum(x['bytes'] for x in files.values()), 'omitted_redundant_frames': len(omitted)}))
