#!/usr/bin/env python3
"""Publish the closed diagnostic pair, all actual inputs and raw responses."""
import argparse, hashlib, html, json, re, shutil
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('--target', type=Path, required=True); a = p.parse_args()
root, target = Path(__file__).resolve().parent, a.target.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = read(root / 'summary.json')
assert summary['status'] == 'closed_both_arms_unknown_after_contradictory_recheck'
assert read(root / 'queue_status.json')['status'] == 'completed'
for kind in ['results', 'artifacts']:
    assert not (target / kind / root.name).exists()
required = set()
for source in root.glob('*/run/vlm_cache/*/*.json'):
    required.update(Path(x).resolve() for x in read(source)['ordered_image_paths'])
actual_inputs = len(required)
required.update(p.resolve() for p in root.glob('*/run/backend/frames/**/*.png'))
files, paths, omitted = {}, [], []
def bind(path):
    key = str(path.relative_to(target)); assert key not in files
    files[key] = {'sha256': sha(path), 'bytes': path.stat().st_size}; paths.append(key)
for source in sorted(root.rglob('*')):
    if not source.is_file() or '__pycache__' in source.parts or '.call_locks' in source.parts or source.suffix == '.lock':
        continue
    if source.suffix in ['.png', '.jpg'] and source.resolve() not in required:
        omitted.append({'path': str(source.relative_to(root)), 'bytes': source.stat().st_size, 'sha256': sha(source)})
        continue
    assert source.suffix in ['.py', '.json', '.jsonl', '.yaml', '.txt', '.log', '.pid', '.png', '.jpg', '.pt'] or source.name in ['STOP', 'EXITED'], source
    if source.suffix == '.pt':
        assert source.name == 'repaint_masks.pt' and source.stat().st_size < 10 * 1024**2
    if source.suffix in ['.json', '.jsonl', '.txt', '.log']:
        assert not re.search(r'\bhf_[A-Za-z0-9]{20,}|X-Amz-(?:Signature|Credential)=|Authorization: Bearer ', source.read_text()), source
    dest = target / 'results' / root.name / source.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest); bind(dest)
dest = target / 'results' / root.name / 'omitted_derived_frames.json'
dest.write_text(json.dumps({'reason': 'Redundant decoded frames are recoverable from the published original; every actual VLM input, backend evidence frame and saved person mask is included.', 'files': omitted}, indent=2) + '\n'); bind(dest)
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ns270 自动对照：矛盾复核后停止，零修复</title>
<style>body{font:17px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}video{width:100%}.inputs{display:flex;gap:16px;flex-wrap:wrap}.inputs img{max-width:30%;object-fit:contain}pre{white-space:pre-wrap;background:#f3f4f6;padding:16px}</style>
<h1>ns270：两组均保留 unknown，零修复</h1>
<p>官方 LongLive-RAG 1.3B 原片，自动检查 10–20 秒。反馈与固定噪声两组各做 12 次模型调用，均未生成候选，原片按未解决交付。两种修复动作的差异没有执行；不支持反馈收益或修复成功。</p>
<p><a href="../../reproduction/native_ns270_conditional_pair/README.md">协议与复现</a> · <a href="../../results/native_ns270_conditional_pair_v1/summary.json">结果与实际成本</a> · <a href="../native_development_pilot12_batch5_v1/index.html">两条新原片与全部固定视图</a></p>
<video controls preload="metadata" src="../native_development_pilot12_batch5_v1/ns270/baseline/videos/rank0-0-0_lora.mp4"></video>
<p>原片在 10 秒抽帧仍是直棕发，15 秒和 19.94 秒已是红卷发。这是原生提示切换的输出。首段未明确发色，不能把更早的外观自动算错；抽帧不能给出精确转变时长。</p>
<h2>联合复核实际收到的三张图</h2><div class="inputs">
'''
for frame in summary['arms'][0]['attributes'][0]['recheck']['frames']:
    path = Path(frame['path']); assert sha(path) == frame['sha256']
    rel = path.relative_to(root)
    page += '<img alt="实际输入 ' + html.escape(path.name) + '" src="../../results/' + root.name + '/' + html.escape(str(rel)) + '">'
page += '</div><p>逐帧判断对卷曲和红色都给出 1 fail、2 pass。随后的单次联合复核将后两张卷发也描述为直发，又将“多数帧红色”判为通过。程序检测到与先前证据冲突，两项保留 unknown；原始回复未重写。</p>'
for arm in summary['arms']:
    page += '<h2>' + html.escape(arm['arm']) + ' · 原始联合复核</h2>'
    for attribute in arm['attributes']:
        view = {k: attribute['recheck'][k] for k in ['status', 'observed', 'reason']}
        page += '<h3>' + html.escape(attribute['aspect']) + '</h3><pre>' + html.escape(json.dumps(view, ensure_ascii=False, indent=2)) + '</pre>'
page += '<p>两组共有 24 次模型调用；队列 355.53 秒。SAM2 人物区域已保存，但没有 CLIPSeg 部位分割、候选生成、成对保护或后续验收。四个未用候选名额均已关闭，不在本例追加提示词调参、窗口替换或固定目标生成。没有独立评分、人评或保留改进。</p></html>\n'
dest = target / 'artifacts' / root.name / 'index.html'; dest.parent.mkdir(parents=True); dest.write_text(page); bind(dest)
bind(target / 'reproduction/native_ns270_conditional_pair/README.md')
parent = target / 'artifacts/native_development_pilot12_batch5_v1/ns270/baseline/videos/rank0-0-0_lora.mp4'
assert sha(parent) == summary['arms'][0]['selected_original_sha256']
dest = target / 'results' / root.name / 'export_manifest.json'
dest.write_text(json.dumps({'files': files, 'actual_vlm_input_count': actual_inputs, 'exported_evidence_images': len(required),
    'omitted_redundant_frames': len(omitted), 'reused_original': {'repository_path': str(parent.relative_to(target)), 'sha256': sha(parent)},
    'new_video_candidates': 0, 'scope': 'Automatic diagnostic abstention, not repair efficacy.'}, indent=2) + '\n')
paths.append(str(dest.relative_to(target)))
(root.parent / 'native_ns270_automatic_export_paths.json').write_text(json.dumps(sorted(paths), indent=2) + '\n')
print(json.dumps({'paths': len(paths), 'bound_bytes': sum(x['bytes'] for x in files.values()), 'actual_vlm_inputs': actual_inputs}))
