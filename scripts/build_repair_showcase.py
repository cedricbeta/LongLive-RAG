#!/usr/bin/env python3
"""Package real, archived repair pairs for viewing; no generation or acceptance changes."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
WIDTH, HEIGHT = 1280, 552
PANEL = (640, 352)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    result = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                  fps=cap.get(cv2.CAP_PROP_FPS),
                  width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return result


def cases():
    automatic = {c['scene'][:5]: c for c in read(ROOT / 'videos/feedback_repair_20260914/comparison.json')['cases']}
    fixed = {c['id']: c for c in read(ROOT / 'videos/machine_entity_repair_20260906/machine_evidence/index.json')['cases']}
    descriptions = [
        dict(id='01_clara_flower_shop', case='ns310', source='automatic',
             title='Clara · 花店外套', target='目标：棕褐色长外套',
             observed='酒红／紫色外套变为棕褐色；另一位角色仍在场。',
             limitation='手部、头发和花瓶运动也有变化；尚无独立身份／动作评估。',
             known='历史自动流程接受第 1 镜头，并完成其有限后续检查。原版与 feedback v1 最终视频相同；不证明反馈增益。'),
        dict(id='02_iris_flower_shop', case='ns090', source='automatic',
             title='Iris · 花店发色', target='目标：红色卷发',
             observed='深色头发变为偏红发；可见局部发色修正。',
             limitation='上衣仍为深色，苔绿外套要求未解决；后续姿态也有变化。',
             known='使用历史 original 最终交付，第 1、3 镜头被当时裁判接受。不是全部脚本属性通过，也不是最新 progressive 结果。'),
        dict(id='03_clara_library', case='ns220', source='fixed',
             title='Clara · 图书馆外套', target='目标：棕黄色长外套',
             observed='灰色衣服变为浅棕黄色外套，后续镜头仍可见该衣色。',
             limitation='站起时机、纸张数量与位置有变化；固定目标实验绕过自动验收。',
             known='单次固定目标修复；无编辑对照复现原片。目标经助手检查，非人工真值，未运行自动保护验收。'),
        dict(id='04_iris_hotel', case='ns190', source='fixed',
             title='Iris · 酒店发色', target='目标：红色卷发',
             observed='银白短发变为红色卷发，早期镜头仍可见绿色外套与电脑。',
             limitation='人物姿态／脸部外观变化；后两镜头卷发和外套仍有漂移。',
             known='单次固定目标修复；无编辑对照复现原片。可展示属性编辑能力，不能证明身份保真或全片成功。'),
        dict(id='05_failure_iris_full_person', case='ns090', source='fixed',
             title='失败对照 · Iris 整人区域修复', target='目标：红色卷发与苔绿外套',
             observed='新增花盆和遮挡，人物边界异常；这些变化延续到后续。',
             limitation='此例不算干净修复；展示属性变化不能自动等同于整体成功。',
             known='同一批五例固定目标实验中的失败，完整保留六镜头。它与上面的自动 ns090 是不同运行。', failure=True),
    ]
    for c in descriptions:
        if c['source'] == 'automatic':
            record = automatic[c['case']]
            c['before'] = record['baseline_video']
            c['after'] = record['arms']['original']['final_delivery_video']
            c['workflow'] = '历史自动纠错 · 本地 5B'
            c['right_label'] = '修复后 / 历史流程交付'
            c['record'] = 'videos/feedback_repair_20260914/comparison.json'
        else:
            record = fixed[c['case']]
            c['before'], c['after'] = record['before'], record['after']
            c['workflow'] = '固定目标能力探针 · 本地 5B · 绕过验收'
            c['right_label'] = '修复候选 / 未经自动验收'
            c['record'] = f'videos/machine_entity_repair_20260906/{c["case"]}/result.json'
        yield c


def canvas(c, shot, slow=False):
    img = Image.new('RGB', (WIDTH, HEIGHT), '#101820')
    d = ImageDraw.Draw(img)
    fonts = {n: ImageFont.truetype(FONT, n) for n in (18, 20, 26)}
    d.text((18, 10), c['title'], font=fonts[26], fill='#f4f5f7')
    d.text((18, 49), c['target'], font=fonts[20], fill='#a4e4ce')
    d.text((560, 16), c['workflow'], font=fonts[18], fill='#c0cbd4')
    mode = '完整首镜头 · 0.5× 慢放' if slow else f'全片原速 · 镜头 {shot + 1}/6'
    d.text((850, 49), mode, font=fonts[18], fill='#e3d9b3')
    d.text((18, 84), '修复前 / baseline', font=fonts[20], fill='#eef2f4')
    d.text((658, 84), c['right_label'], font=fonts[20], fill='#a4e4ce')
    d.text((18, 472), c['observed'], font=fonts[20], fill='#eef2f4')
    d.text((18, 506), c['limitation'], font=fonts[18], fill='#f0cda0')
    for txt, size in [(c['observed'], 20), (c['limitation'], 18)]:
        if d.textlength(txt, font=fonts[size]) > WIDTH - 36:
            raise ValueError(f'Caption too long: {txt}')
    return np.asarray(img)[:, :, ::-1].copy()


def encoder(path, fps=24):
    return subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
        '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def close_encoder(proc):
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')


def render(c, output, overview):
    left, right = Path(c['before']), Path(c['after'])
    before, after = metadata(left), metadata(right)
    if before != after or before['frames'] != 384 or abs(before['fps'] - 24) > 1e-8:
        raise ValueError(f'Unaligned source pair: {c["id"]}, {before}, {after}')
    caps = [cv2.VideoCapture(str(p)) for p in (left, right)]
    movie = output / (c['id'] + '.mp4')
    proc = encoder(movie)
    bases = [canvas(c, i) for i in range(6)]
    slow_base = canvas(c, 0, slow=True)
    try:
        for frame in range(384):
            frames = [cap.read() for cap in caps]
            if not all(ok for ok, _ in frames):
                raise ValueError(f'Unexpected EOF in {c["id"]} at {frame}')
            pixels = np.hstack([cv2.resize(f, PANEL, interpolation=cv2.INTER_AREA) for _, f in frames])
            combined = bases[frame // 64].copy()
            combined[112:464] = pixels
            proc.stdin.write(combined.tobytes())
            if frame == 32:
                cv2.imwrite(str(output / (c['id'] + '.jpg')), combined, [cv2.IMWRITE_JPEG_QUALITY, 93])
            if overview and frame < 64:
                slow = slow_base.copy()
                slow[112:464] = pixels
                overview.stdin.write(slow.tobytes())
                overview.stdin.write(slow.tobytes())
        if any(cap.read()[0] for cap in caps):
            raise ValueError('Source contains unexpected extra frames')
    finally:
        for cap in caps:
            cap.release()
        close_encoder(proc)
    return {**c, 'before': str(left.relative_to(ROOT)), 'after': str(right.relative_to(ROOT)),
            'before_sha256': sha(left), 'after_sha256': sha(right),
            'source_metadata': before, 'video': movie.name, 'video_sha256': sha(movie),
            'poster': c['id'] + '.jpg', 'full_source_frame_range': [0, 384],
            'display_transform': 'Same 0.5 spatial resize on both full frames; no crop, color adjustment or interpolation',
            'independent_quality_evaluation': False}


def write_page(output, rows):
    esc = html.escape
    cards = []
    for c in rows:
        cards.append(f'''<article id="{c['id']}">
        <div class="tag">{esc(c['workflow'])}</div><h2>{esc(c['title'])}</h2>
        <p><b>{esc(c['target'])}</b> · {esc(c['observed'])}</p>
        <video controls playsinline preload="metadata" poster="{c['poster']}" src="{c['video']}"></video>
        <p class="limit">{esc(c['limitation'])}</p><p class="small">{esc(c['known'])}</p>
        <p><a href="{c['video']}" download>下载全片对照 · 16 秒</a>
        <button type="button" data-seek="0">首镜头</button>
        <button type="button" data-seek="10.667">后两镜头</button></p></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>真实修复实例 · 2026-09-17</title><style>
    *{box-sizing:border-box}body{margin:0;background:#101820;color:#eaf0f5;font:17px/1.7 system-ui,sans-serif}
    main{max-width:1160px;margin:auto;padding:32px 20px 60px}h1{font-size:34px;line-height:1.3}
    h2{margin:8px 0;font-size:25px}p{margin:12px 0}.lead{color:#c5d3dc;max-width:960px}
    article,.overview{background:#192630;border:1px solid #334550;border-radius:14px;padding:22px;margin:28px 0}
    video{display:block;width:100%;background:#000;border-radius:8px}.tag{color:#a4e4ce;font-size:14px}
    .limit{color:#f0cda0}.small{font-size:15px;color:#b5c3ce}a{color:#a4e4ce;text-underline-offset:4px}
    button{background:#263e4c;color:#eaf0f5;border:1px solid #617987;border-radius:6px;padding:7px 12px;margin-left:12px;cursor:pointer}
    nav a{display:inline-block;margin:6px 16px 6px 0}footer{margin-top:35px;color:#b5c3ce;font-size:15px}
    @media(max-width:600px){main{padding:20px 10px}h1{font-size:28px}article,.overview{padding:12px}button{margin:8px}}
    </style><main><div class="tag">真实归档生成 · 左原片 / 右修复 · 2026-09-17</div>
    <h1>看得见的局部属性修复</h1>
    <p class="lead">四个开发实例展示衣服和头发属性的实际变化。前两例来自历史自动流程，后两例为固定目标能力探针。全部为本地 5B 结果；不代表最新官方 1.3B 渐进反馈已成功。</p>
    <p class="lead">每个全片对照保留 384 帧、六个镜头与原始 24 fps；速览展示完整首镜头并以 0.5× 播放。实例经过挑选，不能据此计算成功率。身份、动作和后续问题见各例说明。</p>
    <nav><a href="#01_clara_flower_shop">花店外套</a><a href="#02_iris_flower_shop">花店发色</a>
    <a href="#03_clara_library">图书馆外套</a><a href="#04_iris_hotel">酒店发色</a><a href="#05_failure_iris_full_person">失败对照</a></nav>
    <section class="overview"><h2>21 秒速览</h2><p class="small">四例完整首镜头 · 0.5× 慢放 · 具体局限始终显示在画面下方</p>
    <video controls playsinline preload="metadata" poster="01_clara_flower_shop.jpg" src="overview_4_examples.mp4"></video>
    <p><a href="overview_4_examples.mp4" download>下载速览</a> · <a href="repair_showcase.zip" download>下载全部视频与说明</a></p></section>
    ''' + '\n'.join(cards) + '''<footer><p>新生成次数：0。直接使用已有真实生成结果，未合成、补帧或修改服装颜色。视频仅等比缩放、并排和加说明。未新增人工评分或独立语义评测。</p>
    <p><a href="manifest.json">来源与视频哈希</a> · <a href="README.md">证据说明与复现命令</a> · <a href="evidence/machine-result-review.json">完整五例固定目标观察（含未入速览者）</a>
    · <a href="evidence/feedback-comparison.json">完整六例自动流程运行记录</a></p></footer></main>
    <script>document.querySelectorAll('button[data-seek]').forEach(b=>b.addEventListener('click',()=>{const v=b.closest('article').querySelector('video');v.currentTime=Number(b.dataset.seek);v.play();}));
    document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause()})));</script></html>'''
    (output / 'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'videos/repair_showcase_20260917')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    overview = encoder(output / 'overview_4_examples.mp4')
    rows = []
    try:
        for c in cases():
            print(f'Packaging {c["id"]}', flush=True)
            rows.append(render(c, output, None if c.get('failure') else overview))
    finally:
        close_encoder(overview)
    evidence = output / 'evidence'
    evidence.mkdir()
    copies = {
        'feedback-comparison.json': 'videos/feedback_repair_20260914/comparison.json',
        'machine-result-review.json': 'docs/machine_entity_repair_20260906/result_review.json',
        'machine-results.zh.md': 'docs/machine-entity-repair-20260906.zh.md',
        'feedback-results.zh.md': 'docs/feedback-repair-v1-results-20260915.zh.md',
    }
    for name, src in copies.items():
        shutil.copyfile(ROOT / src, evidence / name)
    shutil.copyfile(__file__, output / 'build_repair_showcase.py')
    manifest = {'created_local_date': '2026-09-17', 'new_generation_calls': 0,
                'new_model_judge_calls': 0, 'human_labels': 0,
                'selection': 'Known development examples selected for visible local attribute changes; no success-rate claim',
                'generator': 'Local historical LongLive 5B; not official 1.3B progressive results',
                'overview': {'file': 'overview_4_examples.mp4', 'sha256': sha(output / 'overview_4_examples.mp4'),
                             'source_frames_per_case': [0, 64], 'playback_speed': 0.5,
                             'method': 'Each source frame shown twice; no motion interpolation'},
                'cases': rows, 'evidence': {name: sha(evidence / name) for name in copies},
                'packaging_seconds': time.monotonic() - began}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('''# 真实修复实例（开发展示）

打开 index.html，或直接播放 overview_4_examples.mp4。四例速览为 21.33 秒，首镜头完整展示、0.5× 慢放；单例视频均为六镜头原速 16 秒，左原片、右修复。

- 01 Clara 花店：历史自动流程，酒红外套变棕褐色。
- 02 Iris 花店：历史自动流程，深色头发变偏红；衣服要求仍未解决。
- 03 Clara 图书馆：固定目标实验，灰衣变棕黄色；纸张与动作变化。
- 04 Iris 酒店：固定目标实验，白发变红卷发；姿态／脸部变化，后期仍漂移。
- 05 失败对照：固定整人区域修复引入花盆、遮挡和异常。

这些都是已暴露的本地 5B 开发结果，经过选择，不代表成功率、身份保真或最新官方 1.3B progressive 的效果。固定目标实验绕过自动 guard；历史自动流程的接受标签不是独立真值。最新 progressive 先前零保留的结论没有因本展示而改变。

manifest.json 保存视频来源、SHA-256 与操作；evidence/ 保留全案例记录。没有新生成、没有新验收、没有伪造人评。完整视频保留全部原帧，仅并排缩小并增加标题，不对场景像素做颜色修正或裁剪。原文件路径仅用于追溯，本文件夹的视频和页面可独立查看。

在 LongLive-RAG 源工作区重新制作（需已有归档、OpenCV、NumPy、Pillow、ffmpeg 和 Noto CJK 字体）：

```bash
python scripts/build_repair_showcase.py --output videos/repair_showcase_new
```

输出目录必须不存在，避免覆盖已交付素材。
''')
    write_page(output, rows)
    with zipfile.ZipFile(output / 'repair_showcase.zip', 'w', compression=zipfile.ZIP_STORED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'repair_showcase.zip':
                z.write(path, Path('repair_showcase') / path.relative_to(output))
    print(json.dumps({'output': str(output), 'seconds': time.monotonic() - began, 'videos': len(rows) + 1}), flush=True)


if __name__ == '__main__':
    main()
