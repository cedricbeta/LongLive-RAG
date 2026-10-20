from pathlib import Path
import hashlib, json, shutil, time
S=Path('/home/chendong/video-gen/LongLive-RAG')
T=Path('/home/chendong/video-gen/ICLR27-Longlive-agent')
R=S/'videos/progressive_repair_20260917'
F=R/'native_fast_cases_v1'
read=lambda p:json.loads(p.read_text())
assert read(F/'ns190/candidate/status.json')['status']=='completed'
assert read(F/'ns190/candidate/audit.json')['measured_engineering_checks_passed']
assert read(F/'repair_completion_queue.json')['phase']=='completed'
assert (F/'developer_observations.json').is_file()
records=[]
def copy(src,rel):
    dst=T/rel
    assert not dst.exists(), str(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    a=hashlib.sha256(src.read_bytes()).hexdigest()
    assert hashlib.sha256(dst.read_bytes()).hexdigest()==a
    records.append({'path':str(rel),'source':str(src.relative_to(S)),'sha256':a,'bytes':dst.stat().st_size})
for src in sorted(F.rglob('*')):
    if not src.is_file():continue
    rel=src.relative_to(F)
    if src.suffix in ('.json','.yaml','.txt','.py','.log'):
        copy(src,Path('results/native_fast_cases_v1')/rel)
    elif src.suffix=='.png':
        if 'ns190/baseline/sample_frames' in str(rel) and src.name!='all_fixed_views.png':continue
        copy(src,Path('artifacts/native_fast_cases_v1')/rel)
for case,branch,name in [('ns220','baseline','ns220_baseline.mp4'),('ns190','baseline','ns190_baseline.mp4'),('ns190','candidate','ns190_candidate.mp4')]:
    copy(next((F/case/branch/'videos').glob('*.mp4')),Path('artifacts/native_fast_cases_v1')/name)
copy(F/'ns190/frame_review/full_60s_comparison.mp4',Path('artifacts/native_fast_cases_v1/ns190_full_60s_comparison.mp4'))
for name in ['native_fast_ns220_v1_telemetry','native_fast_ns190_v1_telemetry','native_fast_ns190_repair_v1_telemetry']:
    assert read(R/name/'status.json')['status']=='completed'
    for src in sorted((R/name).glob('*')):
        if src.is_file():copy(src,Path('results')/name/src.name)
for name in ['prepare_native_fast_cases.py','audit_native_fast_case.py','audit_native_fast_repair.py','review_native_fast_repair.py']:
    copy(S/'scripts'/name,Path('implementation/scripts')/name)
copy(S/'example/narrastream_bench/streambench_324.jsonl',Path('reproduction/native_fast_cases/streambench_324.jsonl'))
copy(F/'ns190/hair_mask/mask.pt',Path('reproduction/native_fast_cases/automatic_hair_mask.tensor'))
manifest={'phase':'complete_bounded_native_fixed_target_cases','exported_unix':time.time(),'files':records,'raw_latents_and_weights_in_git':False,'history_resident_content_hashes_measured':False,'human_labels':0,'candidate_accepted':False}
(T/'results/native_fast_cases_v1/export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths=[r['path'] for r in records]+['results/native_fast_cases_v1/export_manifest.json','reproduction/native_fast_cases/README.md','artifacts/native_fast_cases_v1/index.html','README.md']
(R/'native_fast_export_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records)}))

from html import escape
obs=read(F/'developer_observations.json')
page=T/'artifacts/native_fast_cases_v1/index.html'
page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>官方 1.3B：两例有界实验</title><style>body{font:17px/1.65 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#192b35;background:#f6f8fa}video,img{width:100%;height:auto;background:#111}article{background:white;padding:22px;margin:22px 0;border:1px solid #dae1e5}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style><h1>官方 1.3B：两例有界实验</h1><p>完整 60 秒，832×480 原片，16 fps。开发暴露脚本；固定目标探针，不是最新 progressive 自动验收成功。人工标注数：0。</p><p class="note">'+escape(obs['summary_zh'])+'</p><article><h2>酒店 Iris：红发目标，唯一一次候选</h2><video controls preload="metadata" src="ns190_full_60s_comparison.mp4"></video><p>左：官方原片。右：固定目标候选。全部 960 帧保留，无剪切、插帧或调色。候选未获自动验收。</p><p><a href="ns190_baseline.mp4">原片</a> · <a href="ns190_candidate.mp4">候选原分辨率视频</a></p><p>强度 1.0；修复种子 4243；首镜头自动 hair mask；保留完整原脚本，追加 Iris has red hair.。下图为生成前记录的定位限制：只覆盖头发两侧，头顶覆盖不足。</p><img src="ns190/hair_mask/mask_latent_0020.png" alt="自动 hair mask，紫色为选中区域"></article><article><h2>固定的 18 个时间点</h2>'+''.join('<img loading="lazy" src="ns190/frame_review/fixed_views_'+str(i)+'.png" alt="成对固定样本 '+str(i)+'">' for i in (1,2,3))+'</article><article><h2>图书馆 Clara：棕色外套已经存在，因此不修</h2><video controls preload="metadata" src="ns220_baseline.mp4"></video><p>18 个固定采样中外套已为棕色。原片仍可见多余/重复人物和后段伪影，不能据此判定全片通过。</p><img loading="lazy" src="ns220/baseline/sample_frames/all_fixed_views.png" alt="图书馆原片固定样本"></article><p><a href="../../reproduction/native_fast_cases/README.md">配置与复现</a> · <a href="../../results/native_fast_cases_v1/summary.json">全部成本和结果</a> · <a href="../historical_repair_showcase_20260917/index.html">历史 5B 四例（不同模型/方法）</a></p></html>')
