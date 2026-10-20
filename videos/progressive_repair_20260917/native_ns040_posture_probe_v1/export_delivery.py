#!/usr/bin/env python3
import hashlib,json,shutil,time
from pathlib import Path
from html import escape
F=Path(__file__).resolve().parent;S=F.parents[2];T=S.parent/'ICLR27-Longlive-agent';R=F.parent
NAME=F.name;records=[];read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(F/'summary.json')['remaining_generation_budget']==0
assert read(F/'pair_check.json')['status']=='passed'
assert not (T/'results'/NAME).exists()
def copy(src,rel):
 dst=T/rel;assert not dst.exists(),dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 assert sha(src)==sha(dst);records.append({'path':str(rel),'sha256':sha(dst),'bytes':dst.stat().st_size,'source':str(src.relative_to(S))})
for src in sorted(F.rglob('*')):
 rel=src.relative_to(F)
 if rel.parts[0]=='B_original' or not src.is_file() or '__pycache__' in rel.parts:continue
 if src.suffix in ('.json','.jsonl','.yaml','.txt','.py','.log'):
  copy(src,Path('results')/NAME/rel)
 elif src.suffix=='.png':copy(src,Path('artifacts')/NAME/rel)
 elif src.suffix=='.pt' and rel==Path('group_mask/mask.pt'):copy(src,Path('results')/NAME/rel)
for arm in ('P_prompt','R_group'):
 copy(next((F/arm/'videos').glob('*.mp4')),Path('artifacts')/NAME/(arm+'.mp4'))
copy(F/'review/full_60s_comparison.mp4',Path('artifacts')/NAME/'full_60s_comparison.mp4')
for name in ('prepare_native_posture_probe.py','prepare_native_group_mask.py','audit_native_posture_probe.py','review_native_posture_probe.py'):
 copy(S/'scripts'/name,Path('implementation/scripts')/name)
# The current source auditor has already been exported under this distinct name.
assert sha(S/'scripts/audit_native_fast_repair.py')==sha(T/'implementation/scripts/audit_native_fixed_target.py')
copy(F/'METHOD_AND_REPRODUCTION.md',Path('reproduction/native_posture_probe/README.md'))
# Include the bounded prerequisite records, avoiding all model/cache blobs and
# raw transport logs containing temporary signed public CDN query strings.
packages={
 'official_narrastream_planner_provision_v1':['plan.json','status.json','public_metadata.json','environment_plan.json','environment_status.json','environment_recovery.json','cpu_import_config_preflight.json','cpu_import_config_preflight_v2.json','provision.py'],
 'official_narrastream_planner_provision_v2':['plan.json','status.json','prior_terminal_status.json','assets_manifest.json','cpu_import_config_preflight_v2.json','recover.py'],
 'official_narrastream_planner_preflight_v1':['worker.py','launch_after_provision.py','dependency_abort.json'],
 'official_narrastream_planner_preflight_v2':['worker.py','launch_after_provision.py','preparation.json','plan.json','manifest.json','status.json','responses.json','result.json','summary.json'],
 'official_narrastream_planner_preflight_v2_telemetry':['plan.json','device.json','status.json','summary.json','samples.jsonl','monitor_source.py']}
for name,files in packages.items():
 for file in files:
  src=R/name/file
  if src.exists():copy(src,Path('results/narrastream_action_preflight_v1')/name/file)
D=R/'native_pilot12_action_diagnostic_v1'
for src in sorted(D.rglob('*')):
 rel=src.relative_to(D)
 if not src.is_file() or '__pycache__' in rel.parts or rel.parts[0]=='server':continue
 if src.suffix in ('.json','.jsonl','.py','.txt'):copy(src,Path('results/narrastream_action_preflight_v1')/D.name/rel)
 elif src.suffix=='.png':copy(src,Path('artifacts/narrastream_action_preflight_v1')/D.name/rel)
obs=read(F/'developer_observations.json');page=T/'artifacts'/NAME/'index.html'
page.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n<title>屋顶姿态：原版、提示词对照与局部修复</title><style>body{font:17px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;background:#f6f8fa;color:#19303d}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce2e5}video,img{width:100%;height:auto}a{color:#165c92}.note{background:#fff1d6;padding:16px}</style>\n<h1>屋顶姿态：一次有对照的固定目标探针</h1><p>官方 LongLive-RAG 1.3B，完整 60 秒。左：已有原版；中：10–20 秒强化站立提示；右：同一时段、同一提示的局部群体区域修复。仅新增两次生成，已关闭预算。</p><p class="note">' + escape(obs['summary_zh']) + '</p><article><video controls preload="metadata" src="full_60s_comparison.mp4"></video><p><a href="full_60s_comparison.mp4#t=10">10 秒开始</a> · <a href="full_60s_comparison.mp4#t=15">15 秒</a> · <a href="full_60s_comparison.mp4#t=25">25 秒后续</a> · <a href="full_60s_comparison.mp4#t=45">45 秒后续</a></p><p>单片：<a href="../native_development_pilot12_v1/ns040/baseline/videos/rank0-0-0_lora.mp4">原版</a> · <a href="P_prompt.mp4">提示词对照</a> · <a href="R_group.mp4">局部修复候选</a></p><p>区域由程序分割得到，覆盖人物并包含部分桌面、物品与灯光；没有假定身份、动作或画质保持。目标在看过原版后固定，区域经过开发者预览，不能作为最新自动系统成功或独立验收。全部后续画面均保留。</p></article><article><h2>全部预定时刻</h2>' + ''.join(f'<img loading="lazy" src="review/fixed_views_{i}.png" alt="三路固定时刻 {i}">' for i in range(1,4)) + '</article><p><a href="../../results/native_ns040_posture_probe_v1/developer_observations.json">观察与限制</a> · <a href="../../results/native_ns040_posture_probe_v1/summary.json">次数与成本</a> · <a href="../../reproduction/native_posture_probe/README.md">复现</a> · <a href="../historical_repair_showcase_20260917/index.html">历史 5B 实例（不同版本）</a></p></html>')
records.append({'path':str(page.relative_to(T)),'sha256':sha(page),'bytes':page.stat().st_size,'source':None})
ref=Path('artifacts/native_development_pilot12_v1/ns040/baseline/videos/rank0-0-0_lora.mp4')
assert sha(T/ref)==read(F/'plan.json')['parent_video_sha256']
manifest=T/'results'/NAME/'export_manifest.json'
manifest.write_text(json.dumps({'phase':'completed_closed_posture_capability_probe','exported_unix':time.time(),'files':records,'reused_files':[{'path':str(ref),'sha256':sha(T/ref)}],'model_weights_and_raw_latents_in_git':False,'mask_tensor_included':True},indent=2)+'\n')
paths=[row['path'] for row in records]+[str(manifest.relative_to(T)),'README.md']
(R/'native_posture_delivery_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
print(json.dumps({'files':len(records),'bytes':sum(row['bytes'] for row in records)}))
