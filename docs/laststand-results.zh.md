# 追加路线最后一搏:结果与归档

2026-08-27。预注册见 [laststand-and-narrastream-prereg.zh.md](laststand-and-narrastream-prereg.zh.md),
判定规则先于渲染冻结:laststand_full 相对 baseline_rerun 的
`anchor_drift_aggregate_consistency` 提升须在**两个场景都** ≥ 冻结 2σ 阈值。

## 配置与执行

三开关全开:`inject_active_steps=[0,1]`(时机前移)、`inject_head_filter={29:[23,1]}`
(E5 普查的记忆头,watchmaker 46.2×/25.6×、driftwood 29.7×/18.8×,第三名骤降)、
`store_phase_matched_layers=[14]`(step0 相位 K,V 保持 clean)。4 臂(2 场景 ×
{full, baseline})seed0 各占一张 A100,全部 exit=0,渲染 480 帧。

开关全部实际生效(diag json 计数器,两场景一致):step 跳过 3600 次、头掩码
bias 110 次调用/2420 头次、相位替换 60/60、零回退。机制级 manipulation check
通过。

## 判定

| 场景 | Δ anchor_drift_aggregate | 冻结 2σ 阈值 | 单场景判定 |
|---|---|---|---|
| bearded_watchmaker_workshop | −0.0078 | 0.0501 | 负 |
| sandy_beach_driftwood | **+0.0747** | 0.0450 | **过线** |

**GATE: ARCHIVE_additive_route。** watchmaker 未过,按预注册规则追加式路线
正式归档,不允许追加第四个杠杆。论文措辞可以写实:位置、选择、剂量、偏置、
时机、头定向、相位匹配七个杠杆全部试过。

## 值得留档的观察(不改判定)

1. driftwood 的 +0.0747 是**本项目追加式干预有史以来第一个单场景超 2σ 的
   数值胜利**(Round 18 最好的 brown_bear +0.082 用的是未升级 arm 与旧阈值
   口径,不可直接比;本轮阈值 0.0450,超线 66%)。模式与 Round 18 相同:
   清晰主体的自然场景吃到收益,人物细节场景(watchmaker)不吃——这与机理
   研究"没有任何层的读取模式对准主体细节"一致:三个杠杆能把记忆送到位,
   但读出来的仍是粗粒度内容,救得了驼色浮木救不了胡须。
2. 若未来有人想重开(需要新立项):正确的下一问不是"再加杠杆",而是
   "为什么同样的杠杆在 driftwood 有效"——候选解释是该场景主体大、低频,
   恰好落在枢纽 token 式读取能覆盖的粒度上。
3. **探针缺口(诚实记录)**:laststand_full 的 attn_probe 只落了 store_k,
   q/window 采样被 `inject_active_steps` 的 skip-retrieve 短路连带跳过,
   mass 级 manipulation check 本轮拿不到,以机制级计数器代替。若做 driftwood
   post-mortem,需要专门跑一次探针 replay(开关开、probe 全量)。

## 对主线的影响

无变化:写入路线仍是主线(机理背书不依赖本轮结果)。本轮的价值是完备性——
追加式的负结果现在是穷尽式的。下一步照预注册顺序:NarraStream 三臂对照
(arm C 修复闭环已接线,31 测试通过)。

产物:videos/laststand_20260827/(renders/ 4 mp4、logs/、laststand_verdict.json、
attn_probe/ 25GB store_k)。
