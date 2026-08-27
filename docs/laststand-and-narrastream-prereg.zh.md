# 预注册:追加路线最后一搏 + NarraStream 三臂对照

2026-08-27。两个实验在机理研究([attention-downweighting-study.zh.md](attention-downweighting-study.zh.md))
和修复实验([editing-repair-report.zh.md](editing-repair-report.zh.md))收口之后立项,
判定规则在跑任何 render 之前冻结在本文,跑完只对照打分,不允许赛后移动球门。

背景一句话:机理研究证明追加式注入的天花板来自 recency 垄断(当前 chunk 拿走
66–84% 注意力 mass,无偏降权 2.3–10.6×),写入路线(masked renoise + V-edit)是
利用垄断而非对抗垄断的通道。实验一花小钱买追加路线的完备性,实验二是论文主表。

## 实验一:追加路线最后一搏(4 renders)

机理研究给追加路线留了三个没试过的便宜杠杆,合成一个 arm 一次打完:

1. **时机前移**(`inject_active_steps=[0,1]`):L29 对注入块的消费随 denoise step
   四步衰减 4 倍(0.0138→0.0034),把注入集中在最早两步;
2. **头定向**(`inject_head_filter={29:[记忆头]}`):E5 头普查发现 L29 有两个头
   per-column mass 达 18–30×,只让记忆头看到注入列,不再摊薄给不读记忆的头;
3. **L14 相位匹配存储**(`store_phase_matched_layers=[14]`):E4 发现 clean 相位
   的存储 K 和噪声相位的查询 q 在中层是两种表征(step3 vs step0 余弦 0.43),
   改存 step0 相位的 K,只对齐 key 匹配,V 仍用 clean 版。

### 设计

| arm | 场景 | 配置 |
|---|---|---|
| baseline_rerun | watchmaker, driftwood | 三开关全关(顺带验证开关的 feature-off 等价) |
| laststand_full | watchmaker, driftwood | 三开关全开,其余与 channel test 同 KV plan |

seed0,共 4 renders。不做单开关消融——预算就是"最后一搏",赢了再拆。

### 预注册判定

- **胜**:laststand_full 的 `anchor_drift_aggregate_consistency` 相对 baseline_rerun
  的提升在**两个场景都** ≥ 该场景冻结的 2σ 阈值(Stage A 台账),且注入列
  attention mass 显著上升(manipulation check,复用探针);→ 扩到 9 场景 ×
  3 seeds 正式一轮。
- **负**:任一场景不过 → **追加式路线彻底归档**,论文按 negative result 写
  ("we exhaustively tried timing, head-targeting and phase matching"),
  不允许追加第四个杠杆。
- baseline_rerun 与历史 baseline 不逐位一致 → BLOCKED,先修基础设施,不出结论。

## 实验二:NarraStream-Bench 三臂对照(论文主表)

外部 benchmark 采用 NarraStream-Bench(arXiv 2605.18733 的配套,MIT,
`example/narrastream_bench/`;转换器 `scripts/convert_narrastream_to_multitext.py`)。
理由:(1) LongLive 是它的正式 baseline,锚点数字现成;(2) 它的 Entity
Grounding / Conditional Long-range Consistency 就是我们欠的实体级指标,带 30 人
human-alignment 验证,部分偿还 M0 指标债;(3) 格式即插即用(已验证子集能被
`MultiTextConcatDataset` 直接加载)。

已知限制,写进论文而不是绕开:只测人物实体(他们的 LLM 抽取模板限定 human
characters);绝对分不可与他们的表跨 backbone 比较(他们 Wan2.1/MemFlow 832×480,
我们 Wan 5B),只报臂间相对差;内部 frozen hard benchmark(含鱼、怀表等非人主体)
保留为第一层评测。

### 设计

迭代期:`multitext_subset33_b8`(stride-10 采 33 场景,8 blocks/段 ≈2.7s/段);
终版:全 324 条 × 30 blocks/段(benchmark 原生 10s/段、60s/条)。

| arm | 机制 | 对应文献位 |
|---|---|---|
| A baseline | native rollout + sink(LongLive 原样) | LongLive |
| B concat 注入 | KV-RAG 检索 + 追加列注入(Round18 升级版 arm) | IAMFlow/MemFlow 的机制类代理 |
| C repair loop | 对比式裁判逐块判漂移 → edit_renoise σ=1.0 修复 → 复核 → commit/quarantine | 我们的写入路线 |

同 seed 配对;arm B 同时报注入列实际 attention mass;arm C 同时报修复触发率、
复核通过率、quarantine 率。

### 预注册判定与预测

- **主判定**:C 在 CLC + EG 两个实体级指标上,跨场景配对 ≥2σ 胜 A → 写入路线
  的核心 claim 成立。
- **预注册预测**(这是机理结论的可证伪测试):B 的注入列 mass 依旧被压在
  个位数百分比,B 对 A 的 CLC/EG 提升不显著。**若 B 显著胜 A,recency 垄断
  结论需要修正**——这条写进论文,输赢都有信息量。
- guard 组照旧:RAFT 运动、多样性下限、adherence 容差(防冻结换一致性,
  他们 Table 2 里 Self-Forcing 的 Dynamic Degree −10 就是我们 guard 要拦的作弊)。

### 评测栈

NarraStream 官方 metric 代码(DINO/CLIP/RAFT/AMT-S/LanguageBind/VTSS)+
MLLM judge。judge 原版是 Qwen3-VL-30B-A3B(vLLM,重);迭代期允许换
Qwen3-VL-8B(仓库已有集成),但**终版数字必须用原版 judge**,且两版 judge 的
臂间排序须一致(他们自己的 4.5 节 cross-judge 稳健性检查,我们照做)。

## 执行顺序

1. 实验一先跑(4 renders,半天):它的结果决定论文里追加路线的最终措辞;
2. 实验二迭代期(33 场景 × 3 臂,8 blocks/段):跑通管线、校准 judge;
3. 实验二终版(全量 × 30 blocks/段):出主表。

依赖:实验一依赖 kv_rag 三开关实现;实验二 arm C 依赖 edit_shot 闭环接线。
两者都已排期实现,冒烟测试过了才准开 render。
