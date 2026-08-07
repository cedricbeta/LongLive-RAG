# VLM-Guided Agentic Memory for Streaming Video Generation — 方法总结与实验计划

2026-07-30。短版;完整实验记录见 [README.md](README.md)。

## 方法

在冻结的流式生成器(LongLive backbone + LongLive-RAG 检索记忆, arXiv [2606.02553](https://arxiv.org/abs/2606.02553))外面套一个 VLM 驱动的验证-控制环:每生成一个 block(3 潜帧),VLM 裁决一次,裁决结果决定这段内容能否进记忆、要不要重来、失败时向哪级升级。思路上是把代码 agent 的验证循环模式(验收门控、独立审查、隔离区、分级升级、账本)移植到视频生成器的内部状态(KV cache / 检索池)上。

![机制图](figures/vlm_agent_mechanism.png)

三层结构:

- **判分**:VLM 双层裁决——绝对式(主体/外观/画质,抓灾难失败)+ 对比式(相对开场参照,抓渐进漂移;绝对式对漂移是盲的,12 条 observe 全单一标签换来的教训);
- **记忆**:准入门(被拒块 KV 禁入检索池)、金帧锚(VLM 选举 2 帧规范参照,强制占检索槽,走位置 0 sink 相位注入)、负记忆(被拒块描述子存 error buffer,margin = sim 正池 − sim 负池做廉价漂移预警);
- **控制**:失败分级升级——重摇(KV 快照回滚+换噪声,预算前置 block 0)→ prompt 手术(N 次全拒同理由触发,VLM 带证据改写后重启)→ 外部参照注入(规划中);两次不过则隔离。

## 已验证

| 结论 | 证据 |
|---|---|
| VLM 引导在原方法自己的记分下更好 | VBench Avg. Rank 2.21 vs 2.42(4 prompts, 单 seed) |
| 重生成同时改善动态性与一致性 | 猛犸场景唯一 dynamic=1.0 且 subject 最高;健康场景零重试、逐位无害 |
| 手术胜盲重摇 | 盲重摇 34 次救 6 块 → 手术 1 轮后 40/40 全过 |
| 同 shot 追加通道成立 | 注入记忆 0.40× uniform、19% 绝对注意力(30 层探针) |
| 注入被当场景统计量消费 | 主体/背景注意力密度比 1.01 → 身份修复需注意力引导或 V-edit |
| 负记忆 margin 可用 | 预测漂移标签 AUC 0.787 vs 正池-only 0.605 |

尚未使用 V-edit;全部干预是检索槽操纵、重摇、文本层。

## 相关工作定位

- **视频测试时扩展**:[Video-T1](https://arxiv.org/abs/2503.18942)(块级 verifier 搜索)、[Stream-T1](https://arxiv.org/abs/2605.04461)(同 backbone 同 benchmark 的最近邻:常开 beam + 标量奖励质量门 + sink 追加;无诊断、无文本干预、无回滚)。我们的差异:语义裁决(能说"画成了熊")支撑条件化动作;按失败付费而非常开;有回滚所以有升级链;
- **VLM 反馈改 prompt(外环)**:[VISTA](https://g-vista.github.io/)、[VQQA](https://arxiv.org/pdf/2603.12310)、[VideoRepair](https://arxiv.org/abs/2411.15115)——整条视频生成完再批评重来;我们的手术嵌在流内 block 0,失败早发现早改;
- **记忆一致性**:[Context as Memory](https://arxiv.org/abs/2506.03141)、WorldMem——按几何/相似度检索,均无记忆验收;
- **验证循环 agent(结构祖先)**:Reflexion(教训记忆)、Voyager(验证过才进技能库=准入门)、AlphaCodium(测试门控迭代)、ToT/过程监督(步级验证+回溯=块级裁决+KV 回滚)、ADAS/AFlow(策略自动发现=Controller 化)。差异:代码域验证有 ground truth,我们的裁判是会犯错的 VLM——裁判可靠性是一等问题。

## 实验计划

按优先级;每条带判定标准与对应文献。

**E1 静态对照 + Stream-T1 复刻**(隔离 VLM 的净贡献,必做)。把 VLM 裁决换成标量奖励移动平均门(即 Stream-T1 质量门的复刻)、金帧换成清晰度启发式选帧,同算力对比。判定:VLM 版须在实体级指标上胜出,否则贡献在机制不在 VLM。参考:Stream-T1 §Memory Sinking。

**E2 触发式 vs 常开算力**(路线之争)。同算力预算下,Stream-T1 式每块 beam(K×M)vs 我们的裁决触发升级链,在健康 + 先验失败混合 prompt 集上跑。预期:健康集打平但我们便宜数倍,失败集(标量奖励看不见的语义失败)我们独赢。参考:Stream-T1、Video-T1。

**E3 内环 vs 外环干预**(早停的价值)。VISTA 式整条重来 vs 我们 block 0 手术,同算力,测达到同等 adherence 的成本。参考:VISTA、VQQA、VideoRepair。

**E4 区域 logit bias → V-edit 判定**(机制主线)。给注入帧主体 token 列加 λ∈{1,2} 偏置,区域探针复测主体密度(现在 1.01);抬得动→测实体级指标;抬不动→放弃追加+引导路线,转原位 V-edit(主仓库已验证通道,方向约束用负原型差)。判定标准预注册。

**E5 对比式准入 + margin guard 上线**(负记忆落地)。生产 gate 加对比式裁决层;margin<θ 才调 VLM(异步化,利用 recent_exclude=5 的时间窗)。判定:长时程(120s)下检索污染率、VLM 调用数、一致性指标三者同报。参考:Reflexion/Voyager 的验证式记忆。

**E6 扩量与正式协议**(发表门槛)。VBench-Long 完整协议(128 prompts、30/60/120s)× 3 seeds;backbone 泛化(Causal-Forcing / Self-Forcing,同 LongLive-RAG 论文矩阵);等算力盲重采对照;gate 预注册(逐场景 2σ 噪声底,沿用主仓库纪律)。

**E7 Controller vs 手写升级链**(去 heuristic 化)。把工具箱、账本、成本表给 VLM,让它自己提动作并预测效果,与手写链对跑猛犸类失败。平=策略可交给模型;输=手写链里有模型没有的机制知识,本身可写。参考:ADAS、AFlow。

**E8 外部参照注入**(升级链第三级)。backbone 概念缺失(象鼻)时,用 T2I 参照帧 VAE 编码 teacher-force 成首帧,参照即金帧;同时是"自检索记忆→外部参照库"的第一个用例。参考:Context as Memory 的外部记忆bank。

远期:把 gate/账本/升级链搬回主仓库 Wan2.2-5B 跨镜头设定与 V-edit 配对(跨镜头追加被降权 4-20×,原位改写是已验证的唯一通道)。
