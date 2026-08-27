# 研究计划 — 因果视频模型的记忆电路：定位、预测与定向控制

> v1 · 2026-07-15 · 负责人：cdwang
> 本计划是当前主线；`docs/multiview-consistency-research-plan.zh.md`（v5，agentic memory
> control 系统线）降级为 fallback——其测量学（M0 尺子）、因果契约（§1.3）与统计协议（§5.3）
> 被本计划全部继承。实证账本见 `docs/retrieval-key-value-study.md`。

---

## 0. 核心主张

从"用 agent 调度 KV 注入"（系统贡献，incremental）上移到三个科学问题：

> **F1（定位）**：一个 32 帧局部窗口的因果视频模型，跨 shot 实体身份由哪些 layer/head
> 携带？何时、如何丢失？
> **F2（预测）**：实体丢失发生前，模型内部状态（注意力分布、KV 统计）是否有可检测前兆
> ——"模型知道自己忘了"？
> **F3（控制）**：只对定位出的记忆电路做定向注入/steering，能否超过整帧 KV 注入
> ——并解释 Round 13–18 为什么 null（喂错了对象）？

18 轮的既有发现是种子证据：注入 token 被降权 4–20×、整帧 KV 只搬运场景统计量、λ 偏置
能抬注意力质量（0.017→0.041）但增益小——这些都是"记忆接口没搞清楚"的症状。
Video diffusion 的时间记忆机制解释目前是空白（LLM 侧 induction head / causal tracing
无视频对应物）。

## 1. Phase B：内部信号预测漂移（先行 pilot，1–2 周）

**为什么先做 B**：成本最低（≤27 次重放渲染 + 纯分析），且无论成败都定向——有信号则
F2 成立并给 F1 指路；无信号本身是约束性发现（记忆不在可观测注意力通道），回退 v5 无损失。

### B-0：挖既有数据（第 1 天，零渲染）

Round 18 `numeric_kv_only.json` 的场景级 `mean_persistent_attention_mass` 与该场景
kv_only 数值 delta 做 n=9 相关——只是 sanity check，统计力不足，不作结论。

### B-1：带仪表的确定性重放（核心）

**实现 `InternalSignalRecorder`**（挂在 [utils/kv_rag.py](utils/kv_rag.py) 现有
diagnostic 路径旁，复用 `record_attention_mass` 的采样 query 行 + einsum 模式，
config 开关 + best-effort never-raise，开销目标 <10%）。逐 block × 逐 layer 记录：

- **S1** sink 帧注意力份额（`multi_shot_sink` 钉住的列）；
- **S2** 上下文列注意力熵 + 距离衰减 profile（mass vs 帧距离曲线）；
- **S3** 逐历史帧注意力 mass 分布（哪些过去帧还被看）；
- **S4** KV key/value 范数统计：逐帧 key 范数、同空间区域 key 随时间的漂移量;
- **S5** cross-attention 中实体相关文本 token（不变量 caption 里的身份词）获得的
  mass——adherence 衰减的内部对应物。

**重放**：baseline seed0 × 9 场景（可选补 seeds 1-2 → 27 shot 序列），确定性同 seed
渲染，用逐 shot 指标与 Round 18 存档比对验证重放一致性。tmux 运行。

### B-2：预测实验与判据

- **对齐**：信号时间序列 ↔ 逐 shot 实体指标（先用现成 anchor_to_shot0 DINO 流，
  M0 masked-entity 尺子落地后换主尺）。
- **两种任务**：forecast（用 shot <k 的信号预测 shot k 的实体分数/失败）与
  nowcast（用 shot k 生成期间的信号判定当前 shot 失败）。
- **必须击败的平凡基线**：① 流逝时间/shot index（漂移随时间增长是平凡规律——
  这是本实验最大的混淆，内部信号必须提供 elapsed-time 之外的增量预测力）；
  ② prompt-only 特征；③ 随机。
- **协议**：leave-one-scene-out；AUPRC + 校准度 + 相对 elapsed-time 基线的增量
  （paired bootstrap CI 排除 0）；样本量 9 场景 × 7–8 shot × ≤3 seeds ≈ 200 点。
- **Gate**：held-out 场景上内部信号显著优于 elapsed-time → F2 成立，进入 Phase A 并
  把信号榜单交给 A-1 作先验；完全无信号 → 记录约束性发现，主线回退 v5。

## 2. Phase A：记忆电路定位与定向控制（3–5 周）

### A-0：boundary snapshot/restore（前置工程，把 v5 的 M7 提前）

patch 扫描的成本决定于此：有了 boundary 快照，每个 patch 实验只需**从切点重生成一个
shot**（6–12 block），而非整段 480 帧视频。最小版本覆盖：output latent、正/负 KV cache、
KV-RAG 分区、RoPE offset/shot index、RNG state、streaming VAE 游标（清单沿用 v5 §3.4）。
**验收 = 无动作等价性**：restore 后重生成与连续运行在数值容差内一致。

### A-1：因果追踪扫描（sufficiency）

对每个（场景, 漂移切点 k）：

1. **donor** = shot0（或最后一个 faithful shot）的上下文 KV；
2. **patch(ℓ)** = 只在第 ℓ 层把 donor KV 加入/替换进 shot k 的注意力上下文，其余层
   不动；从快照重生成 shot k，测该 shot 实体指标；
3. 扫描所有层（先确认 wan_5b 层数），有效层内再按 head 组细分（KV 张量带 head 维，
   flash 路径外做切片即可）；
4. **对照**：错误实体 donor（他场景的 shot0）、token-shuffled donor、全层整帧注入
   （= Round 18 arm，已知弱基线）。

产出 = **layer × head 的效应量定位图**及其跨场景/seed 一致性。成本估算：~30 层 ×
9 场景 × 单 shot 重生成，单卡数天量级。

### A-2：必要性方向（necessity）

在不漂移的场景/切点上，把定位层的身份 KV 换成他场景 KV 或噪声——身份是否被破坏。
sufficiency + necessity 双向才能宣称"电路"。

### A-3：定向控制 arm（F3，接回冻结 benchmark）

- **targeted-KV**：注入只作用于定位出的 layer/head（token 预算与 Round 18 整帧 arm
  持平或更低）；
- **steering-vector 变体**：faithful − drifted 的激活差方向，加在定位层的 hidden state；
- 评估协议全套沿用 v5：冻结 benchmark、同 seed 配对、guards、负控制、masked-entity
  主尺、paired bootstrap CI；对照 = B0 baseline、Round 18 整帧 arm。
- **判据**：targeted 在实体指标上 > 整帧 arm（CI 排除 0）且 guards 全过——
  "定位解释并修复了 null"。

### A-4：遗忘机制分析

在定位头上单变量检验：RoPE 位置重映射（降权是否位置性）、sink 竞争（S1 与身份头
mass 的此消彼长）、λ 偏置与定位头的交互——回答"为什么忘"。

## 3. 并行测量线（继承 v5 M0，不变）

masked-entity（SAM2/grounding + DINOv2/SigLIP crop）+ ArcFace + degrade test
（swap 降分 / copy-freeze 被拦 / 合法视角不误伤）。这是 F1–F3 一切实体主张的前提，
与 Phase B 并行开工。

## 4. 里程碑

| 周 | 工作 | gate |
|---|---|---|
| W1 | B-0 + `InternalSignalRecorder` + 重放 seed0×9 | 重放与存档一致；信号落账完整 |
| W2 | B-2 预测分析（∥ M0 尺子） | F2 gate：击败 elapsed-time（CI 排除 0）；否则回退 v5 |
| W3 | A-0 snapshot/restore | 无动作等价性通过 |
| W3–4 | A-1 层扫描 → head 细分 | 定位图跨场景一致；否则记录"分布式表征"发现 |
| W5 | A-2 necessity + A-3 targeted arm | targeted > 整帧（paired CI） |
| W6 | A-4 机制分析 + 汇总 | 论文骨架 F1–F4 齐备 |

**止损/转向**：

1. B 无信号 → 约束性发现入账，回退 v5 系统线（v5 文档保持可执行状态）；
2. A-1 效应弥散无定位 → "身份是分布式表征"仍是发现，论证转向 LoRA/训练必要性，
   或降级为 v5 + 负结果分析；
3. A-3 targeted 不优于整帧 → F1/F2 仍构成论文主体（定位 + 预测），控制部分如实报 null。

**论文骨架**：F1 记忆定位图（where）、F2 内部前兆预测（when，"model knows when it
forgets"）、F3 定向 steering 修复整帧注入的失败（how）、F4 机制解释（why：位置降权/
sink 竞争）+ Round 13–18 null 序列作为 motivation 链。agent/在线系统降为交付载体
（F2 = 零开销在线 probe，F3 = 控制动作），可选一节，不再是主张。

## 5. 工程注意

- flex_attention 是 `torch.compile` 编译的（[causal_model.py:28](wan_5b/modules/causal_model.py#L28)）
  ——所有诊断/patch 必须走 flash 路径之外的旁路（采样 einsum 重算），沿用
  `record_attention_mass` 的既有模式，避免 graph break；
- 长渲染一律 tmux（harness 后台任务曾两次被杀）；
- 每步落账 manifest/config/ledger，重放与 patch 实验记录完整可恢复命令。
