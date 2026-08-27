# Codex Goal — Agentic Boundary Runtime v1（工程闭环，不做质量主张）

> 2026-07-16 · 方向：**Model-Native Memory Agent for Autoregressive Video Generation**。
> 本 goal 只交付工程闭环：把现有离线多轮脚本重构为可暂停、可观察、可行动、可提交的
> `CausalShotSession`，实现单 shot 的 observe → diagnose → act → verify → commit/rollback。
> **v1 的验收全部是工程等价性与协议完整性，禁止夹带任何质量/一致性提升主张**——
> 正式对比实验（baseline / static / prompt-only / agent / matched-compute retry）的 gate
> 将在 v1 通过后另行预注册。

## 0. 已验证的事实基础（capsule pilot 2026-07-15/16 遗产，不要重新论证）

- **原位 V 编辑通道有因果力**：live-window 实体 token 的整向量 V 编辑（P=I, α=1.0）
  4/4 arm 到达计算图，编辑点后 100% 帧改变，身份指标摆动 ±0.09——但**符号不受控**
  （同一动作不同 shot 上正负都出现）。工具在 `utils/capsule.py`（probe/edit 通道）。
- **表征弥散性**：clean-recache V 中实例身份高于 chance 可解码但无低秩线性集中结构
  （G1/G1b 双估计器一致）。含义：不要在 v1 里做任何子空间分解类动作。
- **重放/仪表透明性已证**：probe-on/off 逐位一致的 harness 在
  `scripts/audit_capsule_probe_replays.py` / `evaluate_capsule_g0.py`，作为 v1 等价性
  测试的模板。
- 现成积木：shot boundary 调度 `pipeline/causal_diffusion_inference.py:835`；
  持久 KV 与 boundary 注入 `utils/kv_rag.py:185`；VLM 诊断/审查门/anchor 选择
  `scripts/run_round17_vlm_loop.py:234`（schema 约束解码 + 指纹缓存照搬）。
- **外部相关工作（2026-06-01 发表，必须知晓）**：LongLive-RAG 论文
  （arXiv 2606.02553，LongLive 原团队；代码 github.com/qixinhu11/LongLive-RAG）——
  自生成 latent 历史的检索记忆，逐 block 检索 top-K 历史条目**追加**进注意力上下文，
  生成器冻结、只训练检索嵌入；单 prompt 连续长视频上 VBench-Long 最佳平均排名。
  与本系统的边界：它是追加式、无验收控制、无诊断、latent 块级；我们的核心动作是
  原位改写、验收准入、诊断驱动、实体 token 级。**命名冲突：本 repo 与其同名，
  论文写作时我们的方法需改名并引用它。** v1 不需要实现它，但接口设计不得阻碍
  之后把它作为 baseline 接入（其检索条目走与 `inject_entity_kv` 相同的追加通道）。

## 1. 运行协议（v1 只实现 shot-buffered）

shot 生成后为 **provisional**，verifier 通过才 commit（交付 + 写可信记忆）；失败则
restore boundary snapshot、执行定向动作、**最多重试一次**；再失败则 accept-with-quarantine
（交付但隔离，不写记忆）。strict streaming 是后续退化配置，v1 不实现但接口不得阻断它
（所有动作标注 forward-only 与否）。

## 2. CausalShotSession 规格

```text
session = CausalShotSession(config, scene)
for k in shots:
    ctx  = session.pre_shot(k)      # 只读：normative graph + 可信记忆 + 内部 telemetry
    plan = controller.decide(ctx)    # 结构化 action program（工具 + 有限参数）
    session.apply(plan)              # 落 ledger
    session.snapshot(k)              # boundary 快照
    shot = session.generate(k)       # provisional
    obs  = session.observe(k)        # 低清帧/实体 crop + 内部 telemetry
    verdict = verifier.judge(obs, normative_graph)   # VLM + 数值 guard
    if verdict.accept: session.commit(k)             # 交付 + admit 记忆
    elif budget: session.restore(k); plan2 = controller.repair(verdict); retry once
    else: session.commit_quarantined(k)              # 交付但记忆隔离
```

**snapshot 必须覆盖**（沿用 v5 §3.4 清单）：已 commit output latent 与当前 shot noise
slice、正/负 KV cache、cross-attention cache、KV-RAG transient/candidate/persistent
分区与索引、shot index、RoPE offset、scheduler、CPU/CUDA RNG state、streaming VAE
待解码状态与输出游标。

**信息边界（硬约束）**：pre_shot(k) 与 verifier(k) 的所有输入必须可审计为
"截至当时已生成"；任何 API 拿到未来 shot 的帧/文本/指标即算失败。参照 v5 §1.3。

## 3. 三类记忆（分区是本系统成立的关键）

| 分区 | 内容 | 写入规则 |
|---|---|---|
| **规范记忆** | 剧本编译出的实体/不变量/允许状态变化（VLM State Compiler 从 global.json + 逐 shot JSON 提取） | 生成开始后**只读**；生成结果永远不能反写它 |
| **视觉记忆** | 通过 verifier 的实体 crop、embedding、mask | 只在 commit 时 admit；quarantine shot 的一切不得进入 |
| **模型原生记忆** | 可信帧对应的 KV/V、token/region 索引、layer 信息 | 与视觉记忆同步 admit/quarantine；供干预工具读取 |

candidate → (verifier) → persistent / quarantine 的三态流转必须有单元测试覆盖。

## 4. 工具接口（VLM 只选离散工具 + 有限参数，不生成张量）

| 诊断 | 工具 | 参数域（有界） |
|---|---|---|
| 文本不变量遵循失败 | `patch_prompt(invariant)` | 只许追加，过审查门 |
| 主体消失 | `inject_entity_kv(entity, boundary)` | 锚帧 ≤4，region mask |
| 身份漂移、构图正常 | `steer_v(entity, layers, alpha)` | layers ⊆ {0,7,14,21,29}，α ∈ {0.25,0.5,1.0} |
| 场景布局漂移 | `inject_scene_anchor_kv(boundary)` | 同 kv_rag 现有机制 |
| 历史 token 被降权 | `rope_remap(entries)` / `logit_bias(lambda)` | λ ∈ {0,1,2} |
| 当前 shot 不可信 | `quarantine(shot)` | — |
| 严重失败且有预算 | `restore_and_retry(action_program)` | 每 shot ≤1 次 |
| 置信度不足 | `abstain()` | 一等公民动作，落账理由 |

每次调用记录：诊断输入指纹、工具、参数、理由、attention-mass 操纵检查（如适用）、
结果指标。VLM 输出走 JSON Schema 约束解码 + 指纹缓存（R17/18 现成设施）。

## 5. v1 验收 gate（全部工程性，逐项出证据）

1. **feature-off 等价**：`agent_runtime.enabled=false` 时与现行 `inference.py` 生成
   **解码帧逐位一致**（复用 G0 harness，4 个 pilot 场景 seed0）；
2. **snapshot/restore 等价**：restore 后无动作重生成与连续运行同噪声逐位一致
   （每场景至少 2 个 boundary 抽查）；
3. **三分区完整**：quarantine 的 shot 的 crop/KV 在后续任何检索/注入中不可见
   （单元测试 + 端到端断言）；
4. **无未来泄漏**：ledger 中每个 VLM 调用的输入帧/文本可审计为当时已生成
   （自动审计脚本，模式照 `frozen_inputs_sha256.json`）；
5. **ledger 完备**：随机抽 3 个动作，能从 ledger 单独复现（同输入 → 同决策，
   靠指纹缓存）；
6. **retry 预算强制**：注入一个必失败的 verifier 打桩，确认恰好 1 次 retry 后走
   accept-with-quarantine，且被隔离 shot 不污染记忆；
7. **端到端演示**：watchmaker + driftwood 各跑一次完整 agent session（真 VLM、
   真动作），产出 2-up 演示视频 + 完整 ledger——演示"系统会转"，不主张"变好了"。

## 6. 并行 Track W — 观察器离线验证（零渲染，与 v1 同时开工）

用 pilot 现成的 probe 重放 dump（4 场景 × 3 seeds）做"When"支柱的验证：
逐 boundary 内部信号（注意力熵、sink 份额、逐帧 mass 分布、V 范数漂移、身份词
cross-attention mass）预测下一 shot 的实体漂移（现有 DINO subject 流）。
**必须击败"流逝时间/shot index"平凡基线**（cluster bootstrap CI），leave-one-scene-out。
- 通过 → verifier/pre_shot 的 telemetry 输入有据，"When"进主张；
- 不过 → When 降级为固定 boundary 全检，照常做 v1，结果如实入 REPORT。

## 7. v1 之后的实验路线（预告，本 goal 不执行）

**第一个实验——动作可靠性研究**：`steer_v` 的 α × donor 选择 × 层组合小扫描，目标是
把"±0.09 的因果力"变成"方向可预期的收益"。它是正式对比实验的前置——不可靠的
动作上 agent 打不过 static/matched-compute 对照。

**正式对比实验中与 LongLive-RAG 论文相关的对照项**（gate 届时预注册，此处先冻结
对照设计，防止事后挑选）：

1. **LLRAG-baseline**：其开源实现接入我们的多镜头 benchmark——训练检索嵌入 + 追加
   注入在跨镜头实体重现设定下是否有效（基于 Round 13–18 的预测：受限；须实测）；
2. **选择器隔离**：同一追加注入机制下，其训练检索器 vs 我们的 VLM 语义选择——
   隔离"选什么"的贡献；
3. **机制类别隔离**：相同 donor 帧下，追加注入 vs `steer_v` 原位改写——隔离
   "怎么给"的贡献；这是"追加治同场景误差累积、改写治跨镜头实体修复"边界条件
   主张的直接检验；
4. **双 regime 报告**：单 prompt 连续长视频（其主场：MovieGenBench 128 prompts、
   Qwen 精炼、30s 起）与多镜头场景切换（我们的主场：冻结难例 benchmark）各报一组，
   两个方法都在两个 regime 上跑；接入其方法时照抄其发表配置
   （context layout `[1 sink ‖ 6 retrieved ‖ 5 local]`、K=6、recency guard），
   避免削弱 baseline 的指控；
5. **记忆卫生消融**：无准入的检索池 vs verifier 验收过滤的检索池（它没有准入控制，
   漂移历史可能被检回——这是我们隔离区设计的直接对照）；
6. **指标与开销双轨**：官方实现的 VBench-Long 六项（与其结果可比；非官方实现只能
   标 proxy）+ 实体级指标（我们的主尺）；开销并报（其检索 4.08 ms/block vs 我们的
   probe/VLM 成本）。

**指标命名纪律（写论文时强制）**：其 "Subject Consistency" = VBench 整帧 DINO 帧间
相似度，单 prompt 连续 regime、天花板 94–97、增益 +0.2~0.8 量级；我们的主指标 =
跨镜头实体重现一致性（masked-entity、to-first、fidelity-gated），难例 baseline 漂移
0.4–0.6、逐场景 2σ 底 0.018–0.135。两者不是同一个量，行文必须区分命名。其协议
无 guard 否决的代价有实例：其 Self-Forcing 30s 上 Dynamic Degree 52.0→42.2（−10）
同时 SC +1.4 仍获最佳平均排名——该 trade 在我们的 guard 协议下会被否决；guard 制、
负控制、配对 2σ 是相对该工作的方法学优势，论文中作为卖点陈述。

## 8. 运维与约束（继承，强制）

- 单驱动文件锁（orchestrator 已有，session runtime 同样要）；失败归档后重试一次再
  fatal；GPU 挑选余量 ≥60 GiB；长渲染 tmux；
- 新代码全部 flag 默认关；冻结 benchmark 与 stage_A 存档只读；
- 交付 `docs/agentic_runtime_v1/REPORT.md`：验收 gate 逐项证据、Track W 结果、
  演示材料链接、复现命令、已知限制。
