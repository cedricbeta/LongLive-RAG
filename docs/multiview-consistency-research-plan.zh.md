# 研究计划 v5 — Agentic Model-Native Memory Control for Long Video Generation

> v5 · 2026-07-15 · 负责人：cdwang
> 本计划继承 Round 13–18 的数据协议、冻结难例和实证账本，但重新定义研究边界、系统形态、
> 外部 benchmark 与阶段性止损点。方法现状见
> `docs/long-multishot-consistency-methodology.zh.md`，实证账本见
> `docs/retrieval-key-value-study.md`，Round 18 报告见
> `docs/round18_bench_20260701/REPORT.md`。

---

## 0. 决策摘要

下一阶段不再把“agent + visual memory”本身当作创新点，而聚焦为：

> **在连续 causal video rollout 中，agent 能否在线诊断跨镜头错误，并控制模型内部的
> entity/region/token 级 KV 与时序记忆状态，在固定 test-time compute 下提高长程实体一致性，
> 同时维持运动、镜头多样性和 prompt adherence？**

原因是 [GroundShot](https://arxiv.org/abs/2606.20799) 已经覆盖 training-free agent、
accepted-shot entity memory、写入前验证和生成前检索；
[VISTA](https://openaccess.thecvf.com/content/CVPR2026/html/Long_VISTA_A_Test-Time_Self-Improving_Video_Generation_Agent_CVPR_2026_paper.html)
也已覆盖 critic 驱动的 test-time 迭代生成。模式 S 与 VISTA 的区别是白盒、forward-only 的内部
记忆控制；模式 B 同样使用 test-time regeneration，但只做 boundary-local state rollback，并把
model-native intervention 作为重试动作。我们的可辨识空间应是：

1. **连续、按叙事顺序的 causal rollout**，而不是重排 shot 或 keyframe-first + 独立 I2V；
2. **model-native memory control**，而不只是把历史帧作为外部 reference；
3. **可定位、可审计的闭环动作**：何时干预、干预哪个实体/区域、写入什么记忆、使用多大剂量；
4. **两种部署模式**：严格流式的 forward-only 漂移抑制，以及 shot-buffered 的有限回滚纠错；
5. **matched-compute 评价**：证明收益来自 agent 决策，而不是多采样或额外算力。

Phase I 仍不训练基础视频模型；主 claim 只覆盖 KV/时序记忆控制。latent 局部编辑、自动
storyboard/Director agent 和 memory-reader LoRA 均不进入主线；只有免训练 action space 在 oracle
条件下仍失败时，才引入最小 memory-reader LoRA 作为诊断上界。

## 1. 研究边界与两个运行模式

### 1.1 主科学问题

研究对象是同一长视频内、跨 shot、长间隔重现的 character/object/location consistency，
主指标聚焦实体身份与 presence；实体状态变化作为防止“冻结换一致性”的受控诊断，不扩展成完整
叙事生成问题。重点不是让所有镜头彼此更像，而是同时满足：

- **身份不变量保持**：脸、服饰、物体形状、地点标志不无故漂移；
- **授权状态变化发生**：湿/干、完整/破损、拿起/放下等变化按脚本发生；
- **镜头与运动不塌缩**：不能靠复制帧、冻结运动或抹掉主体来换一致性；
- **因果性成立**：任何在线动作只能使用当时已经生成并可观测的信息。

### 1.2 双运行模式

| 模式 | 交付语义 | 允许动作 | 研究价值 |
|---|---|---|---|
| **S：strict streaming** | 帧生成后立即交付 | 只允许影响未来的 KV/memory/prompt 动作，不回滚 | 隔离 model-native 前向控制是否真的能抑制漂移 |
| **B：shot-buffered** | 当前 shot 验收后才提交 | boundary snapshot、最多 `R` 次局部重试、accepted-only memory commit | 面向实际离线/半在线 production，追求最终质量 |

模式 B 不允许改写已经 commit 的历史 shot，也不允许无限 best-of-N。默认先研究模式 S，证明
memory control 的机制价值；之后在相同动作空间上增加 rollback/commit，量化有限恢复权的增益。

### 1.3 在线因果契约

每个 shot 明确拆成两个时点，禁止把 post-shot 信息用于 pre-shot 决策：

```text
pre-shot planner(k)
  输入：shot k 的文本/entity schedule + shot < k 的 accepted history
  输出：shot k 使用的 memory、KV action 与剂量
            │
            ▼
       generate shot k
            │
            ▼
post-shot verifier(k)
  输入：shot k 的已生成 probe + 截至 k 的授权状态
  输出：memory admit/quarantine；模式 B 还可 accept/retry
```

- **模式 S**：shot k 已交付，post-shot verifier 不能改它；其结果只控制 shot k 是否进入长期记忆，
  以及 shot k+1 之后的动作。
- **模式 B**：shot k 在 buffer 中是 provisional；verifier 可在固定预算内 restore boundary state 并
  重试 shot k，通过后才交付并写入长期记忆。
- 若未来需要“先看 shot 开头再控制同一 shot”，必须单列为 chunk-buffered 协议；不能把当前离线使用
  entering-shot 前两帧的选择器直接称为 strict-streaming online controller。

## 2. 已有证据与关键风险

支持继续探索的证据：

- Round 18 首次出现越过逐场景配对 `2σ` 噪声底的数值胜利：`kv_only` 在
  `brown_bear_river`、`sandy_beach_driftwood`、`african_savanna` 上通过；前两个场景
  裁判与数值同向。
- `λ=1.0` 将锚注意力质量从 0.017 提升到 0.041，且记录到 1100 次 bias 调用；干预通道
  已经到达模型内部，不再是“功能没有生效”的问题。
- 60 个 boundary 计划产生 34 种独立帧集，52/60 使用中段帧，说明 VLM 选帧不是固定首帧启发式。

决定计划顺序的风险：

- **身份信息没有被正确搬运**：整帧 KV 更像在搬运色调、布局和环境统计量；在
  `tattooed_noodle_chef` 中甚至丢失人物。region/token 级选择是核心前提，不是后期优化。
- **尺子尚不可靠**：整帧 embedding 看不见人物细节，人物场景上裁判与数值系统性分歧。
  在实体级 metric 通过 degrade test 前，不再用新增渲染宣称身份提升。
- **agent 可能只是 test-time scaling**：若不与 matched-compute best-of-N、固定规则和静态计划比较，
  无法证明 agency 的价值。
- **无恢复权的上限有限**：模式 S 只能缓解未来漂移，不能修复已生成的灾难性叙事错误；因此模式 B
  是质量上限实验，而不是改变模式 S 的定义。
- **当前 runtime 还不是在线闭环**：Round 18 的 VLM 计划来自完整 baseline render，且选择器使用
  entering-shot 前两帧作 query；pipeline 也没有 snapshot/restore 或 verified commit 接口。新计划先
  证明 action 可控，再实现因果 hook，不能把离线 replay 直接改名为 online agent。

## 3. 系统形态：诊断—控制—提交闭环

```text
shot k spec + accepted memory
           │
           ▼
  pre-shot planner/controller
   ├─ select entity/region/history
   └─ select KV action and dose
           │
    [snapshot]  （仅模式 B）
           │
           ▼
     generate shot k
           │
           ▼
  post-shot verifier/critic
   ├─ identity / presence / state
   ├─ prompt / motion / diversity
   └─ fault location + confidence
           │
     ┌─────┴─────────────────────────────┐
     │模式 S                            │模式 B
     ▼                                  ▼
delivered already                  accept ──► deliver
admit/quarantine memory            reject ──► restore + retry
```

### 3.1 状态表示

每个 episode 建立显式 `entity-state graph`：

- entity registry：character / object / location 的文字不变量、可选 reference、首次出现位置；
- mutable state：每个 shot 前后的授权状态与禁止变化；
- visual memory：历史 crop、embedding、对应 frame/token/KV 索引、质量与可信度；
- generation state：当前 shot、local window、已 commit boundary、剩余干预/重试预算。

这能避免“保持一致”被错误理解为冻结一切，也为 EntityBench 的 entity schedule 提供直接映射。

### 3.2 Observer 与 critic

post-shot boundary 处只看当时已经可见的低成本 probe 帧，输出逐实体：

- `presence`：主体是否出现、是否被遮挡或丢失；
- `identity_drift`：外观是否偏离可信 memory；
- `state_error`：脚本要求的变化是否发生，或出现未授权变化；
- `quality_guards`：prompt adherence、motion、composition diversity、artifact；
- `fault_location`：shot、实体、区域、疑似成因及置信度。

先在 Round 18 已有渲染上离线回放，目标是识别**当前刚完成 shot** 的实体失败，而不是含混地
“预测未来失败”。threshold 在 calibration split 上确定，并用 leave-one-scene-out 或独立场景评估
AUPRC、校准度和 false-intervention rate；通过后才把它接入在线控制。

### 3.3 Controller 与动作空间

控制器先采用可审计规则策略；VLM 只负责语义诊断、实体匹配和困难 case 仲裁。不要一开始训练
policy，否则无法区分 observer、action 和 policy 各自的贡献。

| 动作 | 粒度 | 优先级 | 目的 |
|---|---:|---:|---|
| 历史锚帧/实体 crop 重选 | entity/region | P0 | 按当前主体检索，而非整帧相似度 |
| 自适应 logit bias `λ` | layer/head/token | P0 | 只在检测到漂移时增加历史锚权重 |
| region/token KV 注入 | entity/region | P0 | 避免背景统计量淹没人物身份 |
| RoPE 位置重映射 | token | P1 | 检验历史 token 被位置距离降权的假说 |
| memory pin/evict/quarantine | entity/token | P1 | 保留可信身份 token，隔离低质量或漂移 shot |
| rollback + retry | shot | 模式 B | 在固定预算内修复灾难性失败 |

所有动作记录 observation、理由、参数、实际注意力质量、成本和结果。两种模式都把“视频是否已交付”
与“是否获准写入 memory”分开：模式 S 的坏 shot 虽已交付，仍可被 quarantine；模式 B 的拒绝 shot
既不交付也不写入 memory。

在开发 observer/controller 前，先用 oracle entity mask、oracle reference 和人工指定 action 做
**controllability test**：如果 oracle 条件下 region/token KV 仍不能改善目标实体，继续堆 agent
没有意义。

### 3.4 Memory admission 与 rollback 状态契约

模式 S 在 M2 就需要 candidate-memory staging：clean recache 产生的 entry 先进入 candidate partition，
post-shot verifier 之后才 `admit` 或 `quarantine`。不能继续沿用“shot end 自动写 persistent memory”的
语义。

模式 B 的 snapshot/restore 是独立工程里程碑，snapshot 至少覆盖：

- 已 commit output latent 与当前 shot noise slice；
- positive/negative KV cache、cross-attention cache；
- KV-RAG transient/candidate/persistent partitions 与索引；
- shot index、RoPE offset、scheduler、CPU/CUDA RNG state；
- streaming VAE 的待解码状态与输出游标。

首先做无动作等价性测试：从 boundary snapshot restore 后重新生成，必须与直接从该 boundary 运行的
结果在预注册数值容差内一致；通过后才允许把 retry policy 接入实验。

## 4. Benchmark 组合

不使用一个 benchmark 承担所有结论。采用“内部机制诊断 + 外部长程主结果 + 通用质量 guard”
三层结构。

### 4.1 Tier A：冻结 hard multishot benchmark（开发主集）

继续使用 `example/benchmark_hard_multishot/` 的 9 个主场景 + 1 个负控制，完整规范见
`docs/benchmark_hard_multishot/BENCHMARK.md`。

用途：

- 快速做 paired multi-seed 机制消融；
- 直接暴露 LongLive `local_attn_size=32` 的遗忘问题；
- 用 watchmaker/noodle-chef 检验人物细节，用 negative control 检验 reward hacking；
- 保留 Round 13–18 的纵向可比性。

冻结集不再添加场景或修改 prompt。若需要补充 mutable-state、long-gap、双相似人物等受控因素，
另建 `benchmark_causal_entity_diagnostic_v1`，在看任何方法结果前冻结；它只用于机制分析，不能替代
外部 benchmark。

### 4.2 Tier B：EntityBench（论文外部主 benchmark）

[EntityBench](https://arxiv.org/abs/2605.15199) 含 140 个 episode、2,491 个 shots，覆盖
character/object/location、easy/medium/hard、最长 50 shots 与最长 48-shot recurrence gap；
[官方代码和数据](https://github.com/Catherine-R-He/EntityBench) 已包含 episode JSON、split 和三支柱
评估器。它应作为主要外部结果，因为它同时测量 intra-shot quality、prompt following 和通过
fidelity gate 的 cross-shot entity consistency。

EntityBench easy/medium episode 为完整的 10–22 shots，hard tier 固定 50 shots；当前 480-frame
rollout 不能合理承载完整 hard episode。接入时必须区分两套协议，不能把截断 episode 当作官方结果：

| 协议 | 生成方式 | 用途 |
|---|---|---|
| **EB-native** | 按官方完整 episode 逐 shot 生成 clips，跨调用保留 controller/entity memory | 与官方 evaluator 和已有系统比较；不作为“单次连续 causal rollout”的证据 |
| **EB-causal** | 选择能够完整放入当前或扩展 horizon 的 10–12-shot easy/medium episode，按原顺序串成一次 continuous rollout | 验证本工作的核心 causal claim；不得截断 episode 后仍称官方 full result |

接入步骤：

1. 将完整 episode JSON 转换为本项目的 `global.json`、逐 shot JSON 和 entity registry；
2. EB-causal 输出按 `shot_durations.txt` 切成官方要求的 `<episode>/<scene>_<shot>.mp4`；
3. 在看主方法结果前冻结一组**完整 episode**：覆盖三种 entity type 与短/中/长 recurrence gap；
4. 开发集用 seeds 0–2；episode 为统计单位，做 paired bootstrap 95% CI；
5. 固定 EntityBench repo commit、官方 evaluator、LLM judge 型号和 API 配置；本地 judge 只用于开发；
6. M2 先输出 full split 的 shot 数、预计 GPU-hours 与 judge 成本。full benchmark 只在预算明确且小规模
   gate 通过后执行；若只跑子集，报告中必须明确写 subset。

50-shot hard tier 是后续 scalability stress test：只有单次 horizon 或跨调用 persistent-state 协议验证
完成后才运行，不作为近期里程碑的默认交付物。

### 4.3 Tier C：VBench / VBench-Long（质量与运动 guard）

[VBench](https://github.com/Vchitect/VBench) 用于 subject/background consistency、temporal flicker、
motion smoothness、dynamic degree、aesthetic quality 和 prompt-video alignment。它不是跨 shot entity
recurrence 的主尺，只作为质量护栏：任何一致性增益若以运动冻结、低动态或严重画质下降为代价，
总 gate 判为失败。

只有调用官方实现的维度才能使用 `VBench` 名称；仓库已有的近似指标继续标为 proxy，并记录
backbone、coverage 与缺失项，避免把 RAFT/CLIP 自定义 guard 混报成完整 VBench/VBench-Long 分数。

### 4.4 直接竞争与可选模块 benchmark

| Benchmark | 使用方式 | 决策 |
|---|---|---|
| **GroundBench**（随 [GroundShot](https://arxiv.org/abs/2606.20799) 提出） | entity-grounded 受控诊断、与 accepted-memory agent 直接比较 | 代码/数据正式开放后接入；当前不阻塞主线 |
| [ViStoryBench](https://vistorybench.github.io/) | 单独评估 script decomposition、storyboard、entity registry/keyframe planning | 仅在 Director/Storyboard agent 成为贡献时使用，不评价视频运动 |
| [MuSS](https://arxiv.org/abs/2604.23789) | cinematic narrative、跨镜头匹配、anti-copy-paste | 若论文主张扩展到叙事/镜头语言再加入 |

## 5. 指标与统计协议

### 5.1 主指标

- **外部主指标**：M0 从 EntityBench 官方 evaluator 输出中预注册一个明确 JSON field 或固定
  aggregation，作为 fidelity-gate-corrected Pillar 3 主 endpoint；不能在 character/object/location
  或不同 judge 分数中事后选最优。Pillar 1/2 作为质量和 fidelity guards。
- **内部主指标**：masked entity consistency；SAM2/grounding 提取实体 crop，DINOv2/SigLIP
  评价 identity，有人脸时另报 ArcFace。整帧 embedding 降为 scene/background 辅助指标。
- **Recurrence-gap curve**：按实体重现间隔分桶，报告 consistency 随 gap 的下降曲线及 AUC，而不只报均值。
- **State-transition accuracy**：分别评价身份不变量和授权状态变化，防止“冻结换一致性”。
  该指标只在单独人工标注的 `benchmark_causal_entity_diagnostic_v1` 上使用：Tier A 没有 mutable-state
  标注，EntityBench `entity_schedule` 只标记每 shot 应出现的实体，也不能直接充当 before/after state
  ground truth。

### 5.2 Guards 与系统指标

- prompt/invariant/contrast adherence；
- RAFT/VBench motion、dynamic degree、temporal flicker；
- inter-shot composition diversity 与 copy-cheat 检测；
- boundary artifact、主体 presence、最低 shot 质量；
- 每 shot wall time、boundary latency、VLM 调用数、GPU-time、峰值显存、KV memory size、retry 数。

### 5.3 可信度要求

- 内部冻结集：所有 arm 与 baseline 同 seed 配对，seeds 0/1/2；历史逐场景 `2σ` 噪声底只用于
  Round 13–18 纵向兼容和工程 gate，不作为论文显著性检验；
- 外部集：episode 为统计单位，paired bootstrap 95% CI，同时给均值、最差分位数和分层结果；
- 论文主结论使用 scene/episode × seed 的 paired effect、bootstrap 95% CI 和 leave-one-scene-out
  稳健性；预先指定一个主 endpoint，其余指标按 secondary/guard 报告，避免多指标择优；
- metric 必须先过 degrade test：subject swap 降分、copy/freeze 被 guard 拦、合法视角变化不过度降分；
- 选择 threshold、子集和最大 retry 次数必须在看主方法结果前冻结；
- 任何缺失 guard、失败 evaluator 或不等算力比较均输出 `winner=null`，不补叙事性结论。

## 6. 必须完成的对照矩阵

| ID | 方法 | 回答的问题 |
|---|---|---|
| B0 | 原始 causal baseline | 基础漂移与质量 |
| B1 | static fixed whole-frame KV/cache policy | 简单 cache 管理是否足够 |
| B2-W/T | same-controller open-loop replay：在 baseline 上产生一次 whole-frame/token 计划后静态执行 | online feedback 是否优于相同 controller、payload 和剂量的开环版本 |
| F1 | frame-prefix/sink baseline | 给模型可见历史帧是否已足够；由于当前 backbone 无 reference-image 接口，不称为 GroundShot 等价实现 |
| O1 | oracle entity mask + oracle reference + static region/token KV action | action space 在理想选择下是否可控；这是进入闭环研究的前提 |
| A1-W | online rule controller + whole-frame KV | 在线 feedback 的基础效果 |
| A1-T | 同一 observer/controller + entity-token KV | token payload 相对 whole-frame payload 的净贡献 |
| A2-T | online VLM controller + 与 A1-T 完全相同的 action space | 语义 agency 相对规则 policy 的净贡献 |
| C-S | 与被测 A1/A2 相同 probe/VLM 调用预算，但 action 固定、随机或跨 boundary 置换 | 模式 S 的收益是否来自正确决策，而非额外调用/干预剂量 |
| C-B | fixed-R critic best-of-R，和 A3 使用相同生成次数与 judge | 模式 B 的自适应 rollback 是否优于普通 test-time scaling |
| A3 | shot-buffered adaptive rollback + verified commit | 有限恢复权与自适应预算的增益 |

核心因果主张只由单变量对照支持：

- **feedback 有价值**：A1-W > B2-W、A1-T > B2-T，且优于各自的 C-S；
- **entity-token action 有价值**：O1 先证明可控，随后 A1-T > A1-W；
- **VLM agency 有价值**：A2-T > A1-T；
- **adaptive rollback 有价值**：A3 > C-B，并报告相同生成/judge 预算下的质量—成本 Pareto。

模式 S 的交付语义禁止 best-of-N，但仍需 C-S 控制“更多 probe/VLM 计算”和“更多干预剂量”；同时
报告额外 wall time、GPU-time 和 boundary latency。模式 B 才与 best-of-R 做严格 matched-compute 比较。

F1 只是当前 backbone 上可实现的 frame-visible baseline，不能支持“model-native 优于 external
reference”的因果主张。若未来换用具有 reference-image 接口的同一 backbone，再增加固定 observer/
controller 下 external-reference vs native-KV 的配对实验。

若 A2-T 不优于 A1-T，论文应降格为“在线自适应 memory controller”，不要强行把 VLM agency
写成贡献。

## 7. 分阶段执行计划与止损点

| 阶段 | 工作 | 进入下一阶段的 gate |
|---|---|---|
| **M0：尺子与协议冻结** | masked entity/ArcFace、presence、degrade fixtures；冻结主 endpoint、pre/post-shot 信息边界和 manifests | swap/freeze/copy/合法视角测试通过；人物场景裁判-数值分歧可解释；任何 online 输入都可审计为当时已可见 |
| **M0b：遗留证据复核** | 用 M0 新尺子重评 Round 18 seed 0；只有信号仍在时才补 seeds 1–2 与 `λ∈{0,1,2}` | 仅决定旧结果能否继续作为 motivation，不阻塞新的 controllability test |
| **M1：oracle controllability** | 实现最小 region/token KV payload；人工 mask/reference/action，对比 whole-frame KV，暂不做 observer/agent | O1 在预注册实体指标上优于 whole-frame，guards 通过，token-level attention manipulation check 成立；否则立即止损 |
| **M2：causal runtime 与外部适配 smoke** | 实现 pre-shot/post-shot hook、candidate-memory admit/quarantine，禁止 entering-shot 泄漏；EntityBench converter、EB-native/EB-causal 各 2–3 个完整 episode；估算 full 成本 | runtime ledger 能证明每次决策的信息边界，rejected memory 不污染后续；adapter/evaluator 可 resume；完整 episode manifest 与成本预算冻结 |
| **M3：observer 可观测性** | 在已有视频上离线 replay post-shot verifier；calibration/test 分离或 leave-one-scene-out | 当前-shot failure 的 AUPRC/校准显著优于随机，false-intervention rate 可接受，threshold 不在测试集调参 |
| **M4：forward-only rule 闭环** | 比较 B0/B1/B2-W/B2-T/A1-W/A1-T/C-S；只使用 M1 已证明有效的最小 action | A1-T 相对 B2-T/C-S 的 aggregate paired 95% CI 排除 0，guards 全过，leave-one-scene-out 方向稳定；历史 `2σ` 结果只作辅助 |
| **M5：外部验证与 action 消融** | 在冻结完整 episode 子集跑 EB-native/EB-causal；再逐一测试 RoPE remap、pin/evict，不同时堆叠 | 官方 Pillar 3 和内部 masked-entity 指标同向；效果随 recurrence gap 不快速消失；明确报告 subset |
| **M6：VLM agency** | A2-T vs A1-T，保持 observer、action space、剂量上限和调用预算一致 | A2-T 在 paired CI 与质量—成本 Pareto 上优于 A1-T；否则保留规则 controller，不使用 agentic 主张 |
| **M7：rollback 基础设施** | 在 M2 candidate-memory 基础上独立实现 boundary snapshot/restore 与 verified video commit；固定规则 critic、最多一次 retry | 无动作 restore 与从 boundary 重跑数值等价；rejected shot 的 latent/KV/RNG/VAE/memory 均不污染后续结果 |
| **M8：恢复权与扩展验证** | A3 vs C-B；通过后再做 60s+、更多完整 EntityBench episode，50-shot hard tier 仅作条件性 stress test | matched-compute 下 A3 更优；完整报告 retry 分布、最差分位数、质量—成本 Pareto 与长度曲线 |

执行顺序上的硬规则：

- 顺序遵循 **measurement → controllability → causal runtime → observability → rule policy → VLM
  policy → recovery**；不得因为 agent 代码易写而跳过 O1。
- M1 只回答 action space 是否有效；M3 只回答能否观测；M4 才回答 feedback 是否有价值。三个阶段
  不合并，避免失败时无法归因。
- `benchmark_causal_entity_diagnostic_v1` 若用于 state-change，需在 M0 冻结 4–6 个带明确 before/after
  标签的场景；它不修改 Tier A 冻结集，也不替代 EntityBench。
- full EntityBench、50-shot hard tier 和 60s+ 只在 M4/M5 小规模 paired gate 通过且预算明确后运行；
  截断 episode 永远不能报告为 full benchmark。
- 长渲染一律在 tmux 中运行（harness 后台任务曾因会话生命周期与邻进程 OOM 两次被杀）；
- 每阶段提交 manifest、config、ledger、转录、失败 case 和可恢复命令，不能只保存汇总表。

**止损规则：**

1. M0 后实体尺子仍无法区分 swap 与合法视角变化：暂停方法研发，先解决评价问题。
2. M1 在 oracle mask/reference 下仍不能改善人物身份：当前 5B backbone 的免训练 KV 通道缺少
   有效读取能力；停止继续堆 agent，增加最小 memory-reader LoRA 作为上界或转向严谨负结果。
3. M3 observer 无预测力：不做 closed loop，退回可解释的固定 memory policy。
4. M4 只优于 baseline、不优于 B2-T/C-S：不能宣称 feedback/decision 有价值。
5. A2-T 不优于 A1-T：保留规则 controller；VLM 只作为离线诊断工具。
6. M7 无法无污染 restore：停止模式 B，论文只保留 strict-streaming 模式 S。
7. A3 不优于 matched-compute C-B：不宣称 adaptive test-time recovery 优势。

## 8. Related-work 边界与论文叙事

必须正面对比以下三类工作：

- **外部 entity memory / agent scheduling**：
  [GroundShot](https://arxiv.org/abs/2606.20799)、
  [VideoMemory](https://arxiv.org/abs/2601.03655)、
  [VideoGen-of-Thought](https://arxiv.org/abs/2412.02259)；
- **test-time self-improvement agent**：
  [VISTA](https://openaccess.thecvf.com/content/CVPR2026/html/Long_VISTA_A_Test-Time_Self-Improving_Video_Generation_Agent_CVPR_2026_paper.html)；
- **模型内部长期记忆**：
  [MemFlow](https://arxiv.org/abs/2512.14699)、
  [OmniMem](https://arxiv.org/abs/2605.30519)、
  [FilmWeaver](https://ojs.aaai.org/index.php/AAAI/article/view/37711)、
  [Memento](https://arxiv.org/abs/2606.14667)。

预期论文主张应是：

> 我们研究的不是一个更复杂的 storyboard agent，而是一个作用于 continuous causal video model
> 内部记忆的在线控制器。它在 shot 前基于脚本与 accepted history 选择 region/token KV 动作，在
> shot 后因果地验证实体并控制 memory admission；在 strict-streaming 模式中抑制未来漂移，在
> shot-buffered 模式中通过 boundary-local rollback 与 verified commit 避免错误进入长期记忆。

最终标题和“agentic”措辞由证据决定：若 VLM policy 没有超过规则 controller，就采用
**Adaptive Model-Native Memory Control for Continuous Causal Video Generation**；只有 A2-T > A1-T，
且模式 B 的主张还需 A3 > C-B，才采用
**Agentic Model-Native Memory Control for Long Video Generation**。
