# Codex Goal v2 — Capsule pilot 解封：修正 G0 判据、诊断 bear、继续 P1

> 2026-07-16 · 本 goal 是 `docs/capsule_pilot_codex_goal.md`（v1）的续篇。v1 的所有
> 设计决定、gate 定义（G1/G2/G3）、P3 人工验证要求继续有效，本文只修订 G0 并给出
> 解封步骤。已有产物在 `videos/capsule_pilot_20260715/` 与 `docs/capsule_pilot_20260715/`。

## 0. 对 v1 结果的裁决（人工审计结论，直接采纳）

v1 的 `verdict: kill` **被推翻，重新分类为 G0-infra-blocked**。理由：

1. 3/4 场景解码帧 SHA-256 与存档逐位一致（带 probe dump）——重放机械与 probe 透明性
   实际上已被证明；它们被判 fail 是因为"存档指标重算 delta > 5e-05"，但帧逐位相同时
   指标差只能来自指标重算环境漂移（open_clip QuickGELU 版本警告即证据），该容差对
   跨环境重算不可达，是 v1 判据设计缺陷；
2. overhead 判据比较的是 15 天前 `stage_a.log` 的 wall-clock，本机共享、负载不可比
   （测出 −11% 的"负开销"即自证无效）；
3. 唯一真异常是 `brown_bear_river`（SHA 不一致、delta 0.322、熊侧面变背面），
   原因未诊断。

**新规则（写进最终 REPORT 的协议节）**：`verdict=kill` 只允许由 G1（无线性身份子空间）
或 G3（capsule ≤ random / 泄漏与 full-vector 相当）触发；G0 类基础设施失败一律记
`blocked` + 诊断，不得记 kill。

## 1. G0 判据修订（替换 v1 的 G0）

- **透明性（硬门）**：probe-on 与 probe-off 的解码帧 SHA-256 逐位一致。
  对 watchmaker / chef / driftwood：存档即 probe-off 产物，v1 已证一致 → **直接判过**，
  不重跑；
- **存档指标对比降级为 sanity check**：容差放宽到 1e-2，只报告不设门；解码 SHA 相等
  是更强的检查，已覆盖；
- **overhead**：删除跨天 wall-clock 对比。可选做一次同会话背靠背配对计时
  （probe-on vs probe-off，同 GPU 紧邻运行），报告数值，门限 <25%，共享机负载异常时
  记录并豁免。

## 2. brown_bear_river 诊断（一次 ~7 分钟渲染 + 决策树）

用现行代码、stage_A 的 bear config、seed0，**probe-off** 渲染一次，然后三分支：

- **A：probe-off == probe-on 重放 ≠ 存档** → 存档 mp4 过期（Round 18 期间 watchdog
  多次崩溃恢复，存档可能出自异配置运行）。处置：以本次 probe-off 渲染为 bear 的新参考
  基线（P2 的配对比较本来就用同批次 baseline，不依赖旧存档），在 ledger 记录 provenance，
  G0 判过；
- **B：probe-off == 存档 ≠ probe-on** → probe 钩子扰动生成（怀疑方向：RNG 消耗、
  torch.compile 图变化）。这是真 bug：修复后重跑 probe-on 直至逐位一致，才继续；
  修复过程与根因写进 REPORT；
- **C：probe-off 两边都不等** → 场景级非确定性。先查 determinism 设置与 compile 缓存；
  若一次内不可解，**换场景**：用 `african_savanna`（同为非人物主体，baseline 漂移 0.41）
  替换 bear，重做该场景 P0（probe-on 重放 + dump + mask QC），并在 REPORT 显著标注
  替换及原因。

## 3. 解封后继续（回到 v1 流程，不改 gate）

1. **补 seed1/2 的 probe 重放 + dump**（4 场景 × 2 seeds，~1 小时 GPU，tmux）——
   跨 seed 负对是 P1 的必需数据；3 个干净场景的 seed0 dump 已在盘上，直接复用，
   **不要重跑**；
2. P1 子空间估计 → G1（判据照 v1：held-out 身份可分性 + leave-one-scene-out，
   不过 G1 才允许 verdict=kill）；
3. G1 过 → P2 三 arm 干预（capsule / full-vector / random-subspace，α 扫描）→ G2/G3；
4. P3 人工验证材料照 v1 全套（4-up、shotstrip、实体特写条、mask QC、human_verdict 表，
   观看顺序与"数值-人工矛盾以人工为准"规则不变）。

## 4. 产物与账本

- 更新 `docs/capsule_pilot_20260715/REPORT.md`：v1 版本标注 superseded，新 REPORT 记录
  G0 修订判据下的结果、bear 诊断分支与证据、后续各 gate；
- 复现命令、config manifest、SHA ledger 照 v1 惯例逐步落账；
- 约束不变：冻结 benchmark 只读、新代码 flag 默认关、长渲染 tmux、单 seed 干预实验
  不作论文级主张。
