# Codex Goal v3 — Capsule pilot 终局双轨：G1b 最终估计器 ∥ 因果通道 mini-test

> 2026-07-16 · 续 v1/v2 goal。G1 的 kill 判决**对其编码的判据成立**（rank-matched 随机
> 子空间打平 Fisher，40 个 layer×rank 组合 CI 全跨零）——该结果保留，不推翻。但它检验
> 的命题（"存在跨场景通用、shot 池化表征下的低秩身份子空间"）比移植应用所需更强。
> 本 goal 预注册**最后两个探针**，双轨并行；**这是 pilot 的最终轮，没有 v4**。

## 终局决策矩阵（先定死，跑完对号入座）

| | G1b 通过 | G1b 仍平随机 |
|---|---|---|
| **通道有效** | 进入 v1 goal 的完整 P2 三 arm | 低秩线性 capsule 永久关闭；通道有效性记录在案，pivot 选项（非线性/learned capsule、整向量原型 steering）写进 future work，pilot 结束 |
| **通道无效** | （矛盾态，以通道为准）capsule 关闭 | capsule 关闭，双重确认 |

任何"再试一个估计器变体"的想法都被本预注册禁止。

## Track A — G1b：最终估计器迭代（零渲染，复用已有 dump）

修正 G1 的两个设计偏差，其余口径（白化度量、100 次随机对照、1000 次 bootstrap）不变：

1. **场景内实例对比，场景方差不得进入类间散度**：逐场景以 3 个 seed 实例为类做
   Fisher（类间散度 = 各场景内部类间散度之和或逐场景独立估计均可——移植应用是场景内
   的，逐场景子空间是应用忠实形态）。禁止再用 12 类 scene×seed 混合 Fisher（其前 11 个
   方向被场景轴占据，特征值秩上限 C−1=11 已证实这点）。
2. **token 级样本，不做 shot 池化**：样本 = 实体区域 token 的 V 向量（允许逐帧实体均值
   的轻池化）。统计上必须用 **cluster bootstrap（按 shot 聚类）** 防伪重复膨胀显著性。
3. **Gate（预注册，最终版）**：场景内 held-out shots（沿用 even/odd 切分）上，
   rank ≤ 8 的 P_id 同时优于同秩随机 null 的 p95 和正交补，cluster-bootstrap CI 不含 0，
   **≥3/4 场景成立**。LOSO 降级为诊断性报告，不再进 gate。
4. 层覆盖沿用 {0,7,14,21,29}；layer 14 是 G1 中身份可解码性最高的层，优先报告。

## Track B — 因果通道 mini-test（≤4 次渲染，与 Track A 并行、互不等待）

**问题**：原位编辑活跃窗口实体 token 的 V，对生成的身份**有没有任何因果效应**？
这与"身份是否低秩"解耦——P = I（整向量），不需要任何子空间，现在就能实现。

- **实现**：v1 goal P2 的 `capsule_edit.*` 机械，donor = 同场景 shot0 实体 token 池化 V，
  `v' = v + α(v̄_donor − v)`，P=I；编辑自该场景漂移最大切点起的所有后续 boundary。
- **4 个 arm**：2 场景（`sandy_beach_driftwood` 漂移最强 + `bearded_watchmaker_workshop`
  人物细节）× 2 编辑配置（仅 layer 14，α=1.0；全部 5 层，α=1.0）。seed0，与 baseline
  同 seed 配对。α 若致崩溃降 0.5 重跑一次并落账。
- **读出**（按证据强度排序）：
  1. **编辑到达性**：编辑首个 boundary 之后的帧与 baseline 是否逐位不同（SHA/帧差）——
     最低门槛，证明编辑进了计算图；
  2. 逐 shot 身份指标方向（现有 DINO subject 流）；
  3. guard 变化——**本测试中构图/运动泄漏不算失败，反而是通道有因果力的证据**；
  4. 人工目检材料：2-up 网格（baseline|edited，编辑点标色）+ 实体特写条，
     沿用 v1 P3 格式与 human_verdict 表。
- **判定**：4 个 arm 中任一出现"身份指标或人眼可见的主体外观变化"→ 通道有效；
  全部 arm 视频不变或仅琐碎扰动 → 通道无效。

## 运维修正（强制，吸收 2026-07-16 的三次事故）

1. **单驱动锁**：orchestrator/队列必须持有非阻塞文件锁（你已在加——验证其存在并写进
   REPORT）；任何 resume 脚本启动前检查锁。
2. **worker 重试语义**：遇到非零 exit_code 的配置，归档失败痕迹（`*.failed.<ts>`）后
   **重试一次**，再失败才 fatal——不再让单次邻居 OOM 杀死整条链。
3. **GPU 余量阈值提到 ≥60 GiB**（61440 MiB）：两次 OOM 都发生在 46–49 GB 余量下的
   VAE decode 峰值。
4. 长渲染 tmux、逐步落账、frozen 输入 SHA 审计照旧。

## 交付

更新 `docs/capsule_pilot_20260715/REPORT.md`：G1 节保留原判；新增 G1b 节与 channel-test
节（含 human_verdict）；末尾按决策矩阵给出 pilot 最终 verdict 与一段"证据链摘要"
（G0 重放透明性 → G1 弥散性 → G1b → 通道），说明每一步杀死/存活的具体命题。
