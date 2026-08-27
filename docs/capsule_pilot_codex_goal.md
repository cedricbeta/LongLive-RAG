# Codex Goal — Typed Memory Capsule 可行性 pilot（kill-test）

## 背景与假设（不要重新论证，直接执行）

本仓库是 Wan2.2-TI2V-5B 因果视频扩散（30 层，`wan_5b/configs/wan_ti2v_5B.py`），
`local_attn_size=32` 帧局部窗口，每潜帧 880 token（`wan_5b/modules/causal_model.py:307`）。
长多镜头生成中跨 shot 实体身份漂移；此前 18 轮实验证明整帧 KV 注入只搬运场景统计量、
注入列被降权 4–20×（账本：`docs/retrieval-key-value-study.md`）。

**假设**：历史 KV 纠缠了 identity/pose/camera/background 多个因素。若在冻结模型的
clean-context KV 上存在低秩线性"身份子空间" P_id，则只移植身份分量（原位编辑，
不追加列）应能改善漂移身份，且不复制旧构图/动作。

**本 pilot 回答一个问题**：这个假设值不值得做成论文？产出 verdict ∈ {promising, kill}。

## 已定死的设计决定（不要更改）

1. capsule 的估计与移植都在 **clean-recache 的 KV 的 V 向量**上（不动 K，不动 RoPE）；
2. 移植 = **原位编辑活跃窗口内实体 token 的 V**，绝不追加新列（避开已知的降权问题）；
3. donor 用**池化原型**：`v'_i = v_i + α · P_id · (v̄_donor − v_i)`，v̄_donor = donor 实体
   token 的均值；P_id 的度量结构（欧氏 vs 白化/斜投影）由 P-1 的方法移植调研定稿，
   但"低秩、闭式、无训练"不变；
4. 所有对比 arm 使用**相同的投放机制**（都原位编辑），只变子空间；
5. 泄漏指标报向量（motion/background/diversity 各一条），不加权合成标量。

## 素材（已核实存在）

- 冻结 benchmark prompts：`example/benchmark_hard_multishot/`（**只读，禁止修改**）；
- 三 seed baseline 视频：`videos/benchmark_hard_multishot/stage_A/baseline_seed{0,1,2}/`
  （480 帧/场景，命名 `baseline_seedS-rank0-<scene>-seedS_regular.mp4`）；
- 本 pilot 用 4 个场景：人物 = `bearded_watchmaker_workshop`、`tattooed_noodle_chef`；
  非人物 = `brown_bear_river`、`sandy_beach_driftwood`；
- KV 注入/存储机械：`utils/kv_rag.py`（`frame_aligned_store`、boundary 注入路径；
  仪表参考 `record_attention_mass` 的"采样 query 行 + einsum 旁路"模式——
  flex_attention 是 torch.compile 的，**禁止在 flash 路径内挂 hook**）;
- 推理入口：`python inference.py --config_path <yaml>`，基础配置
  `configs/inference_kv_rag_long_multishot.yaml`；
- 逐 shot 指标参考实现与 Round 18 数值存档：
  `docs/round18_bench_20260701/iter01/numeric/`（per-shot anchor_to_shot0 DINO subject /
  CLIP background 流）；shot 切点见各场景 `shot_durations.txt`。

## P-1 — 方法移植调研（半天–1 天，纯推导，先于一切编码）

先研究三个成熟方法家族的做法，产出 `METHOD_TRANSFER.md`：一张映射表，把我们的每个
设计选择对应到一个已验证的方法模板，并据此定稿 P1/P2 的公式细节。若有网络就核对原文，
没有就依据下面的摘要（细节引用留到写论文时再核）：

**A. LLM 线性概念结构与 steering**

- *Steering vectors / ActAdd / RepE（representation engineering）*：用对比 prompt 对
  （有概念 vs 无概念）在某层取激活差的均值作为方向向量，生成时以系数 α 加到该层
  hidden state 上；实践要点 = 逐层扫描找最优注入层 + α 扫描 + 用困惑度等副作用指标
  防止过度干预。→ **映射**：我们的移植公式本质是 target-conditioned steering；
  照搬其"层 × α 二维小扫描 + 副作用监控"的实验纪律（副作用指标 = 我们的 guard 组）。
- *LEACE / INLP（概念擦除）*：闭式仿射投影移除表征中某概念的全部线性信息；LEACE 的
  关键是**先按协方差白化再投影**（最小范数、不扭曲其他方向），裸欧氏投影会因特征
  间相关性误伤无关因素。→ **映射**：我们的 P_id 应该是白化度量下的投影（Fisher 广义
  特征分解天然给出 Σ_within^{-1} 斜投影，与此一致）；移植操作在白化坐标系里做，
  变换回原坐标后再写入 V。P-1 要把这个公式显式写出来。
- *ROME / MEMIT（locate-then-edit）*：因果追踪定位承载事实的层，再做低秩权重编辑；
  我们不改权重，但**照搬其评估三件套**：efficacy（编辑生效 = 身份恢复）、
  generalization（跨场景/跨切点成立）、specificity（不伤邻域 = 我们的泄漏向量）。
  G3 的报告结构直接用这三个词。

**B. Diffusion 语义空间编辑**

- *Asyrp / h-space*（U-Net bottleneck 的语义方向）：编辑只加在预测-x0 分支（非对称
  应用），且随去噪步长有 schedule（早期步 = 粗语义）。→ **映射**：我们在 clean-recache
  KV 上编辑，**天然绕开了 timestep-schedule 问题**（上下文 KV 不随 σ 变）——这要作为
  设计优势写进 METHOD_TRANSFER；若未来扩展到生成期 hidden state 编辑，再借它的
  schedule 技巧。
- *StyleAligned / ConsiStory（training-free 共享注意力做 subject 一致性，图像侧）*：
  跨图共享 subject token 的 self-attention K/V 实现同主体生成；用 cross-attention map
  自动得到 subject mask。→ **映射（最近的机制亲戚，必须写进定位）**：我们的整帧/region
  KV arm 就是它们在因果视频上的类比；capsule = 升级为"只共享 V 的身份子空间分量"。
  另外它们的 **cross-attention mask 技巧直接可偷**：用身份词的 cross-attention map 做
  实体 mask，可能比 CLIPSeg 更贴合模型自己的 token 归属（见 P0 修订）。
- *IP-Adapter / PhotoMaker 等*：身份保持但需训练 adapter——列为 related、不可移植，
  在 METHOD_TRANSFER 里标注为"我们的 trained 对照上界"。

**P-1 交付**：`METHOD_TRANSFER.md` 映射表 + 定稿后的移植公式（白化版）+ G3 报告
按 efficacy/generalization/specificity 组织的字段定义。

## P0 — 仪表与重放（gate G0）

1. 实体 mask 三选一，按此优先级验证：① 模型自己的 cross-attention map（身份词 →
   空间 token，ConsiStory 式，与 token 归属天然对齐、无需外部模型）；② transformers
   的 CLIPSeg；③ 人工框（每 shot 一个粗框，pilot 精度要求低）。抽 几帧人工目检 mask
   质量并存进报告。像素/注意力 mask → 880-token 潜格映射要写清楚。
2. 实现 KV dump（config 开关 `capsule_probe.*`）：确定性重放 baseline seed0 × 4 场景，
   在层 {0, 7, 14, 21, 29} dump clean-recache 阶段实体区域 token 的 V（fp16、只存实体
   token、可按帧下采样，单场景 dump 控制在几 GB 内）。
3. **G0 判据**：重放的逐 shot 指标与 stage_A 存档一致（数值容差内）；dump 开销 <10%。

## P1 — 身份子空间估计（gate G1，纯离线，不渲染）

配对构造（全部来自已有/P0 数据，零新渲染）：
- **同身份 × 不同姿态**：同场景 shot 内不同帧 + 漂移前早期 shot 间（32 帧窗口内身份稳定）；
- **异身份 × 相似语境**：同 prompt 跨 seed（baseline_seed0 vs seed1/seed2 的同场景同 shot 位，
  seed1/2 需要补做 KV dump 重放）+ 跨场景（4 个身份互为负对）。

估计：每层对实体 token V 做 within-identity / between-identity 协方差的广义特征分解
（Fisher/cPCA 风格，闭式，无训练），取 rank ≤ 16 的 P_id。

**G1 判据**（held-out 验证，leave-one-scene-out 或 held-out shots）：
- 至少一层上，P_id 内的身份聚类/分类显著优于其正交补和同秩随机子空间；
- 姿态/时间变化的方差主要落在正交补。
**不过 G1 → 立即停止，写报告，verdict=kill（理由：无线性身份子空间）。**

## P2 — 移植实验（gates G2/G3，唯一的渲染开销）

实现 config 开关 `capsule_edit.*`：在指定 shot boundary，对活跃窗口内实体 token 的 V
做原位编辑（公式见上），α ∈ {0.25, 0.5, 1.0} 先在单场景试稳定性再定一个 α 跑全部。
donor = 该场景 shot0 实体 token 池化 V（从 P0 dump 读取）。编辑发生的 boundary 选
该场景 Round 18 numeric 记录中漂移最大的切点起的所有后续 boundary。

**三个 arm，4 场景 × seed0，同 seed 与 baseline 配对**（共 12 次渲染 + P0 的重放；
长渲染一律 tmux）：
1. **capsule**：P = P_id（G1 选出的最优层）；
2. **full-vector**：P = I（整向量复制——检验"子空间选择"是否是增益来源）；
3. **random-subspace**：同秩随机正交 P（安慰剂）。

评估（逐 shot，与 baseline 配对）：
- 身份：现有 DINO subject anchor_to_shot0 流（人物场景另加人脸 crop 的 DINO 相似度，
  insightface 未安装，不要求 ArcFace）；
- 泄漏向量：RAFT 运动 delta、CLIP background delta、inter-shot diversity delta；
- 现有 guard 全套照跑。

**G2 判据**：三个 arm 无 NaN/崩溃/明显 artifact（α 扫描内至少一档稳定）。
**G3 判据（最终 verdict）**：
- promising = capsule 在漂移 shot 上身份优于 baseline 与 random-subspace，
  且泄漏显著小于 full-vector（full-vector 预期表现为 diversity/motion 下降或构图复制）；
- kill = capsule ≤ random-subspace（子空间无因果作用），或 capsule 改善身份的同时
  泄漏与 full-vector 相当（分解失败），或只在单一场景有效且无法泛化。

## P3 — 人工可视验证材料（必交，G3 的最终裁决靠人眼）

数值指标（DINO/CLIP 流）已知看不清人物细节，本 pilot 的 promising/kill 裁决必须辅以
人工目检。为每个场景生成以下四类材料（格式沿用 Round 18 的既有产物，参考
`videos/round18_bench_20260701/grid_comparisons/` 与 `docs/round18_bench_20260701/qualitative/`）：

1. **4-up 网格视频** `videos/capsule_pilot_<date>/grids/<scene>_4up.mp4`：
   2×2 = baseline | capsule / full-vector | random-subspace，同 seed 同步播放；
   每个 shot 切点画竖线标记，**被编辑的 boundary 用不同颜色标出**并烧字幕注明
   （层号、α、rank）；各象限左上角烧 arm 名。
2. **逐 shot 首帧长条图** `qualitative/<scene>_shotstrip.png`：
   4 行（每 arm 一行）× N 列（每 shot 首帧），列头标 shot 编号、行头标 arm——
   一眼看出身份/构图随 shot 的演化差异。
3. **实体特写条** `qualitative/<scene>_entity_crops.png`：
   用 P0 的实体 mask 从每 shot 中段抽 1 帧裁出主体区域放大，同样 4 行 × N 列——
   身份判断（胡须/纹身/熊的毛色斑纹/浮木形态）必须看特写，整帧看不清。
4. **mask 质检图** `masks/<scene>_mask_qc.png`：每 shot 抽 1 帧叠加 mask 半透明轮廓，
   人工确认 mask 没框错对象（mask 错 = 后面全部作废）。

**人工检查清单**（逐场景填写，结果表进 REPORT.md 的 `human_verdict` 一节）：

| 检查项 | 看什么 | 期望（promising 签名） |
|---|---|---|
| 身份恢复 | capsule 行的实体特写：后段 shot 的主体是否回到 shot0 的样子（watchmaker 的胡须/眼镜、chef 的纹身/头巾、熊的毛色、浮木形态） | capsule 优于 baseline，且优于 random |
| 构图/动作泄漏 | capsule 行的整帧：后段 shot 的镜头角度、主体动作是否仍跟随各自 shot prompt（对照 `N.json`），还是复制了 shot0 的构图姿态 | capsule 不复制；full-vector 出现复制/冻结（这是子空间起作用的反证据） |
| 运动质量 | 4-up 视频实速播放：capsule 象限是否变呆滞、冻结、重影 | 与 baseline 无肉眼差 |
| 编辑点伪影 | 被标记的编辑 boundary 前后 1 秒：闪烁、色偏、鬼影、主体消失（noodle_chef 旧失败模式） | 无新增伪影 |
| 安慰剂 | random-subspace 行 | 与 baseline 无肉眼差（若 random 也变好/变坏 = 实验有混淆） |

每场景约 3 分钟目检（4-up 实速一遍 + 两张条图）。REPORT.md 中每项给
{better, same, worse, broken} 四档 + 一句话证据（指明 shot 编号）；**数值 G3 与
human_verdict 矛盾时，以人工结论为准并如实记录矛盾**——这正是本项目"裁判-数值分歧"
的已知问题，不许用数值单方面判 promising。

## 交付物与约束

- `docs/capsule_pilot_<date>/REPORT.md`：G0–G3 逐项结果、`human_verdict` 表、verdict、
  失败案例帧、每次渲染的 config/manifest/可复现命令；中间数据与 P3 全部可视材料放
  `videos/capsule_pilot_<date>/`（REPORT 里逐一给出相对路径链接，标明"人工必看"清单
  与建议观看顺序：先 mask 质检 → 实体特写条 → shotstrip → 4-up 实速）；
- 所有新代码走 config flag，默认关闭；不修改冻结 benchmark、不动 stage_A 存档；
- 这是单 seed 探索性 pilot：报告里明确写"不构成论文级证据"；
- 预算意识：渲染总量 ≈ 12 + 重放；若 G1 就失败，渲染开销为零；
- 时间盒：一周量级。遇到与"已定死的设计决定"冲突的实现困难，在报告中记录并提出
  替代方案，不要静默改设计。
