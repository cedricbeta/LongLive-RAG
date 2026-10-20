# 论文表格与结果来源核对

核对日期：2026-09-17。本文的数字是**原论文报告值或明确注明的重算值，不是我们的复现成绩**。已核对主文方法/实验及评测相关附录；另渲染检查了 LongLive PDF 第 7 页、LongLive-RAG 第 8 页、IAMFlow 第 9 页、VideoRepair 第 6 页的表格。固定原文版本、PDF 哈希、关键数值与重算见 [机器可读记录](paper_table_audit_20260917/evidence.json)。

## 1. LongLive：三个实验，不能混成一个排行榜

依据：[2509.22622v1，§4、Tables 1–4](https://arxiv.org/html/2509.22622v1#S4)。

| 表 | 输入与评测 | LongLive 原文数值 | 对本项目的意义 |
|---|---|---|---|
| Table 1 | 官方 VBench prompts，5 秒；Wan2.1 1.3B，832×480 | Total 84.87 / Quality 86.97 / Semantic 76.47；单 H100 20.7 FPS | 短片能力和效率锚点，不能当作长片结果 |
| Table 2 | 自建 160 组脚本，60 秒，6×10 秒；全片 custom VBench-Long 质量、逐段 CLIP | Quality 84.38；六段 CLIP 28.85 / 25.68 / 24.64 / 24.23 / 24.32 / 24.32 | 需要同时展示整片质量与时间轴上的遵循度 |
| Table 3 | 官方 VBench-Long prompts，30 秒单提示词，官方切片 | Total 83.52 / Quality 85.44 / Semantic 75.82 | 不是 RAG 的 MovieGenBench，也不是 NarraStream |
| Table 4 | 10 秒视频在第 5 秒切换提示，比较三种 KV 策略 | No cache / unchanged cache / recache | 多提示词 baseline 必须正确处理旧文本条件 |

Table 1 的 Semantic 最高值属于 Self-Forcing chunk-wise（81.28），不是 LongLive；Table 2 的第三段 CLIP 最高值也不属于 LongLive。参考这些论文的论证方式，应明确每项优势及代价，不能把一行总分领先写成全面领先。原文没有给出这些主表的成对置信区间；不能从表内均值推断统计显著性。

## 2. LongLive-RAG：同 backbone、同上下文容量的增益

依据：[2606.02553v1，§4、Tables 1–3、附录 E/F](https://arxiv.org/html/2606.02553v1#S4)。其 Table 2 使用全部 128 条 MovieGenBench prompts，经 Qwen2.5-7B-Instruct 润色；分别生成 30/60/120 秒，在 Self-Forcing、LongLive、Causal-Forcing 内比较 base、∞-RoPE、Deep Forcing 和 RAG。

控制变量是各 backbone 内的采样设置与总注意力窗口 12。RAG 配置为 1 sink + 6 retrieval + 5 local，排除最近 5 个历史条目；不能让 Ours 多看历史，再把收益全归于纠错。检索编码器经过离线训练；准确表述为“不更新生成器”，不能称整个 RAG 方法无需训练。[官方固定配置](https://github.com/qiz1907/LongLive-RAG/blob/973884a3cd3ad4b314c3d4ab42274c52e7a0b22a/configs/longlive_latentmem.yaml)

下表只摘录与本项目直接相关的 LongLive block；六项分数均为原文百分制。

| 时长 | 方法 | SC ↑ | BC ↑ | MS ↑ | DD ↑ | AQ ↑ | IQ ↑ | Avg. Rank ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 30s | LongLive | 97.35 | 96.15 | 98.70 | 44.74 | 59.38 | 68.15 | 2.67 |
| 30s | + RAG | 97.53 | 96.39 | 98.77 | 44.84 | 59.24 | 68.42 | 1.67 |
| 60s | LongLive | 97.13 | 95.89 | 98.61 | 44.56 | 58.17 | 67.56 | 2.83 |
| 60s | + RAG | 97.32 | 96.08 | 98.62 | 49.90 | 58.30 | 67.79 | 1.33 |
| 120s | LongLive | 96.93 | 95.64 | 98.58 | 47.12 | 57.90 | 66.95 | 2.50 |
| 120s | + RAG | 97.22 | 95.88 | 98.62 | 50.25 | 57.57 | 66.95 | 1.33 |

SC/BC/MS/DD/AQ/IQ 分别为主体一致性、背景一致性、运动平滑、动态程度、美学、成像质量。Avg. Rank 是每个 backbone×时长 block 内四种方法的六项名次均值，用未舍入数值计算；它不是质量总分。增加 Ours 后必须在新的固定参赛集合内重排名，不能继承原文的 1.33。

值得针对性研究的是：120 秒 SC 仅增加 0.29 分，AQ 下降 0.33 分，IQ 显示精度下相同。30 秒 Self-Forcing 的 DD 从 52.03 降到 42.24。原论文有明确收益，也有权衡；我们应检验 RAG 后的残余错误，而不是声称 RAG 无效。

Table 3(a) 在 30 秒 Causal-Forcing 上比较 random、average-pool、AE-only、AE+SeqDelta 与完整编码器；3(c) 在固定窗口下比较 retrieval budget。对应我们应做“保留策略、反馈、后续检查”消融，而不是用模块数量证明有效。3(b) 的 VLM 分数尤其容易被误读：附录 F 用 Gemini 3.1-Pro 对随机抽取的 20 个 30 秒案例评价 **曝光质量，0–5 分**，不是实体正确率，也不是全面的指令符合度。

Table 1 的 490 ms/120s、4.08 ms/block 仅描述 retrieval encoding/search 开销，不是完整生成耗时。论文表中的分项 3.96+0.08 与总项 4.08 存在显示数值差异，照录并保留，不自行“修正”。六 A6000 约一周的主表推理评测、另约一周的检索训练数据生成，也不能合并成我们的推理速度基准。

## 3. IAMFlow：总分增益来自哪里

依据：[2605.18733v1，Table 2、附录 D/E](https://arxiv.org/html/2605.18733v1#S4)。324 条脚本，60 秒、6×10 秒，832×480；11 项指标分成三个组。以下三行全部来自该论文 Table 2，不包含我们的方法。

| 方法 | SC | BC | TF | MS | VTSS | BS | CAC | CLC | EG | DT | VLM | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LongLive | 95.36 | 93.32 | 90.27 | 57.76 | 77.29 | 49.19 | 83.06 | 89.97 | 62.40 | 63.33 | 62.22 | 73.17 |
| MemFlow | 95.37 | 93.21 | 90.23 | 57.37 | 77.67 | 47.06 | 83.92 | 89.71 | 61.93 | 64.14 | 60.85 | 72.88 |
| IAMFlow | 95.88 | 93.43 | 90.36 | 59.49 | 77.56 | 65.03 | 86.07 | 91.25 | 62.03 | 64.56 | 62.61 | 75.73 |

组内先做指标宏平均，再对三个组等权平均：

\[
VQ=(SC+BC+TF+MS+VTSS)/5,\quad TC=(BS+CAC+CLC)/3,
\]
\[
IC=(EG+DT+VLM)/3,\quad Overall=(VQ+TC+IC)/3.
\]

段内/段间分数此前还经过 narrative-aware 权重聚合；不能简单对 11 列求均值。EG 每提升 1 分只直接贡献 Overall 的 1/9 分。依据显示数值重算，IAMFlow 相对 LongLive：

| 量 | 差值/贡献 |
|---|---:|
| 原文 Overall 差值 | +2.56 分 |
| BS 差值 | +15.84 分 |
| BS 对 Overall 的算术贡献 | +1.76 分，约占总差值的 69% |
| EG 差值 | **−0.37 分** |
| VLM 差值 | +0.39 分 |
| IC 组分差值（重算） | 约 +0.42 分 |

这里的 69% 是聚合公式的算术分解，**不是 APT 的因果贡献估计**；模块归因必须看受控消融。它提示我们：如果研究的是实体纠错，必须把 EG 和独立错误审计列为主要终点，不能靠切换更平滑赢总分。也不能把 EG=62.03 直接解释成“37.97% 实体有错”；它是评分协议的输出，非二元缺陷发生率。

Table 4 包含去掉 VLM、LLM、ID memory、不同检索和分配策略、不同切换方式。IAMFlow 已有视觉核验和属性更新；“首次引入双模型”“首次视觉反馈”均不能成立。我们的可检验区别应是改写已经生成的错误状态、累积有效修复、限制新错误，并在真实总成本上优于替代策略。

### 3.1 同名指标与速度不能直接横比

IAMFlow Table 3 使用 NarraStream prompts 的 custom VBench-Long 六项：SC、BC、TF、MS、Overall Consistency、DD；它不是 RAG 的 SC/BC/MS/DD/AQ/IQ 六项组合。其 LongLive MS=17.202 与 RAG 的约 98 分没有直接可比性，应核对切片、实现、版本和归一化，不能挑一个作为“赢对手”的参照。

Table 5 报告 LongLive 218.43s、IAMFlow 156.69s；相对 LongLive 为约 1.39×，相对 Self-Forcing 264.86s 为 1.69×。附录 B 明确全部方法使用**双 H20**、warmup 后从 prompt encoding 到完整像素输出；FPS=12/平均 chunk 时间，是 block throughput，不是帧数/上述端到端时间。不能与 LongLive 单 H100 20.7 FPS、RAG retrieval-only 490ms 直接拼表。

复现存在需解决的配置差异：[公开默认 yaml](https://github.com/Eddie0521/IAMFlow/blob/7dda1d3e02e061ad7440f18bd264dffa2dfc5bfa/configs/iamflow.yaml) 设置 `use_tinyvae: false`、`async_vae_enabled: false`、FP8 DiT；论文加速配置使用 TinyVAE 和异步解码。设备分布也与附录描述不同。需要分别记录默认发布配置与论文速度配置，核验入口覆盖参数；不能未经核验把默认运行叫“论文速度复现”。

### 3.2 裁判、人工与权重可用性

主评测 judge 是 Qwen3-VL-30B-A3B，importance planner 是 Qwen3.5-27B；生成端是 Qwen3-4B + Qwen3-VL-2B。Table 6 固定视频、采样和规则，更换 GPT-5/Gemini-3.1-Pro；它支持 CAC/CLC/VLM 排序的稳健性，并不证明 IAMFlow 的 EG 领先。Table 7 是 30 个样本、30 名志愿者的**人工偏好与自动指标的相关性**，不是“人类认为修复成功”的成功率。[原论文 §4、附录 B/E/G](https://arxiv.org/html/2605.18733v1)

本次重新读取公开模型 API：`Eddie0521/IAMFlow` 为 gated=auto，列有 `iamflow_fp8.safetensors`；`Eddie0521/IAMFlow-FP8` 不 gated，但只列 `.gitattributes` 和 README。后者可访问不等于权重可下载。之前的 HTTP 403 是历史访问记录；本次未尝试凭据下载，不据此断言当前授权一定失败。[正式模型页](https://huggingface.co/Eddie0521/IAMFlow)、[FP8 页面](https://huggingface.co/Eddie0521/IAMFlow-FP8)

## 4. 相关论文必须影响实验设计

### MemFlow：不能省去 IAMFlow 的直接 backbone 对照

[2512.14699v1，§4、Tables 1–4](https://arxiv.org/html/2512.14699v1#S4) 使用 1.3B、3000 steps streaming long tuning；多提示词实验是自建 **100** 组 60 秒脚本，与 LongLive 的 160 组及 NarraStream 的 324 组不同。30 秒单提示词 Total/Quality/Semantic 为 84.51/85.92/78.87；单 H100 18.7 FPS。Table 3 对比 no memory、frame sink、NAM、NAM+SMA，NAM→NAM+SMA 的 throughput 为 17.6→18.7 FPS。

IAMFlow 使用 MemFlow 权重，故新主表需要 MemFlow 行；LongLive→IAMFlow 不是在同一 checkpoint 上只打开身份模块。论文自己也指出低动态视频会提高一致性，因此我们必须保留 DD/动作完成度及质量保护，不能把静止视频当作成功纠错。

### VideoRepair：必须比较已有的纠错方法

[ACL 2026 正式版本，§3、Tables 1–5](https://aclanthology.org/2026.findings-acl.1817/) 已有问题生成、视觉检测、自动定位分割、局部提示词、保留正确区域与候选筛选。Table 1 使用 EvalCrafter 的 count/color/action/others 四类划分；Table 2 使用 T2V-CompBench v1 的属性绑定、空间关系、数量三项。Wan2.1-1.3B 的 alignment 44.83→49.01，是 +4.18 **分**、约 +9.32% **相对提升**；CompBench 平均 0.5358→0.5646。

成绩来自同一个初始视频上的 **K=5 候选、一次 refinement**，不是一次额外采样；同分时再用 BLIP-BLEU 选择。VideoCrafter2/T2V-Turbo 为 16 帧，CogVideoX/Wan 为 81 帧；它没有在这里证明 60 秒 AR 状态修复。主要规划/判断模型为 GPT-4o，定位 MolmoE-1B、分割 Semantic-SAM。预算对照必须计入候选和模型调用，不能与我们的一次尝试直接比较。

可做两种明确命名的实验：原实现短片复现检查；保留其核心流程、接入相同 AR 生成器的 long-video adaptation。若只复刻部分机制，应称 VideoRepair-style control 并列出偏差，不能当作官方 VideoRepair。我们需要证明的新增价值是长视频状态和后续轨迹上的收益，而不是再证明“局部重画有用”。

### LongLive-2.0：本地 5B 的来源和规模混杂

[2605.18739v1，Tables 3–5](https://arxiv.org/html/2605.18739v1#S4) 的 5B、1280×720、NVFP4/GB200/异步 VAE 与上述 1.3B 设置不同。Table 4 的 BF16 4-step Total 为 85.06，NVFP4 2-step 为 83.14；45.7 FPS 对应后者，不应把最好质量和最高速度组合为同一配置。Table 5 的 60 秒六项表仍包含画质/动态的权衡。本地 5B 可作为规模补充，不能替代 1.3B 的成对机制证据。

## 5. 把原文证据变成我们的实验要求

1. 复用论文的比较结构：同一 backbone 上的开关比较、相同实际资源、完整指标、机制消融、不同时间长度和独立评价。
2. 保留三个任务族：NarraStream 60 秒叙事、MovieGenBench 30/60/120 秒单提示词、纠错方法的受控预算比较。不同任务族各自成表。
3. 所有 paper-reported 数字仅在上述参考表中出现；新主表统一在冻结数据和评测器上重跑，并另列 paper/reproduced gap。
4. 不把源码默认 seed/num_samples 当作论文实际多种子清单。原文未明确、配置不一致或无法获取的部分写成复现缺口，先做开发验证。
5. “优于他们”必须具体化：在同预算下减少实体错误，有独立证据，且不以换人、减动作、破坏后续或增加失败为代价。正式规则见 [实验协议](paper-comparison-protocol-20260917.zh.md)。
