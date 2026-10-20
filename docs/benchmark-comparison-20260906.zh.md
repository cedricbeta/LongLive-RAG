# Benchmark 对照与执行状态（2026-09-06）

本文件严格区分历史结果、新启动任务、尚未运行的长视频和因访问权限受阻的 IAMFlow。下面不是已经完成的新一轮全量结果。

## 方法与可比范围

| 方法 | 实现 / backbone | 现有 33 × 16 秒 | 新一轮状态 |
|---|---|---|---|
| A / LongLive | 本仓库 native sink，Wan2.2-TI2V-5B | 33/33，有历史完整指标 | 新 C 直接复用这 33 份 baseline 视频 |
| B / KV-RAG concat | 本仓库追加式记忆代理，同一 5B backbone | 33/33，有历史完整指标 | 保留为内部机制对照；不是 IAMFlow |
| 旧 C | 旧实体裁判 + masked renoise，同一 5B backbone | 33/33，有历史完整指标 | 仅作为历史实现，不作为正确累计修复的证据 |
| 当前 C | 现有 live judge + 累计修复 + H=2 + 不添人 | 新任务单独运行 | 输出 `videos/benchmark_20260906/iteration33/`；未有完整分数 |
| IAMFlow 官方 | 官方 FP8 checkpoint；Wan2.1-T2V-1.3B | 无本地实测 | 官方权重下载 403，等待授权或已有合法本地权重 |

不同 backbone 的 IAMFlow 对比是同任务、同评估协议下的系统比较，不能单独归因于记忆或修复机制。另列分辨率、实际时长、FPS、GPU 型号/数量、精度和吞吐。

## 现有完整指标表：历史 33 场景、16 秒、seed 0、Qwen3-VL-8B

11 个原始指标及 4 个汇总分原样取自 `narrastream_threearm_results_20260903/*_results_latest.json`。三个文件均有 33 个不同 sample_id；原始指标覆盖字段均为 33/33，但这不能保证 VLM 没有解析失败后的默认分回落。

| 指标 | A | B | 旧 C | 旧 C − A |
|---|---:|---:|---:|---:|
| subject_consistency | 0.9138 | 0.9187 | 0.9090 | -0.0048 |
| background_consistency | 0.9039 | 0.9070 | 0.9023 | -0.0016 |
| temporal_flickering | 0.8363 | 0.8347 | 0.8310 | -0.0053 |
| motion_smoothness | 0.2869 | 0.2853 | 0.2711 | -0.0158 |
| vtss | 0.6135 | 0.6162 | 0.6189 | 0.0054 |
| boundary_smoothness | 0.0100 | 0.0035 | 0.0086 | -0.0014 |
| conditional_adjacent | 0.6199 | 0.6505 | 0.6173 | -0.0026 |
| conditional_longrange | 0.7461 | 0.7703 | 0.7312 | -0.0149 |
| entity_grounding | 0.5911 | 0.5846 | 0.5929 | 0.0018 |
| dynamic_trajectory | 0.5483 | 0.5570 | 0.5459 | -0.0024 |
| vlm_score | 0.8066 | 0.7617 | 0.7955 | -0.0111 |
| quality_score | 0.7109 | 0.7124 | 0.7065 | -0.0044 |
| temporal_score | 0.4587 | 0.4748 | 0.4524 | -0.0063 |
| instruction_score | 0.6487 | 0.6344 | 0.6447 | -0.0039 |
| total_score | 0.6061 | 0.6072 | 0.6012 | -0.0049 |

旧 C 的非累计修复问题和裁判调用波动已经审计，见 [旧结果与审计更正](narrastream-threearm-results-20260903.zh.md)。这张表不证明当前 C 输，也不支持“人物缺席是唯一瓶颈”的解释。不要把 8B 迭代裁判、16 秒子集的绝对分与论文的 60 秒、30B 裁判主表拼接。

历史结果文件 SHA-256：

- armA_baseline: `93ef99accb69e415782c16d6f97a9aee9f3f0aa728479052b994539da586a4be`
- armB_concat: `fd86427e5b656a3cb39905bebbce71631265ecb3a93cc6b3e2d85bee0c0388e9`
- armC_entity: `0cacf99573822b5f58802111282dbed2e181f55c464319e7399e6932a4e2f9eb`

## 本轮已完成：五例固定目标执行器初查

五个无编辑控制均逐像素复现原视频；五次单次修复、全六镜头重放和全部固定抽样检查完成。四例出现清楚的目标属性收益，但伴随姿态/物体变化，不能称为 80% 成功率。ns090 有明显局部异常及后续延续，ns190 后期仍有发型/服装类别漂移。

完整观察见 [五例报告](machine-entity-repair-20260906.zh.md)。这是助手选择目标并检查结果的探索性机器观察，不是人工 oracle、独立盲评或最终 benchmark。

## 当前 C：冻结的迭代期复测

- 场景：现有全部 33 个，六镜头 × 八块；每条 16 秒，seed 0。
- baseline：直接复用已审计与历史 A 相同的完整视频，KV-RAG 关闭，不把重新采样差异算作 C 的收益。
- 控制器：当前 live entity judge；不接入此前校准未通过的实验候选，不再为这轮结果调提示/阈值。
- 与旧 C 相同的最多三次修复/镜头预算；sigma 1.0，累计编辑，H=2，`--place_absent never`。
- 控制器 temperature 0、seed 0、请求内容指纹缓存。三 crop 投票是现有策略，不宣称为校准后的概率。
- 冻结仓库源码、配置及数据指纹，每场景前后检查；独立新目录，逐场景失败日志，不自动换例/重试/删除失败。
- GPU 0 生成、GPU 3 控制器；GPU 1/2 的其他任务不动。
- 注意：归档 A 无 latent digest，首个修复分支沿用现有的解码帧容差前缀检查，不宣称 latent 逐位认证。外部依赖和大模型权重没有复制为隔离快照。
- 运行状态以 `iteration33/run.json` 为准。`generation_complete` 仅表示生成阶段结束，`evaluation_complete=false` 时不得宣布 benchmark 完成。

启动工具：[run_frozen_entity_benchmark.py](../scripts/run_frozen_entity_benchmark.py)。本轮相关回归测试 78 passed（不是全仓库测试）。

## VBench / 长视频评估

已获取官方 [VBench-Long 代码与说明](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/README.md)，固定 commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。隔离环境位于 `videos/benchmark_20260906/tools/vbench_env/`。

已通过六项指标的短片冒烟：subject_consistency、background_consistency、motion_smoothness、dynamic_degree、aesthetic_quality、imaging_quality。输入是 ns090 的 64 帧目标镜头（2.667 秒），不是整段或长视频，结果仅验证评估管线，不进入方法比较表。使用官方 `long_custom_input --dev_flag`，semantic splitting 关闭，官方 slow-fast 融合配置保持不变。

首次冒烟发现 torchvision 0.16.2 与继承的 av 17 不兼容；只在新环境固定 av 12.3.0 后通过，未改 metric 实现。该环境使用 torch 2.1.2 / torchvision 0.16.2 / transformers 4.37.2 / pyiqa 0.1.13；不是官方全部依赖的精确复刻。未使用的 xformers 扩展不匹配警告保留在日志。正式运行应冻结依赖清单及权重指纹。

| 数据档位 | 定义 | 当前是否有配套实测 |
|---|---|---|
| 迭代期 | 33 × 16 秒 | 有历史 A/B/旧 C；新 C 已排完整场景任务 |
| 原预注册终版 | 324 × 60 秒、官方 Qwen3-VL-30B-A3B 裁判 | 尚未启动；需要完整实体 spec 和匹配各方法渲染 |
| 120 秒扩展 | 同一批脚本/人物，在更长生成时域测试 | 尚未启动；不冒称官方标准成绩，不能循环拼接短视频充数 |

六项自定义长视频指标不是 VBench 官方 16 维标准 prompt suite 的“全套成绩”。NarraStream 的实体 grounding、条件长程一致性和指令遵循仍需一起报告。60/120 秒均应新生成，按各方法实际 prompt 切换时间切段，而不是给 IAMFlow 套用本仓库的 64 帧边界。

## IAMFlow 接入状态与研究含义

已获取 [IAMFlow 官方实现](https://github.com/Eddie0521/IAMFlow)，commit `7dda1d3e02e061ad7440f18bd264dffa2dfc5bfa`。代码本身已包含 LLM 身份登记、异步 VLM 核验及 identity-aware memory，因此不能只用“VLM in the loop”作为相对它的创新点。

官方说明下载目标 `Eddie0521/IAMFlow` 的 `iamflow_fp8.safetensors` 返回 GatedRepoError / HTTP 403，当前账号不在授权名单。源代码和 Qwen3-4B-Instruct-2507 已到本机；后者不是 IAMFlow 的生成器权重，不能据此宣布 IAMFlow 已可运行。没有使用代理方法、非官方替代 checkpoint 或论文数字填实测格子。

需要在 [官方模型页面](https://huggingface.co/Eddie0521/IAMFlow) 获得访问权限并在本机配置授权，或提供已获授权下载的本地权重路径。不要把 token 发在聊天里。A100 上 FP8 kernel 可能回退到反量化路径，必须记录实际路径，不照搬官方其他硬件上的速度。

## 接下来如何得到可信的最终表

1. 当前 C 33 场景完成后，重新准备 A/B/旧 C/当前 C 的统一评估输入，保留全部场景、失败、UNKNOWN 和歧义覆盖；独立评估器不读取控制器的接受结论。
2. 给最终评估器也加请求内容缓存/固定采样设置，并审计解析回落。控制器缓存已经有，不等于外部 NarraStream 评估器自动具备相同缓存。相同视频的评估噪声未控制前，不解释微小差值。
3. 报全指标及逐场景配对差值/区间；按外观违背、真实缺席、剧本允许/歧义离场分层，不只报总体均值。
4. 同列端到端耗时、GPU-seconds/video-second、VLM 调用数/耗时、修复次数、接受/回退率、后续退化。当前执行器每次重放整条视频，不是在线流式延迟；不能把重放成本全归咎于 VLM。
5. IAMFlow 权重就绪后按官方完整配置实测，再做匹配时长的 60/120 秒比较。官方 backbone 差异在表中显式标注；同 backbone 的机制结论由内部成对消融支持。

优先研究问题不是再加若干场景规则，而是：在固定执行器和计算预算下，VLM 反馈是否比无反馈/固定调度产生可重复的净收益；修复能否只改错误属性而保留动作；修好的属性能否跨后续镜头持续。需做的消融先冻结，再跑数据。五例和历史表目前不足以回答这些因果问题。

