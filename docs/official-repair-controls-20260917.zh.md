# 官方 1.3B RAG 修复回放与定位诊断

这是开发记录，尚未证明纠错净收益，也不是留出实验。官方生成器已经接入可选的原生 masked replay；progressive controller、可靠验收和正式实验尚未完成。既有 Iris 5B 候选仍为拒绝、保留改进数仍为零。

## 实际通过的 OFF/ZERO 对照

父版本是已公开的 ns310 Clara/Miles 六段、每段 10 秒实测视频。配对保持官方 base、LoRA、AE、文本编码器、VAE、tokenizer、种子和时间协议一致，12 个资产逐文件验证。

| 运行 | 墙钟秒 | 额外修复去噪调用 | 与父版本的 latent / 视频 / 检索 / 切换状态 |
|---|---:|---:|---|
| repair OFF | 831.6224 | 0 | 全部一致 |
| 第二段 zero mask | 955.2772 | 56 | 全部一致 |

两次都是完整 60 秒、960 帧；241 个 latent 和所有解码帧逐项比较。78 条检索记录、五次提示切换的 CPU 历史池和 AE descriptor 内容哈希、resident K/V、游标及池长度也与父版本一致。ZERO 的区间是 `[41,81)`，包含非空的 41-latent 前缀和 160-latent 后缀。14 个编辑 block 都真实执行四步去噪，并确认全局随机数状态不变、原作者文本的 cross-attention 精确恢复、mask 外 latent 精确保持。

[完整审计](../results/official_repair_controls/audit.json)保留逐帧误差数组和来源哈希。墙钟包括载入、全片 replay、全历史哈希审计和导出；不能把 OFF/ZERO 的总耗时差直接解释为修复内核耗时。ZERO 的单独修复阶段合计为 26.9556 秒，已包含在总耗时中。每次只分配一个 GPU，报告的 allocated GPU seconds 是保留设备时间，不是 kernel 时间。

## 实现边界

非空自动 mask 的[实测摘要](../results/official_nonzero_replay_gpu2/summary.json)与[完整状态审计](../results/official_nonzero_replay_gpu2/audit.json)也已完成，墙钟 897.3767 秒，额外 56 次去噪。前 41 个 latent 与编辑段 mask 外精确保持；25 个被覆盖的 latent 改变，全部 160 个后续 latent 改变。67 个 block 的 clean AE descriptor 内容与 zero control 不同，编辑后的四次切换所见历史池、resident K/V 均不同，检索记录也发生变化。这证明改动进入原生状态及后续生成，不能说明变化有益。

该次特意沿用定位不足的全图 coat mask，作为非空状态传播测试；没有验收器判定，没有执行后续质量 guard，也没有保留候选。[完整并排视频](../artifacts/official_nonzero_replay_gpu2/parent_candidate_comparison.mp4)及[固定时间预览](../artifacts/official_nonzero_replay_gpu2/preview.png)均标注为开发观察。编辑前有 39 帧压缩像素不同，尽管前缀 latent 精确相同；本次未另行审计原始 VAE 前缀，不能声称前缀像素完全保持。

首次 GPU 0 尝试在 admission 阶段拒绝启动；新目录中的 GPU 2 尝试通过无其他 compute PID 的检查后完成。首次离线审计出现字符串/Path 类型错误，失败日志与源码保留；只修复审计入口并重读同一份产物，没有重新生成视频。

修复入口验证父 latent、MP4、配置、时间安排和 mask 的来源；只接受对应于一个完整作者片段的 block 对齐范围。实际 mask tensor、prompt、seed、范围及父 latent 共同构成动作指纹。mask 仍由外部定位程序提供，准备器不会把一个形状合法的 mask 标记为定位正确。

原有去噪调用仍逐 block 执行，以保持全局 RNG 时间线。前缀在 clean recache 前替换为父分支 clean latent；编辑段使用官方四步 x0/re-noise sampler，力度固定为 1.0，随机噪声来自私有 Generator。每步将 mask 外约束到对应父分支；结果进入原生 AE descriptor 和 clean KV recache。作者文本条件恢复后正常继续后缀生成。

这仍是有显著成本的全片回放，不是 KV/RNG 快照加载，也不声称实时流式。官方路径尚未接入自动选择、部分改善保留和后续质量验收。工程状态正确不等于视频内容正确。

## 定位失败及单因素诊断

以下均使用同一父视频第二段、同一 pinned CLIPSeg、阈值 0.35、全图 woman 区域与 face 保护，以及每 latent 对应四帧的 temporal union。坐标遵从连续 VAE 加首帧裁去协议，不沿用本地 5B 的八帧 chunk 映射。

| 衣服查询与输入上下文 | 空 mask 数 / 40 | 平均网格覆盖 | CPU 墙钟秒 |
|---|---:|---:|---:|
| clothing，全图 | 30 | 0.4908% | 172.956 |
| coat，全图，仅改名词 | 15 | 2.4251% | 187.087 |
| clothing，程序人物裁剪 | 0 | 7.2312% | 199.082 |

[机器可读汇总](../results/official_mask_components_diagnostic/summary.json)和[四个组件的热图](../artifacts/official_mask_components_diagnostic/components.png)显示：抽样帧 160、240、316 中，clothing 主要响应另一位角色的服装；三个位置的 face 排除都未删掉 entity/part 交集。现有证据不支持为了扩大 mask 而删除脸部保护。

人物裁剪版本只改变部位模型的输入上下文。box 从同一 entity 阈值图自动得到，外加一格边缘；全图 entity/face、查询 clothing、阈值和投影不变。抽样覆盖了目标外套，但没有标注真值、分割精度或安全边界指标。手部、衣领和时空边界仍可能混入错误，不能将“非空”直接当作“可接受”。这也不是既有 SAM2 人物跟踪路径的等效替换。

首次 context 运行因 inference tensor 原地更新报错，耗时 6.1496 秒；[失败状态](../results/official_native_mask_context_control/status.json)和当时源码完整保留。修正为构造新 tensor 后，在单独目录运行；没有覆盖旧记录。

## 复现与尚缺证据

[安装器复现](../results/official_repair_controls/installer_reproduction.json)确认干净官方源码加安装器得到所有相同源码哈希。[准备入口检查](../results/official_repair_controls/preparation_check.json)验证配对资产和 OFF/ZERO 配置，未启动第二轮生成。当前 23 个契约测试通过，原始 21 个测试记录另外保留。

[复现步骤](../reproduction/official_repair/README.md)使用用户自己的目录与资产。历史 manifest 的绝对路径只保留来源，不是唯一执行入口。完整 raw latent 较大，由重新执行生成并校验；OFF/ZERO 视频与已公开父视频相同，保留审计而不重复上传文件。

正式的独立检查器、官方 RAG 逐项纠错验收、反馈动作、IAMFlow 适配、同预算比较、留出模板、长单提示任务、真实人工比较和统计分析仍缺失。IAMFlow 权重访问 403 只阻塞该依赖，不意味着整个任务完成或停止。
