# 渐进纠错实施记录：冻结历史、逐属性诊断与受控复现

最新检查点：官方 1.3B RAG 的六提示词开发试运行已实际导出 960 帧、16 fps、60 秒，五次提示切换的历史缓存审计通过；仍未接入纠错。身份控制实验中，Qwen 显式字段版错误接受不同角色，Molmo 结构化版有 unknown/解析失败，普通文字版还在纯灰色图像上编造面部。没有采纳新验收器，也没有改判 Iris。已更新论文、视频和可迁移的复现入口。详见 `timing-identity-controls-20260917.zh.md`（交付仓库中提供）。下面较早的运行中状态与“60 秒尚未完成”均为历史记录；当前仍非 submission-ready。

状态：开发阶段，**不满足 submission-ready**。完整目标仍是 `submission-goal-20260917.zh.md`；下面的阶段交付不替代正式实验、跨系统适配、独立评分或论文。

## 当前事实

- 在任何算法修改前保存 `videos/progressive_repair_20260917/source_pre_progressive`，303 个源码/配置文件带 SHA-256。它是本次开始时的脏工作区，不冒充历史原版。
- 重新校验历史 v1/original manifest 的 643 项、v2 manifest 的 357 项源码/输入哈希，均通过。归档源码及实验没有覆盖。
- 当前工作区和目标仓库均使用独立分支 `research/progressive-repair-20260917`。目标仓库原有 ICLR 2027 模板文件完整保留。
- 官方 LongLive-RAG 固定在 `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a`；官方 IAMFlow 固定在 `7dda1d3e02e061ad7440f18bd264dffa2dfc5bfa`。LongLive-RAG 的 base、LoRA、检索 AE 和 128 prompts 已从官方 `qixinhu11/LongLive-RAG` 固定 revision 下载，三个权重 SHA-256 与官方 LFS 哈希一致。官方单提示词开发样例已成功运行；它不是正式 baseline 质量成绩。
- IAMFlow revision `62ddeb21293edfd6b730715226c110067828d8e2` 的 `iamflow_fp8.safetensors` 在当前账户返回 HTTP 403。备用 `IAMFlow-FP8` 页面只有 README，不能替代权重。

## A：Clara 的 2×2 开发诊断

`scripts/run_repair_factorial.py` 在观察本轮结果前生成全部四组配置：归档 v1/v2 mask × v1/v2 prompt。实际 mask 张量从归档复制，种子 4243、σ=1.0、完整 replay 的源码、数据和其余参数固定。新目录拒绝静默覆盖或重试，记录运行 PID、失败/超时、GPU、墙钟、视频哈希。

四组均生成完成：466.8、481.5、472.2、470.7 秒。这是单 GPU 分配墙钟，包含加载，**不是核函数 GPU 时间，也不是包含全部检查器的总成本**。视频为 1280×704、24 FPS、384 帧、16 秒，是暴露开发案例，不是正式 60 秒任务。

旧 mask + 旧 prompt 复现的第 1 镜头 64 帧，与历史 v1 对应解码帧的逐像素 MAE 全为 0。这个检查只证明该镜头的动作复现，不证明画质、身份或后续正确。

第 32 帧四宫格显示：两种范围下，保留旧 prompt 的 tan-coat 强调句均生成棕褐外衣；去掉该句的两组仍为紫色外衣。这只是固定开发样例的目视线索。统一的冻结历史检查器正在对 baseline 和全部四组的第 1 镜头复评，不能提前把这张图当作总体有效性或后续保持证据。

本地产物：`videos/progressive_repair_20260917/clara_factorial/`，包含四个运行记录、`comparison_grid.mp4`、`comparison_frame32.png` 和 `archived_reproduction_check.json`。

## B：逐属性证据接口

新增 `utils/agent_attribute_diagnosis.py`，通过 `--comparer attributes` 显式启用。脚本要求在看候选画面前提取，使用原文连续片段及稳定 ID。分别记录部位、属性类别、期望、观察、pass/fail/unknown 和实际 crop 哈希；实体缺席时保留相同要求并标 unknown。分歧不投票变成 pass；相同图片不能重复计为多份证据。最多一次视觉复核，保存初次及复核结果，标明不是独立模型。

EntityDriftVerifier 保存逐属性结果及原视频帧坐标。明确违例优先于汇总 low-confidence 标签；纯低置信度注册条件改为 unknown。已确认 hair/garment 错误可进入对应现有区域构造。

第一轮 Iris probe 将颜色/发型合并，暴露提取粒度不足。其结果保留在 `iris_attribute_probe`，没有覆盖。后续只在该开发案例上增加 aspect（颜色、发型、衣物类型等）和已知遗漏检查，再冻结到 `source_b2_atomic_diagnosis`。

第二轮真实 probe 分开记录：头发颜色 fail→pass，garment/color 与 garment/type 仍 fail；发型被当前模型判为 pass。这不是独立真值，当前提取器也可能遗漏其它要求（例如宽泛的主体类型）。还没有成对身份/动作/画质验证，因此不能自动接纳该候选为安全部分改善或可信身份参考。

验证入口：`tests/test_agent_attribute_diagnosis.py`、`tests/test_agent_constraints.py`、`tests/test_agent_repair_planner.py`、`tests/test_agent_entity_judge.py`，以及现有 runtime、branching、appearance、region 测试。测试只覆盖控制与证据接口，不验证真实生成质量。

## 尚未完成及下一步

1. Clara 冻结检查器复评已完成：baseline 拒绝；两种 mask 配旧 prompt 均通过、配新 prompt 均拒绝。检查范围只有第 1 镜头，没有独立模型、全片或后续保证。完整记录在 `clara_factorial_evaluation/results.json`。
2. C 的离线实现和 113 项相关测试已完成，含 9 项专门的渐进事务测试；真实 Iris 首镜头试运行已启动。详见 `progressive-repair-stage-c-20260917.zh.md`。目前通过父 clean latent 重建缓存，不是逐位 KV/RNG 快照；真实改善累积仍待证实。
3. 实现 D：首次保留可靠通用动作，之后根据证据受限调整；真实 mask/prompt/seed 判断重复；父版本改变后重建区域并校验来源。
4. 已在隔离环境安装并校验与 Torch 2.9/CUDA 12 匹配的官方 FlashAttention 2.8.3 wheel，官方入口的导入与 `--help` 已通过。首个单提示词开发 pilot 已完成，332.64 秒生成 474 帧、29.625 秒、832×480、16 fps，记录 34 次 AE 记忆检索。官方分块解码每 60 latent 帧少输出 3 帧，正式时间安排须按实测解码帧对齐。多提示词适配与无纠错一致性检查仍未完成。
5. 等 IAMFlow 合法权重访问恢复后完成对应适配。继续登记开发暴露，开展 60 秒开发 pilot 并估算成本，之后才能冻结正式协议和规模。
6. 官方主表、同预算重试/best-of-N、必要消融、120 秒扩展、独立模型与匿名人工评价、脚本/模板聚类统计、论文及完整可复现发布均未完成。不得用当前测试或四组开发视频代替。

## 活跃任务与可恢复入口

本轮任务均在 `videos/progressive_repair_20260917`。检查 `*_launch.json` 给出的 PID 与实际进程/输出后再采取行动，不能仅凭旧状态文件重跑。`clara_factorial_evaluation/results.json` 为增量复评记录；`attribute_server/STOP` 仅用于正常结束本轮自己启动的视觉服务。GPU 3 的既有进程未触碰。

目标交付工作区为同级 `ICLR27-Longlive-agent`。阶段 README 保留 not submission-ready。阶段提交 `a190c0d` 已推送到指定仓库的独立分支；通过新建认证 SSH clone，逐文件核验全部已提交文件和四组视频的内容哈希。核验记录为 `remote_delivery_check.json`；这只证明阶段材料已交付，不代表总目标已完成。

## C 阶段开发检查点与暴露审计

阶段提交 `9ff51c0` 已推送并从认证远端重新获取，495 个已跟踪文件逐项哈希一致。导出的实现通过 113 tests + 9 subtests。Iris 新基线与历史归档 MP4 哈希相同，384 个解码帧逐项一致；首个 hair/color 候选正在真实生成，尚无改善结论。运行器 supervisor PID 2939248、driver PID 2939260，后续先核验其存活与 ledger 再操作。

`scripts/audit_development_exposure.py` 保守登记了 33 个开发脚本；全部 324 个 NarraStream 脚本共享至少一个已暴露的场景类别，12 类场景全覆盖。这只是场景字符串层面的重叠审计，尚非完整叙事模板分类；不能把余下 291 个脚本直接认证为独立模板留出。

本轮 SAM2 环境报告 native `_C` 扩展不可用，跳过 hole-filling 后处理；已保留警告并单独记录，正式环境尚需固定。共享 5B 生成器、文本编码器、VAE 与初始化权重的 13 个文件已计算内容哈希；这不是上游版本或许可审计的替代品。


最新官方回放检查点：见 [官方修复回放与定位诊断](official-repair-controls-20260917.zh.md)。OFF/ZERO 已通过完整状态/视频审计；原生非空回放仍运行中。本记录不代表修复质量结论。
