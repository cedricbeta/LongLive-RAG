# 拒绝候选后的结构化证据分流

这次改动区分验收和下一步动作。`unknown` 仍不能算作改善，失败候选仍隔离；新增的可选路径只决定是否允许从保留的父分支提出一次不同的目标属性表达。默认不开启。

## 已确认的控制流程问题

此前 [Iris 原生候选](../results/native_iris_names_v6_window/summary.json)的两项衣服要求，从父视频的三帧 fail 变成候选的 fail/pass/pass。有限联合复核又返回 fail，原聚合规则因此保持 unknown。`improvement_decision` 正确拒绝候选，但后续停止分支把所有 violated→unknown 都称为 `existing_failure_became_unobservable`，无法区分跨帧标签不同与不可见。

本轮没有把这些标签升级为真实时间变化。尤其是已知的 coat 标签与解释矛盾依然存在；结构化标签不保证视觉判断正确。两组 [视觉输出协议对照](visual-checker-protocols-20260917.zh.md)均未采用。

## 新增路径的边界

程序按图像 SHA256 合并逐帧状态，保留同图判断分歧、跨帧标签混合、未知可见性及支持不足的区别。它不扫描自由文本中的 `low_confidence` 等词，也不把置信度当作触发器。

只有满足以下条件才开放这条路径：

- 父分支的每个目标仍为 violated，且至少两张不同图片给出一致 fail；约束定义未改变。
- 候选目标为 unknown，不同图片同时出现 pass 和 fail，没有 unknown 或同图分歧；同一组图片已经做过一次有明确状态的有限联合复核。
- 候选的其余 unknown 仅能是由这些目标派生、成员 ID 明确绑定的脚本属性汇总。身份、存在性、动作或其他独立约束的 unknown 均不符合。
- 父分支与固定原始分支的退化检查仍须通过；保护检查报错或 unknown 不走此路径。

允许的菜单仅有原提示词后追加原脚本目标值的表达变化，仍保留人物、动作和场景。没有新增掩码范围、没有纯噪声重试。相同父分支与目标至多执行一次这种变化；已经执行过的动作内容不能换文件名再次进入菜单。规划者可以选择停止。

首次配方、力度 1.0、真实 mask/prompt/seed 去重、预算、候选回退、成对保护及后续检查均继续生效。新元数据只补存联合复核状态及脚本属性汇总的成员 ID，不修改检查器状态、分数或要求。

## 已完成的验证

[原历史轨迹的离线回放](../results/mixed_evidence_contracts/archived_routing_replay.json)逐项核对了父/候选各七项约束的定义、状态、分数和违反记录，全部保持一致；原拒绝决定也保持一致。新规则将这一次情况分流到可提出受限动作，零新模型调用、零新视频、零人工标签。这不是新动作有效的证据。

[导出环境测试](../results/mixed_evidence_contracts/exported_tests.txt)为 239 项测试和 11 项子测试。新增事务测试覆盖：未知候选不保留、下一次从原父分支生成、身份记忆不提升、不同表达仅执行一次，以及不可见、同帧冲突、未完成复核、非目标 unknown、父缺陷支持不足等停止条件。测试使用构造数据，不能替代真实视频质量评估。

## 已冻结、尚未完成的真实对照

[反馈试验计划](../results/native_iris_mixed_evidence_v1/plan.json)和[固定重试计划](../results/native_iris_fixed_noise_v1/plan.json)使用同一原始父视频、编译器、检查器、自动定位和生成器。两臂均限制 3000 秒 worker 时间、最多两次生成、相同 seed 4243/4244 和一次后续检查。超过预算不再启动新操作，已在途操作可完成；渲染另有 1800 秒上限。

固定对照共享新的结构化证据门槛，但只保留原提示词和掩码、改变 seed，不调用规划者。其[两项软件测试](../results/mixed_evidence_contracts/fixed_control_tests.txt)验证没有属性表达或区域变化。它在查看反馈试验候选结果之前冻结，等待前一作业退出及 GPU 空闲后启动。报告实际成本，不将相同尝试上限写成相同实际计算量，也不虚构未花费的预算。

反馈臂首次实际动作与历史动作的父视频、父 latent、mask 张量、时序坐标、提示词和 seed 已[逐字段一致](../results/mixed_evidence_contracts/first_action_binding.json)。这仅证明首动作未改变，尚不是视频输出或质量结果。

真实自动试验的结果将在终态审计后加入此文；不得把已启动或尚未结束的作业记作完成。多次生成选择、独立评分和实际人工评价仍未完成。

另行复核了 IAMFlow 权重依赖：[带权重的官方仓库](https://huggingface.co/Eddie0521/IAMFlow)仍为 gated，实际文件请求返回 403；[公开 FP8 仓库](https://huggingface.co/Eddie0521/IAMFlow-FP8/tree/main)当前只有 `.gitattributes` 和 `README.md`，没有权重文件。[版本及访问记录](../results/iamflow_public_fp8_access_v1/access_check.json)保留这一差别，不能把公开说明页当成已取得模型。
