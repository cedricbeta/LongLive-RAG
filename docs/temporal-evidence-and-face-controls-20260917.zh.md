# 官方 Iris：跨帧证据冲突与脸部诊断

本轮是开发阶段修正，不是正式效果实验。官方 Iris 基线、原始错误通过结果、重算规则与控制图片全部保留。身份验收尚未解决，旧的 5B Iris 拒绝结果没有改判。

## 原始脚本与真实基线

使用 NarraStream 第 90 项原始六段提示词，每段 10 秒，seed 0；没有从旧的实体描述额外加入年龄、身材或参考图。官方 1.3B RAG 导出 960 帧、16 fps、60 秒，耗时 780.6240 秒，包含 259.7717 秒提示切换及历史哈希审计。五次切换均保持历史内容、游标及池长度，记录 78 次 AE 检索。[审计](../results/official_iris_60s_baseline_prepared/audit.json)和[视频](../artifacts/official_iris_60s_baseline_prepared/videos/rank0-0-0_lora.mp4)可查看。Iris 已用于开发，不能算留出样本。

第三段固定采样导出帧 352、400、448。四项要求逐字引用当前及更早脚本：red、curly、moss green、coat。这里只诊断单人物整帧中的外观，不声称完成角色跟踪或动作评测。

## 发现的错误通过链条

同一 Qwen3-VL-8B 开发检查器给出下列记录：

| 要求 | 三个单帧判断 | 旧的最终结果 | 同证据重算 |
|---|---|---|---|
| 衣服颜色 | fail / fail / pass | pass | fail |
| 衣服类型 | fail / fail / pass | pass | fail |
| 头发卷曲 | pass / pass / pass | pass | pass |
| 头发颜色 | pass / pass / pass | pass | pass |

前两帧明确描述为围裙加白色上衣。第三帧和联合复核却把围裙解释为 coat 的一种。旧聚合器将不同时间的混合结果统一视为判断分歧，随后让一次联合答案覆盖为通过。这同时暴露了模型对类别要求的错误解释和程序对证据的错误处理。

[原始逐帧观察及联合复核](../results/official_iris_baseline_attributes_gpu2/result.json)没有修改。[重算记录](../results/official_iris_attribute_temporal_fix/result.json)只复用相同的图片哈希、要求和单帧判断；零次新模型调用、零次重新生成。新规则针对整个窗口持续成立的外观要求：

- 两个不同图像中的明确冲突达到原有支持门槛后，其他时间的通过不能将它抵消；这不是多数投票。
- 同一图像字节被判断为互相冲突的状态时，该来源保持 unknown，不采用最后一次结果冒充独立票。
- 一次缺少逐帧归属的联合复核不能抹掉此前方向相反的明确证据；这种冲突保持 unknown。

这只修正证据处理，不保证模型观察正确。颜色判断仍可能受光照或颜色类别解释影响；没有人工真值，不宣称重算结果等于视觉真值。该修正来自已暴露开发案例，不可冒充留出验证。

更新通过 [43 项测试和 9 个子测试](../results/official_iris_attribute_temporal_fix/attribute_temporal_fix_tests_final.txt)，包括“更多通过帧也不能抵消两个明确冲突”、同字节矛盾和联合复核的反向覆盖。原始 42 项记录也保留。

## 脸部通道的有限进展

此前文字模型的身份控制出现过不同角色误通过、灰图编造脸部等失败。现增加一个未接入验收的 CPU 诊断，使用固定版本的 OpenCV YuNet 检测和 SFace 对齐嵌入，不读取脚本属性、修复目的或规划者解释。

原来的五个构造控制全部符合预期：两组同图约为 1.0；不同生成角色及反序比较的余弦分数均为 0.1662，低于固定示例阈值 0.363；灰图零检出，返回 unknown。另 24 个由四张原图构造的压缩、亮度、边框、遮挡、严重像素化、多脸及空输入控制也全部符合构造预期。

这些是程序控制，不是人工标注，不是 29 个独立身份样本，也没有总体误差率或真实跨姿态校准。具体图片、检测和分数见[控制记录](../results/face_evidence_controls/result.json)、[扩展记录](../results/face_evidence_stress/result.json)及[复现入口](../reproduction/face_evidence/README.md)。在官方 Iris 的三个诊断帧上，该检测器均为零检出；这不证明图中没有脸，只说明此通道不能提供身份保持证据。该项仍是 unknown。

## 执行与下一步

衣服 mask 来自程序人物区域裁剪后的 CLIPSeg clothing 查询，全图 face 保护不变，没有人工像素范围。40 个位置均非空，平均覆盖 18.3321%，CPU 耗时 204.8566 秒。[mask、参数和计划](../results/official_iris_garment_mask/)完整保留。它的边界没有人工真值，不能仅凭覆盖面积认定安全。

一次基线属性服务在其他任务进入 GPU 0 后中断；停止的是本任务自己的进程。GPU 2 上的新尝试完成，旧中断及清理日志没有覆盖。发现旧聚合结果为 pass 后，第一次候选准备因没有满足目标前提、未创建 prompt 而终止，没有执行生成。随后完成上述同证据修正，才冻结一次 garment/color 候选计划；不是重新采样原始诊断以寻找想要的答案。

该次官方候选现已完成，耗时 803.0706 秒；编辑范围为 latent [81,121)，对应 20–30 秒。前 81 个 latent 与编辑区外背景精确保持，40 个编辑 latent 和 120 个后续 latent 均改变；原生 AE/KV/检索记录相应改变。额外 56 次 denoiser 调用耗时 27.3747 秒，已包含在生成总时间中。压缩 MP4 的编辑前片段有 35 帧不同；本次没有单独核验原始 VAE 前缀像素，因此不把 latent 精确性写成已验证的视频像素精确性。[状态审计](../results/official_iris_garment_candidate/audit.json)、[并排视频](../artifacts/official_iris_garment_candidate/parent_candidate_comparison.mp4)及[预览](../artifacts/official_iris_garment_candidate/comparison_preview.png)均已保留。

生成后、查看候选结果前固定了诊断范围：当前帧 352/400/448，以及唯一后续窗口 40–50 秒的帧 672/720/768。沿用同一要求、采样和开发模型，三组外观检查加两次成对保护耗时 345.9080 秒，共 11 次调用。加上已完成的基线 4 次调用，共 15 次成功调用；中断客户端和服务开销另列于[机器可读总结](../results/official_iris_garment_candidate/summary.json)，不能只报告新增 denoiser 时间。

| 诊断 | 当前父版本 → 候选 | 后续父版本 → 候选 |
|---|---|---|
| 衣服颜色/类型，原始模型判断 | fail → pass | pass → pass |
| 头发颜色/卷曲，原始模型判断 | pass → pass | pass → pass |
| 成对保护 identity / non-target，原始模型判断 | fail / fail | fail / fail |
| 脸部通道 | unknown | unknown |

这些状态不是人工真值。后续父视频的三次外观判断再次把围裙扩展解释为 coat。当前保护把父版本旧问题混入候选退化；后续保护还把实际的衣服颜色目标说成头发颜色。两次 identity fail 的解释均表示脸部证据不足，与本应输出 unknown 的规则不符。动作、场景和画质在两次保护中输出 pass，也只是三个静帧上的不可靠开发判断，不能保证完整动作和运动保持。

独立于上述文字判断的 CPU 脸部通道，在两个窗口共 12 张实际图片上均零检出，耗时 0.4886 秒，两对均 unknown；没有把 unknown 改判通过。这里的“独立于文字判断”仅指单独的视觉计算通道，不代表已完成独立正式评价或身份校准。

本次选择仍为原父版本，保留改善数为 0，衣服颜色和类型保留为未解决项。候选和所有原始判断公开供检查。没有官方 progressive controller 的完整闭环结果，没有独立效果结论；后续仍需解决观察模型的类别与角色绑定问题，并验证身份、动作及同预算效果。初版汇总脚本漏计嵌套目录的调用，已从原始缓存更正为 15 次；错误版本、源码及勘误一并保留，没有重新调用模型或生成视频。

复算规则可直接运行，不需要 GPU 或 VLM 服务：

```bash
python implementation/scripts/reaggregate_attribute_evidence.py \
  --probe results/official_iris_baseline_attributes_gpu2 \
  --output /your/new-reaggregation
```

它使用既有 Python/PyTorch 检查器环境；输出目录必须新建。输出路径里的旧绝对路径仅作为原始来源标识，重算不依赖它们读取视频。
