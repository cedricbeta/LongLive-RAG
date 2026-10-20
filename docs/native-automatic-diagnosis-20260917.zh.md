# 官方原生自动诊断接入：真实失败与定位探针

本检查点没有新增视频生成或保留改善。自动脚本编译仍不可靠，官方完整 diagnosis → generation → paired acceptance 尚未跑通。以下全部使用已暴露的 ns310 开发脚本，不是留出评测。

## 实现与边界

[`NativeEntityVerifier`](../adapters/official_longlive_rag/entity_verifier.py) 将一个虚拟 chunk 映射为一个真实导出帧，复用多人联合角色匹配、SAM2 跟踪和逐属性检查。三个实际帧用于角色匹配，SAM2 跟踪全部 160 帧；CLIPSeg 在对应人物轨迹框内分割部位。程序将每个原生 latent 对应的四帧区域求并，再排除同期面部与其他人物的保护区域。最初 warm-up latent 使用第 0 帧。区域绑定父视频、父 latent、timing 和源 mask 的哈希；外观通过不能晋升为身份参考。

[`agent_script_entities.py`](../implementation/utils/agent_script_entities.py) 从原始六段文字提取角色、出场时间和逐属性要求。原文引用、完整词边界、时间持续性和覆盖检查能拦住部分错误，**不能证明语义归属正确或脚本没有遗漏**。[`run_native_progressive.py`](../implementation/scripts/run_native_progressive.py) 接入原生事务后端、有限反馈和当前/后续保护；当前三个真实运行都在文字阶段失败，没有调用生产生成器。

## 全部实测记录

| 运行 | worker 秒 | 实际模型调用 | 结果 |
|---|---:|---:|---|
| [8B joint v1](../results/native_auto_multientity_v1/run/status.json) | 18.79 | 1 | 提议原文没有的 straight hair 等属性，引用校验停止 |
| [8B joint v2](../results/native_auto_multientity_v2/run/status.json) | 54.97 | 1 | 输出达到 4096 tokens，JSON 截断；未进入视觉阶段 |
| [8B quotes/atoms v3](../results/native_auto_multientity_v3/run/status.json) | 13.99 | 3 | 外观引用混入动作；一次受限修正后仍不覆盖。另有 long coat 错归 long hair |
| [30B 同协议对照](../results/native_script_compiler_30b_v3/run/compiled_script.json) | 12.48 | 3 | 程序检查通过，但把 buzz cut 当头发颜色、光照短语当发型；未采用 |
| [定位探针 v1](../results/native_multientity_localization_v1/run/status.json) | 27.83 | 1 | 资源检查包装器缺少 cache_dir，首次角色匹配前停止 |
| [定位探针 v2](../results/native_multientity_localization_v2/run/result.json) | 96.47 | 6 | 修正接口后完成真实 160 帧多人跟踪与区域导出；无修复判断 |

四种编译设置不是预注册比较研究；它们是顺序开发诊断。30B 对照只改变 checkpoint，保留 v3 输入、提示词、schema 和校验规则。通过程序检查但语义错误的输出完整保留，不能宣称解析成功。共 15 次真实模型调用，所有失败计入记录。worker 时间包含自身加载和执行，server 启动/退出另列；并发时间不能直接相加成端到端耗时。没有测量 kernel GPU time，GPU 3 未使用。详见[机器可读汇总](../results/native_entity_contracts/development_summary.json)。

## 定位结果及限制

定位探针使用此前模型给出的原文角色描述，单独检查定位链路，不消费失败的原子属性输出，也不渲染。四个区域各覆盖 41 个原生 latent，均非空；平均占比分别为 Clara hair 1.5631%、garment 6.9203%，Miles hair 1.3286%、garment 12.8647%。[实际数组审计](../results/native_multientity_localization_v2/audit/audit.json) 确認四帧并集减保护区的精确映射和零保护区重叠；这不是 mask 真值精度。

[区域预览](../artifacts/native_multientity_localization_v2/region_preview.png) 显示 Clara 衣物在第 80/159 帧只剩小块肩部区域，Miles 头发越过可见头顶边界。Clara 衣物源区域在帧 0/80/159 为 434/81/45 个网格单元，这三帧后续保护区排除量均为零，不能把缩小归咎于最终保护区排除。[诊断记录](../results/native_multientity_localization_v2/run/selected_frame_region_diagnostic.json) 保留这一观察；它不是人工研究标签，也没有区域真值。SAM2 可运行，但缺失可选 `_C` 扩展，hole filling 被跳过，日志和环境版本均保留。

导出的实现通过 [132 tests + 9 subtests](../results/native_entity_contracts/exported_tests.txt)，包括原生坐标、父分支、缓存失效和资源包装器的实际编号图路径。测试不验证属性分类、图像判断或视频净收益。

## 人工评测准备

[参与者包](../artifacts/human_blind_development_packet/participant_packet.zip) 包含现有两个官方原生非零诊断候选与各自原始父视频的匿名 A/B 对比，每段完整 60 秒。选择范围是本轮自动试运行之前全部两个已完成的此类开发候选；这些候选不是已验收的最终交付结果。界面展示原始六段脚本，分别收集外观/场景、身份、动作、画质与总体比较，允许平局和无法判断，仅本地下载 JSON。

[清单与校验](../results/human_blind_development_packet/manifest.json) 记录随机侧别密钥的承诺哈希；密钥不在仓库或参与者包中。四段视频均为 832×480、16 fps、960 帧、60 秒，哈希、ZIP 和 JavaScript 语法已核验，未做真实浏览器展示检查。**实际标注者 0、人工标签 0**。尚待用户提供可用标注团队/平台，未向任何人发送材料。不要将模型结论或工程目视观察写成人工评分。

下一步需要验证有类型约束的属性提取、定位范围可靠性和成对保护；之后才能推进自动真实生成、同预算对照、正式留出协议与独立评估。IAMFlow 权重访问和完整官方实验仍未完成。
