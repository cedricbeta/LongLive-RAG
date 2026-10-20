# Iris：自然服装表达的单次对照

本次实际生成表明，同一自动局部修复动作仅增加一句自然表达，能够改变中后段的服装形态；它尚未证明目标完整修复或安全保留。候选未进入控制器验收，没有改变此前两臂实验的零保留结论。[完整 60 秒四路视频](../artifacts/native_iris_natural_expression_v1/four_way_comparison.mp4)、[所有固定帧与原文](../artifacts/native_iris_natural_expression_v1/index.html)、[机器结果](../results/native_iris_natural_expression_v1/summary.json)及[复现命令](../reproduction/native_expression_control/README.md)均保留。

## 只改变实际提示词

基于已暴露 Iris 案例的失败证据，在固定第二候选的原提示后追加 `Iris wears a moss green coat.`。姓名、颜色和服装类型由既有自动编译脚本产生；`moss green coat` 是原脚本的连续文字，未增加隐藏属性、人工参考图或人工 mask。

父视频与原始 latent、自动 mask、编辑区间 [81,121)、力度 1.0、seed 4244、12 项生成资产和五个已安装原生源码文件均与原固定第二候选一致。实际动作唯一不同字段是 `prompt_sha256`。新提示保留全部原句和约束，并非删掉衣服以外的要求。与旧反馈第二候选相比，同样仅有提示词及派生哈希不同。

这是两臂实验结束后、看过开发失败后冻结的一次额外 replay，不是原 3000 秒预算对照的第三臂或正式消融。计划在生成前规定只运行一次，并固定帧 320/352/400/448/479/672/720/768 及沿用的 Molmo 观察协议，没有根据新结果修改种子或挑选采样时刻。

## 实际结果与边界

工程审计通过：前 81 个 latent 与原版精确一致，编辑区 mask 外精确一致；40 个编辑 latent 和 120 个后续 latent 改变。它们进入原生后续重放和检索状态，修改不限于导出的 MP4。这不等于身份、动作或画质保持。MP4 压缩后的前缀像素也不能直接当作 latent 精确性的替代证据。

新视频的八个固定帧、灰图和 400 帧原样复测共执行 10 次 Molmo 调用，全部输出合法 JSON。模型只见图片和既有观察提示，不见目标要求、分支名称或旧标签。灰图为零人，复测输入张量和生成 token 精确一致。

| 导出帧 / 秒 | 模型的衣服描述原文 |
|---|---|
| 320 / 20 | Black apron over white short-sleeved shirt |
| 352 / 22 | White shirt with green shawl or sweater, sleeves rolled up |
| 400 / 25 | Brown coat with belt |
| 448 / 28 | Green coat with rolled-up sleeves, white shirt underneath |
| 479 / 29.9375 | Brown jacket with rolled-up sleeves |
| 672 / 42 | Green jacket with rolled-up sleeves, white shirt underneath |
| 720 / 45 | Green button-up shirt with rolled-up sleeves, white undershirt visible |
| 768 / 48 | Green jacket with rolled-up sleeves, white shirt underneath |

八条衣服文本中两条含 `coat`，三条含 `jacket`，一条含 `apron`；旧四分支的 32 条文本均不含 `coat`。这是事后字符串清点，不是准确率、修复率或真实服装分类。颜色描述在 brown 与 green 之间变化，模型仍有观察误差。

开发检查实际固定帧时可见：320 帧仍为白色短袖加深色围裙，352 帧出现衣服转变，中后段出现带翻领的外套状服装。479 帧头发、花束等处的块状失真在原版、旧反馈和旧固定候选同帧也存在，不能记作本次新增缺陷。这些是开发过程中的模型辅助画面检查，没有真实受试者标签。完整动作、时序过渡、身份及非目标损坏仍未完成成对验收，不能据此采用候选。

帧 320 属于编辑区映射范围：latent 81 对应导出帧 320–323。简单的“它只是保留前缀”解释不成立；不能移除该失败帧。这里并未测量逐帧 VAE 因果影响。

## 实测成本

| 单独测量阶段 | 秒 | 设备能耗 |
|---|---:|---:|
| 原生准备与资产核验 | 21.203 | 未独立测量 |
| 原生生成 supervisor | 794.112 | 126.068–126.292 kJ |
| 十次观察 supervisor | 59.597 | 5.814–6.154 kJ |

生成中的局部修复阶段为 27.700 秒、56 次额外去噪调用，已包含在 794.112 秒内。观察 worker 为 34.073 秒，其中加载 14.654 秒，也已包含在 59.597 秒内。准备、生成、观察的阶段和为 874.912 秒，不是连续流水线重新计时。此前 35 次观察的 109.427 秒、旧两臂、原生成及自动编译/分割成本另行保留，不能省略成方法总成本。环境准备、下载及证据导出未计入这些阶段。

生成/观察 telemetry 分别有 388/29 个区间内采样，无查询错误或观测到的外部计算 PID；生成最大采样间隙 16.121 秒。能耗包含设备空闲，离散采样不排除短暂未观测进程，利用率积分不是 kernel 时间。

离线核验重建全部 960 帧的父子像素差异、40 个固定观察帧的输入及全部 45 次原文记录，检查自动 mask 实际字节与张量、动作、源码、模型清单和 telemetry。原始 latent 与权重未发布，离线核验不会重新执行生成或假装重新验证未发布 latent。后续应先对新候选进行固定的成对语义检查，再决定是否需要对自然属性表达做小规模方法试验；本例本身不足以冻结正式实验协议。
