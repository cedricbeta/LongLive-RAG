# 官方 VBench-Long 六项指标：单条 120 秒基线开发结果

已完成官方 MovieGenBench prompt 0、seed 0 的原始 120 秒视频评测。使用官方 VBench `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` 原始 CLI，模式为 `long_custom_input --dev_flag`。这六项是官方支持的自定义长视频视觉指标；不包含完整标准提示词套件，也不评价脚本实体约束或纠错收益。[官方说明](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_long/README.md)

| 指标 | 官方输出 |
|---|---:|
| subject_consistency | 0.979060 |
| background_consistency | 0.954936 |
| motion_smoothness | 0.984275 |
| dynamic_degree | 0.983333 |
| aesthetic_quality | 0.615749 |
| imaging_quality | 0.657658 |

[机器可读结果](../results/official_vbench_moviegen000_summary/summary.json)保留完整精度、各阶段成本和输入哈希。[原始视频](../artifacts/official_moviegen_000_120s_v1/videos/rank0-0-0_lora.mp4)没有修改；评测前后 SHA256 均为 `2c5bd9f2165db1e0f05a11fccbed439d9cbd54b706f8d575c8fd9517d7c53fe1`。

样本量为 **1 条已暴露开发视频**。官方切片实际沿用源视频 16 fps：60 个两秒片段，每段 32 帧。两种 fast 视频各含 60 个片段首帧。官方重组记录中的路径由片段名派生，不应据此把 60 条记录当作 60 条独立视频。审计从所有记录重算六项聚合，检查 imaging 的归一化和 subject/background 的融合，未更改官方打分代码。这里不报告置信区间或自行合成总分。

首次真实尝试在 12.840 秒后因 Torchvision 0.16.2 与 PyAV 17.0.0 的 `pict_type` 接口不兼容而失败，尚未产出分数。[失败日志](../results/official_vbench_moviegen000_120s_v1/evaluation.log.txt)及设备成本完整保留；该接口变化也见 [PyAV 上游讨论](https://github.com/PyAV-Org/PyAV/discussions/1666)。通过合成视频的编解码检查后，新尝试仅将 PyAV 改为 12.0.0，成功耗时 252.672 秒。[恢复绑定](../results/official_vbench_moviegen000_120s_v2/recovery_binding.json)核对其余协议一致，历史失败没有被覆盖。

两次评测共 265.512 秒；另计原始生成 1001.283 秒，分阶段相加为 1266.795 秒。这不是一次新计时的连续端到端流程，未包含环境安装、下载和材料导出。成功评测 NVML 设备能耗区间为 35.726–36.007 kJ，126 个区间内采样，无查询错误或观测到的外部计算进程。能耗包含空闲功耗，利用率积分不等于 kernel GPU 时间。

[复现入口](../reproduction/vbench_long/README.md)包含两个尝试的离线复核、原始 CLI 的启动与监测命令、459 项冻结文件检查、14 个模型文件的来源和校验值、实际源码快照、9 个模型的 CPU 前向预检及全部安装失败。当前有效环境是已有诊断环境上的独立可写 overlay；实际六项指标已运行成功，但尚未验证干净独立环境安装器，不能把全局 `pip check` 的未解决依赖说成通过。离线导出复核不重新执行模型特征提取。

本结果补上长视频基线的官方视觉评测。仍没有纠错前后独立语义比较、正式留出统计或真实人工标签；Iris 两臂仍为零保留改善，不能用本表替代核心方法证据。
