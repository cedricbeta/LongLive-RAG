# 缺席实体：改写 prompt 整镜头重采样（2026-09-13）

## 为什么换动作

09-11 的 C-new 在 33 个场景上跑完，闭环碰过的段有可测的正收益，全集平均只动 1 分。
复盘时发现两件事把提升压在评测的盲区里。一是裁判和评分器是同一个 Qwen3-VL-8B，
颜色否决让裁判看见了 8B 看不见的走样，评分器却还是那个 8B，修对了它也不给分。
二是评分器最能反应的是实体在不在（EG），而闭环对缺席一处都没修：新裁判在 A baseline
上找到 16 处缺席，C-new 跑完还剩 21 个缺席镜头没解决。

缺席之所以没修，是 09-04 的放置实验：在锚点旁边画一个人形框，σ=1.0 masked renoise
四个种子，一个人都没长出来（docs/narrastream-constraint-smoke-20260903.zh.md）。当时的
结论是"文本通道就是丢人的那条通道"。但那次测的是局部框：框外的内容在每一步都被钉
回去，框内要凭空长出一个人来和周围打架。整镜头重采样是另一个问题：整个镜头的 latent
从噪声重画，文本条件拿到整个镜头，周围没有东西可以打架。这一轮就测这个。

## 动作

裁判判定有 required 实体缺席时（`--absent_action regenerate`），不再找框，而是：

1. 让裁判自己的语言模型改写这一镜头的 caption：缺席实体放到第一句做主语，带上剧本
   描述；caption 里有位置关系（"to Nora's left"）就写进去；在场的人、场景、动作都保留，
   不加人不减人，90 词以内。文本调用，走同一个 VLM 服务，同一个内容指纹缓存。
2. 改写结果做校验：每个缺席实体的名字必须出现，原 caption 里在场的名字不能丢，不能
   超长。不过就记下原因，退回一个固定模板（"Iris, woman with red hair, is clearly
   visible in the frame, to Noah's left. <原 caption>"）。改写来源（llm / template）、
   被拒的改写和原因都进 metrics。
3. 发一个 edit_shot：mask 是这一镜头每个 chunk 的全 1 网格，entry_sigma 1.0，prompt
   是改写后的 caption。pipeline 的 masked renoise 在全 1 mask、σ=1 下退化成"从父分支的
   KV 状态出发重画整个镜头"：x 直接等于噪声，每步的背景钉回对全 1 mask 无效，
   cross-attn 缓存用改写后的 prompt 重新填。pipeline 一行没改。
4. 每轮尝试用不同的噪声种子（后端 `noise_seed + attempt`），第二轮改写时把前几轮的
   改写发给模型要求换一种说法。预算还是 `max_edits_per_shot`（3 轮）。前瞻两个镜头、
   最近稳定分支回退、前缀一致性检查全部沿用。

有缺席时这个动作优先于放置和冒名顶替者重画；只有走样没有缺席的镜头走原来的属性
mask 路径。实现在 `utils/agent_prompt_rewrite.py` 和 `EntityDriftVerifier._regenerate_verdict`，
测试 `tests/test_agent_prompt_rewrite.py`、`tests/test_agent_entity_placement.py::RegenerateJudgeTest`。

## 排队的实验

两条链都在等卡，脚本在 `videos/benchmark_20260913/logs/`，状态看各自的 `chain.log`。

**absent_regen**（`videos/benchmark_20260913/absent_regen/`）：16 个 baseline 或 C-new 里
有缺席镜头的场景（`absent_scenes.json`），复用归档的 A baseline，裁判配置和 C-new 一样
（color、indexed、accept_unknown、attribute_masks、sparse）加 `--absent_action regenerate`。
需要一张 ≥40 GB 的卡跑 8B 裁判服务，两张 ≥60 GB 的卡做渲染道。跑完用修好的评测打
judged 和 features 两组指标，按段对 A（fixed rep1）和 C-new 各比一次
（`eval_absent_regen/touched_vs_*.json`）。要回答的问题按顺序：改写后的 prompt 能不能
让人出现；出现的人裁判认不认；前瞻会不会因为后面镜头变了把它撤掉；EG 在这些段上
动不动。

**calibration_30b**（`videos/benchmark_20260913/calibration_30b/`）：用论文的
Qwen3-VL-30B-A3B 做裁判，在 09-05 冻结标签上跑 evidence 和 color 两个比较器、indexed
角色协议、development 和 holdout 两个 split。要回答的是：更大的模型自己能不能达到
像素否决的召回（12/12，误报 ≤2/86）。能，像素规则就删掉，裁判回到纯模型。需要一张
≥70 GB 的卡，模型 58 GB 已经下好。sglang 0.5.3 有 qwen3_vl_moe，服务用
`scripts/launch_qwen3vl_server.py` 起（不限 0.45 的显存比例）。

## 会怎么读结果

整镜头重采样让人出现了：缺席这一大块可修，EG 会真的动，下一步是 30B 阅卷重跑 33。
没出现：文本通道在整镜头尺度上也带不回人，这是 KV 写入必须做的证据，Codex 09-08
"暂不投入 KV 写入"的判断要改。出现了但换了别人：验收侧还是那个漏洞（ns190 镜头 2），
DINO 向后一致性门槛要先补。
