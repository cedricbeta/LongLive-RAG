# VLM-in-the-loop 三 Arm 升级:每个做法做了什么、带来什么收益

日期:2026-07-02 · 分支 `feat/multiview-consistency` · 关键 commit:`13ef860`(benchmark)、`be98235`(多 seed 审计)、`ee3ab3f`(arm 升级)、`f25d7b1`(采样加密)

本文面向组会讨论,逐项说明当前方法栈里每个组件的**动机(旧问题)→ 做法 → 收益(含已验证证据)**。

---

## 0. 背景:实验设计

一条 480 帧 / 8 shot / ~20s 的长多镜头视频(Wan2.2-TI2V-5B,`local_attn_size=32` 帧,即注意力只看最近 32 帧),四个 arm 对照:

| arm | prompt | KV cache |
|---|---|---|
| `baseline` | 原始授权 prompt | 默认(只看最近帧) |
| `prompt_only` | VLM 改写后的 prompt | 默认 |
| `kv_only` | 原始 prompt | VLM 选帧注入 |
| `both` | VLM 改写 | VLM 选帧注入 |

优化器 = 本地 Qwen3-VL-8B(SGLang);裁判 = GPT-5.5(codex OAuth,独立闭环);另有数值 guard(DINO/CLIP 漂移指标 + RAFT 运动 + 多样性 + 不变量探针 + 负控制)双重把关。

---

## 1. 冻结难例 Benchmark(对应会议"固定验证 case")

**旧问题**:每轮实验都对全部 13 个场景做大规模评估,又慢又贵;且 Round 13/14 曾在 baseline 一致性≈0.92 的"天花板场景"上做实验,任何方法都测不出增益(不可证伪)。单 seed 结论被 0.017-0.043 的噪声底淹没。

**做法**(`example/benchmark_hard_multishot/` + `docs/benchmark_hard_multishot/BENCHMARK.md`):

- 按 baseline 漂移从高到低冻结 7 个难场景(sandy_beach_driftwood 0.61 → warm_indoor_dining 0.45),排除接近天花板的易场景;
- 新写 2 个**人物精细特征场景**(制表匠:银灰胡须/浓眉/痣/铜框眼镜;面馆厨师:光头/龙纹身/耳环/红头巾)——因为 Round 16/17 裁判判 KV 赢的证据集中在这类细节,而整帧 embedding 指标疑似看不见;
- 1 个带 `negative_control: true` 标记的自相矛盾场景做反作弊探针;
- 所有场景补齐 invariant/contrast caption(修掉"不变量 guard 永远 NaN"的旧洞);
- seeds 0-2 各渲一遍 baseline,冻结**逐场景 2σ 跨 seed 噪声底**作为胜出阈值(african_savanna 最灵敏 0.018,aquamarine 最噪 0.135)。

**收益**:实验从"每次全量扫描"变成"对着固定靶子打";高漂移场景保证有提升空间;新场景直接针对"细节假说";逐场景噪声底让"赢"第一次有统计含义。多 seed 审计确认 9/9 主场景全部准入,两个新场景正好落在难例区(0.570 / 0.449)。

---

## 2. 按 boundary 检索式 KV 选帧(核心升级)

**旧问题**(Round 17 取证,`docs/round17_20260618_fast/`):所谓"VLM 选帧"实际是一次调用输出全部计划,结果 20 个 boundary 计划**全部**是 shot0 采样帧的前缀(11 个一模一样是 `(0,48,95,96)`),没有任何中段帧;漂移最重的后段 boundary 反而被整段忽略(brown_bear 5-7、skateboarder 2-5 无计划);理由全是"provide a robust anchor"式模板句。根因:候选池只有每 shot 3 帧、一次性输出全计划导致模型偷懒单调堆积。

**做法**(`select_kv_anchors_per_boundary`,commit `ee3ab3f` + `f25d7b1`):

- **每个 shot boundary 单独一次 VLM 调用**:给出该 boundary 的 query 帧(进入 shot 的开头帧)+ 候选缩略图(带 `C0/C1/...` 标签),要求按**画面内容**选出最能钉住主体身份/场景布局的帧;
- 候选池从每 shot 3 帧加密到 **8 帧**,每 boundary 上限 24 张(均匀时间采样,首尾必保留),防止后段 boundary 调用膨胀;
- 明确指令"早的帧不自动更好",允许**显式 skip**(必须给理由)——不再有静默遗漏;
- 候选仍强制来自活跃注意力窗口之外(保证注入的是模型"看不见"的信息)。

**收益**(Round 18 中期实测,60 个 boundary 计划):

| 指标 | Round 17 | Round 18 |
|---|---|---|
| 独立帧集数 | ~5 种(全 shot0 前缀) | **34 种** |
| 用中段帧(>96)的计划 | 0/20 | **52/60** |
| 后段 boundary 覆盖 | 静默丢弃 | 全覆盖,仅 5 次显式 skip |
| 理由质量 | 模板句 | 点名画面内容("Frame 41 provides a clear, unoccluded view of the elephant... tusks against the acacia trees") |

选帧第一次表现出"检索"行为:进入 shot 7 的 boundary 选的是紧邻切点的 343-397 帧,而非视频开头。这是"用 agent 替代 heuristic"(KB-RAG 思路)真正落地的版本。

---

## 3. 按场景的 KV 注入计划(修被掩盖的实现缺陷)

**旧问题**:Round 17 里各场景的选帧决策在注入前被 `universal_anchor_plan` **跨场景合并**成一份按 boundary 的并集(`utils/kv_rag.py` 只按 shot 序号查表)——场景 A 选的帧会影响场景 B 注入什么。"为该场景选相关帧"这个概念从未真正到达生成器,每个场景的报告里"Shared manual plan differs from per-scene KV plan: True" 就是这个缺陷的痕迹。

**做法**(manual plan v2,`ee3ab3f`):

- 计划文件加 `scenes: {场景名: {boundary: [帧]}}` 层级,旧的全局 `boundaries` 保留为兜底;
- `inference.py` 在每个样本生成前把场景名传给 KV bank(`set_scene_name`);
- bank 按样本名解析场景(精确匹配 → 分隔符定界的最长匹配,杜绝 `cat` 抢走 `cathedral` 样本),命不中才退回全局计划;
- 诊断事件里记录 `plan_source: per_scene/global`,可审计。

**收益**:每个场景注入的就是为它选的帧——kv_only/both arm 第一次真正执行了优化器的决策。这是**不改假设、纯粹修实现保真度**的升级,任何后续结论都建立在"干预真的发生了"之上。

---

## 4. Humanize 式 Prompt 审查门(auto-review)

**旧问题**:优化器看的是 baseline 渲染,而 baseline 本身会漂移/违背 prompt。Round 17 里它把看到的漂移样子直接写成"不变量":授权 prompt 说"白色 V 领 T 恤",它加"深蓝 T 恤绿 logo";说"红色连帽衫",它加"黑 T 恤白字"——把缺陷固化成了指令,prompt_only 和 both 双双被污染,还制造了"文本覆盖"式的假一致性。

**做法**(`review_prompt_additions` + `apply_review_verdicts`,`ee3ab3f`):

- 参考 humanize 的 pair-code + 独立 review 模式:第二次独立 VLM 调用扮演**审查者**(不是作者),对照授权 global/invariant caption 逐条审查优化器提出的添加项;
- 三种裁决:`keep`(与授权一致或使其更锐利)/ `rewrite`(与授权矛盾 → 改写为重申授权不变量)/ `drop`(无关或压缩镜头多样性);
- 裁决全部落账(applied/invalid 分开记录),可追溯。
- 同时诊断 prompt 里也加了硬规则:"渲染与授权矛盾时,矛盾是要修的缺陷,重申授权,**永不固化渲染出来的样子**"。

**收益**:Round 18 实测抓到一次真实矛盾——sandy_beach_driftwood 的 baseline 漂移出一个不存在的人物,优化器想把"深色头发、橄榄绿帽子、黑色比基尼的女人"写进纯物景的全局不变量,审查者以"授权不含任何人物"为由改写。**Round 17 的污染模式在渲染前被拦截**。34 条添加项:33 keep、1 rewrite,说明门槛不误伤正常添加。

---

## 5. JSON-Schema 约束解码 + 解析重试(对应会议"格式输出问题")

**旧问题**:VLM 输出结构化 feedback 格式不稳定,靠"找大括号"解析;失败无重试;sanitize 静默丢弃格式错误的片段(Round 17 有"snapped invalid frame 96 → 48"这类痕迹,模型选了候选表外的帧)。

**做法**(`qwen_chat_json`,`ee3ab3f`):

- 所有优化器调用(诊断/选帧/审查)都带 JSON Schema,经 SGLang 的 `response_format: json_schema` 做**约束解码**——语法层面保证输出合法;
- 旧版 SGLang 不支持时自动降级为"prompt 约束 + 解析重试"(一次带错误信息的追问);
- 每次调用的 `schema_enforced`/`parse_attempts` 落账。

**收益**:从"祈祷模型输出对"变成"结构性保证"。Round 18 全程 `schema_enforced: True`、零解析重试——会议里的格式痛点从根上解决,而不是继续调 prompt 工程。

---

## 6. 内容指纹缓存 + Resume 标记 + 懒启动(工程可靠性)

**旧问题**:多小时的运行随时可能断(共享 GPU 被挤爆、会话中断);断了重跑要么全部重来(浪费几小时),要么无脑复用旧产物(把旧 prompt/旧模式下的决策错当新的——codex 审查抓出的高危项)。

**做法**(`ee3ab3f`,含 codex 审查后的修复):

- 每次 VLM 调用的转录带**内容指纹**(模型 + prompt 文本 + 有序图像哈希 + schema 的 SHA256),缓存命中必须指纹完全一致——prompt 改一个字、图换一张、顺序变一下都会重算;
- `optimizer_config.json` 标记记录所有影响优化器输出的配置,`--resume` 只在完全一致时整体复用,否则打印原因重算;
- SGLang 服务器**懒启动**(全部命中缓存时根本不占 GPU),异常路径保证进程回收(不留孤儿);
- 渲染层本来就有按条(per-stem)断点续渲。

**收益**:断点续跑既**便宜**(已完成的一律跳过)又**正确**(任何配置变化自动失效缓存)。今天 Round 18 已经历三次中断,每次重启零重复渲染、零脏复用。

---

## 7. 多 Pass 反馈环(对应会议"multi-pass refine")

**旧问题**:loop 的多轮骨架虽然存在(iter>1 沿用上一轮 refined prompts),但优化器每一轮看的都是**原始 baseline 渲染**——它永远看不到自己干预后的结果,"审查→干预→再审查"的反馈边缺失,多轮 refine 退化成对同一个输入的重复诊断。

**做法**(`--multi_pass_watch_arm`,默认 `both`,`ee3ab3f`):

- 第 2 轮起,优化器观看**上一轮 both arm 的渲染**,诊断残余的断点,在其基础上增量改 prompt / 重选锚帧;
- 判分与配对仍然对着 baseline(评估口径不变,只有优化器的"眼睛"换了);
- `--multi_pass_watch_arm baseline` 可回退旧行为做对照。

**收益**:多轮 refine 有了真实的梯度信号——第 N 轮修的是第 N-1 轮**没修好**的地方,而不是反复修同一个 baseline。这正是会议里"对不满意的 shot 迭代 refine"的实现路径(Speaker 3 确认有论文先例)。Round 18 先跑 1 轮,验证方向后 `--resume --rounds 2` 即可增量开启。

---

## 8. λ Logit-Bias 杠杆首次启用

**旧问题**:Round 14 的注意力诊断发现注入的锚帧确实被消费,但注意力质量被降权 4-20 倍——"信息给了但模型不重视"。对症的 logit-bias 杠杆(给持久锚列加注意力偏置)当时就推荐了,但 Round 17 实际跑的是 λ=0.0、bias 调用 0 次,**从未被执行过**。

**做法**:Round 18 起 `primary_logit_bias_lambda=1.0` 成为 kv 系 arm 的默认;attention-mass 诊断照常落账作为操纵检查(manipulation check);`--extra_logit_bias_lambdas` 支持后续 λ 扫描。

**收益**:机制假设("降权是瓶颈")第一次得到真实检验;attention mass 是否随 λ 上升、一致性是否随之改善,两条曲线能区分"注入不够"和"注入了也没用"两种失败模式。

---

## 9. 运维层:tmux + 看门狗 + 动态 GPU + 并行预渲染

**旧问题**:共享 8×A100 机器上邻居任务随时蜂拥;Claude 会话自带的后台任务会随会话终止;固定 GPU 编号的任务会被反复挤爆(今天 GPU 1/GPU 3 各阵亡一次)。

**做法**(`docs/round18_bench_20260701/*.sh`):

- 长任务一律 tmux(脱离会话生命周期);
- **看门狗**每 2 分钟检查,进程死亡且未到终态就自动重启(上限 8 次防死循环),重启时**动态选当时最空的 GPU**,叠加 ablation 内置的满载自动切换;
- **预渲染车道**:主 loop 串行渲 kv_only 时,空闲 GPU 倒序预渲 both 场景;"倒序 + loop 一进入 both 阶段立即停手"两条规则保证两条车道永不写同一文件,做完的场景直接从 loop 待办里消失;
- 终态标记(`GRIDS_DONE`/`GRIDS_FAILED`)+ 通知探针。

**收益**:今天实测——11:5x GPU 3 被挤爆,12:00 看门狗自动重启并切到健康 GPU,12:56 预渲染车道贡献 4/10 both 场景后按规则退出,全程无人工干预、零文件冲突。串行 5.5 小时的尾程压缩约 1.5-2 小时。

---

## 10. 效果汇总与下一步

**已验证**(Round 18 中期,优化器阶段完成):选帧从退化堆积变成内容检索(34 种帧集、52/60 中段帧、全 boundary 覆盖);审查门拦截真实矛盾注入 1 例、误伤 0;schema 全程生效、零解析失败;自愈体系经历 3 次中断零损失。

**待出**(Round 18 尾程):GPT-5.5 裁判三 arm 对比、数值 gate(逐场景 2σ 阈值)、λ=1.0 下的 attention mass、每场景 4-up grid。

**注意**:升级后的 arm 与 Round 16/17 的结果**不可比**——旧轮次的 kv/prompt arm 带着上述实现缺陷,它们的 null 不外推到新 arm。

**下一步优先级**:① Round 18 结果 + 人工看 grid;② 若方向为正,`--resume --rounds 2` 开多 pass、补 seeds 1-2 做全功率配对检验;③ λ 扫描(0 / 1 / 2);④ 指标侧升级(subject masking + ArcFace)裁决"裁判 vs 数值指标"分歧;⑤ token/region 级 KV 选择(与 subject mask 共用分割)。
