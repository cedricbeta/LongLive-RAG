# 实体裁判分配修正与离线回放（2026-09-05）

## 结论

本轮只修改裁判、实体标注及审计工具，没有调用视频生成、没有跑新的 33 场景 arm C，也没有改添人的执行器。已有放置、关系、历史轨迹、guard 和回退代码保留。

离线证据支持“问题首先在角色分配，而非 enumerate”：ns290 中 Nora 原先跨两人的松框变成自己的紧框，之前被当作 impostor 的主体恢复为 Nora。白发 Iris 也能被分配到 Iris 角色。但还不能宣布裁判已可靠：分配器仍会因外观差异弃权，comparer 仍会漏判明显的外观问题。现在适合做小规模裁判校准，不适合把放行率增加直接解释成修复收益。

## 代码与判定规则

- `enumerate → joint role assignment → SAM2 → comparer`：每帧只枚举一次，角色联合选择候选 ID；不允许另画坐标。一个候选最多归属一个角色；冲突 ID、无效 ID、近乎重合的重复候选都降为 uncertain。
- 在原帧副本上画出实际候选编号，避免 VLM 把枚举顺序误当成从左到右的顺序。旧原帧不变。初版只有 JSON 坐标时确实出现过三个角色全部错位，最终回放使用了带编号版本。
- 分配参考剧本动作、同镜头前序位置、其他角色及参考图；允许角色外观走样。分配提示不直接列外观规格，并移除 caption 中与规格完全相同的具名外观同位语。原始生成和修复 caption 不变。这只是减少外观锚定，不保证模型完全不受外观影响。
- SAM2 仅接收选中的紧框。候选是否已归属由分配 ID 决定，不再根据 SAM2 覆盖比例把已分配的人重新认作陌生人。
- `assigned / absent / uncertain` 分开记录。不确定分配对应 UNKNOWN，而不是证实缺席；不据此把疑似“陌生人”重画。
- comparer 分别对三个不同时间位置的紧 crop 调用三次。`confidence` 改为票决一致率，不再使用 VLM 自报概率。首次参考注册必须三票全为匹配，降低旧阈值也不能绕过；既有参考的外观判断仍采用多数票。
- 三个 crop 使用同一个模型，错误可能相关；一致率不是校准后的正确率。本轮没有解决这一研究问题。
- VLM 固定 `temperature=0, seed=0`；entity 模式和回放均使用 `max_tokens=1024`。采样参数加入内容指纹，支持显式共享缓存目录。旧缓存不覆盖；离线枚举由旧记录显式读取。缓存可保证相同请求复用原响应，但不承诺绕过缓存后的 GPU 推理绝对确定性。

主要代码：[entity judge](../utils/agent_entity_judge.py)、[constraint adapter](../utils/agent_constraints.py)、[VLM cache](../utils/agent_vlm.py)、[demo](../scripts/run_agent_edit_demo.py)、[测试](../tests/test_agent_role_assignment.py)。

## 33 场景实体表

逐一阅读了 33 场景的 198 条 caption。修改后的表有 102 个角色条目、29 个 ambiguous 角色×镜头条目。这是模型辅助的文本审阅，不是独立人工 ground truth；每处修改保留了前后值、理由和对应 caption，便于复核。

- 明确冲突共移除 5 个出场要求：ns110 Luna 的镜头 0、4；ns120 Luna 镜头 3；ns230 Evan 镜头 3；ns300 Luna 镜头 3。
- 镜头内离场、附近活动或延续出场不明确的情形保留并标记 ambiguous，不按关键词删除。
- 补齐了原表遗漏的 6 个角色：ns060 Luna、ns070 Mina、ns150 Nora、ns160 Noah、ns240 Clara、ns250 Iris。原提取 schema 最多容纳 4 人，现提高到 8；没有扩展镜头内时间区间 schema。缺乏外观描述的新增角色仅填 `a person`，不从其他场景借外观。
- 另补回 ns060 Miles 镜头 3、ns070 Ava 镜头 1 的明确出场要求。

`presence: {"3": "ambiguous"}` 是每个角色的镜头覆盖字段。排除粒度是角色×镜头：该角色不计成功或失败、不注册参考、不触发修复，但仍尝试定位以保护其区域；同一镜头其他 required 角色照常评估。含 ambiguous 参与者的关系也排除。若全镜头没有 required 角色，交付可以继续，但 `repair_score=null`，不能算评测成功。

表与审阅记录：[entity specs](../example/narrastream_bench/entity_specs_subset33.json)、[spec_review.json](entity_judge_assignment_20260905/spec_review.json)。

## A/C baseline 审计

检查了 `videos/narrastream_threearm_20260827` 中 33 对最终非 partial 记录：

| 检查 | 结果 |
| --- | --- |
| C baseline 的 KV-RAG 开启 | 0 / 33 |
| 渲染配置相同（忽略输出位置与文件名） | 33 / 33 |
| caption 等 authored 输入相同 | 33 / 33 |
| baseline MP4 SHA-256 相同 | 33 / 33 |

全部视频已经逐字节相同，因此无需额外解码来证明画面一致。配置模板含 `kv_rag` 段并不代表启用。这排除了此次归档 A/C baseline 不同的混杂，但不证明旧的修复指标有效。

[完整审计](entity_judge_assignment_20260905/baseline_audit.json)；复现：

```bash
venv/bin/python scripts/audit_narrastream_baselines.py \
  --output docs/entity_judge_assignment_20260905/baseline_audit.json
```

## 离线回放：7 个案例

使用 09-03 的 ns190 与 09-04 的 ns290 已有 `entity_judge/frames`。每个案例冻结原枚举框和当时的注册参考图，只重新调用分配和比较，并重新跟踪已有帧。没有重新渲染，也没有让新注册参考在案例之间传播。所有案例 `--place_absent never`。

下表镜头号、attempt 均从 0 开始。新旧判断变化同时包含分配、比较及歧义标注变化，不是单因素消融，更不是 arm C 的新实验结果。

| 场景 / 镜头 / attempt | 旧判定 | 新判定与证据 |
| --- | --- | --- |
| ns290 / 1 / 0 baseline | 缺席 + impostor | 三人均 8/8 chunk 定位，三人各 3/3 外观通过，放行 |
| ns290 / 1 / 1 repaired | 通过 | 三人均 8/8，继续通过 |
| ns290 / 2 / 0 parent | 缺席、请求添人 | Clara 分配不确定；Nora/Ethan 各 7/8；不凭不确定分配添人或重画 |
| ns290 / 2 / 2 repaired | 缺席 + impostor | Nora/Ethan 均 8/8，Nora 的 impostor 误归属消失；Clara 仍分配不确定 |
| ns290 / 3 / 0 parent | 缺席 + impostor | Clara 为 ambiguous，不计分；Nora/Ethan 均通过，允许继续交付 |
| ns190 / 0 / 0 baseline | 缺席 + impostor | 白发主体被认作 Iris，8/8 定位；外观票为对/错/对，禁止首次参考注册 |
| ns190 / 0 / 1 repaired | 通过 | Iris 8/8，三票全通过，允许注册 |

### 需要保留的反例

1. **ns290 镜头 1 不是已验证的“在场但外观违反”。** 新 comparer 三票都接受 Clara 的 baseline crop。因此旧的缺席错误得到修正，不等于外观修复产生了收益。需独立复核此 crop 的标签，不能强行按预想解释。
2. **ns190 白发 Iris 仍是 comparer 的反例。** 三票中两票认为符合脚本，多数票依然偏宽松；本次仅因为是首次注册，三票全同意的门槛挡住了错误锚点。不能据此推断既有参考的外观评估也已可靠。
3. **分配器尚未完全摆脱外观锚定。** ns290 镜头 2 的 parent 最后一个 chunk 连 Nora/Ethan 也弃权；Clara 在缺人画面中常为 uncertain 而非 absent。UNKNOWN 必须单独报告，不能以分母缩小制造提升。
4. 新结果的 `unassigned_subjects=[]` 不总意味着证实没有陌生人：存在不确定角色分配时，相关 count 约束为 UNKNOWN，这是保守策略。

主要对照图（左旧右新，chunk 0/4/7）：

- [ns290 镜头 1 baseline：松框变紧框](../videos/entity_assignment_replay_20260905/indexed/ns290/shot01_attempt0_boxes.png)
- [ns290 镜头 2 attempt 2：Nora 不再是 impostor](../videos/entity_assignment_replay_20260905/indexed/ns290/shot02_attempt2_boxes.png)
- [ns190 镜头 0 baseline：白发主体仍分配为 Iris](../videos/entity_assignment_replay_20260905/indexed/ns190/shot00_attempt0_boxes.png)

完整本地结果：[ns290](../videos/entity_assignment_replay_20260905/indexed/ns290/results.json)、[ns190](../videos/entity_assignment_replay_20260905/indexed/ns190/results.json)。结果保存输入视频、spec、核心实现的 SHA-256、冻结枚举指纹、采样参数、每帧候选/角色分配、crop 票决和约束状态。`videos/` 是本地实验产物，不保证随 Git 分发。早期未编号提示的试运行保留在非 `indexed/` 目录，不与最终结果混用。

回放工具：[replay_entity_judge.py](../scripts/replay_entity_judge.py)。它缺少归档帧或枚举响应时直接报错，不会偷偷重新枚举或生成视频。例如：

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/replay_entity_judge.py \
  --source_run videos/narrastream_place_smoke_20260904/ns290/ns290_at_the_restaurant_prep \
  --case 1:0:baseline --case 1:1:edit_shot01_attempt1 \
  --case 2:0:edit_shot01_attempt1 --case 2:2:edit_shot02_attempt2 \
  --case 3:0:edit_shot01_attempt1 \
  --output videos/entity_assignment_replay_20260905/indexed/ns290 \
  --gpu 0 --tracker_device cuda:1
```

本次 SAM2 环境提示可选 `_C` 后处理扩展不可用，跳过对应后处理；没有变更依赖。离线任务启动的 VLM 服务已随任务退出，未留下 GPU 进程。

## 验证与下一步

相关裁判、约束、放置、运行时、分支测试：94 passed，另 9 subtests passed。生成环境中的 edit-shot 测试另有 16 passed。不是全仓库测试声明；原有用户工作区改动未回滚。`git diff --check` 通过。

建议下一步先冻结一小批独立复核的判例，包含真实缺席、角色走样、允许离场、分配不确定和正常对照；记录候选 ID、角色标签与明确违反的属性。用这些样本分别检查分配和 comparer，尤其是白发 Iris 与 Clara baseline 这两个反例。冻结裁判版本、spec、缓存和失败类型标注后，再用同一裁判重评 A，并启动关闭主动添人的 33 场景 C；报告每类的改善、退化及 UNKNOWN/ambiguous 数量，不只报总放行率。

贴片实验仍可独立排期，但本轮未启动；现有放置代码无需删除。全体 33 场景的 baseline 缺陷分类也尚未完成，本次 spec 审阅不能替代它。
