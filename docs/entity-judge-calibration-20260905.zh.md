# 实体裁判校准：独立标签与冻结门槛（2026-09-05）

## 当前决策

**角色分配留出检查未通过；不启动新的 33 场景 arm C。** 单索引协议修复了“assigned + -1”的自相矛盾输出，开发集 51/51，但留出集为 54/57（94.7368%），低于预先冻结的 95%。不四舍五入放行，也不删掉反例或调整阈值。

新分配器和外观候选保留为独立实验模块，尚未接入 live demo。生产部分本轮仅增加缓存请求锁、原子写入，以及不改变规则的票决聚合函数提取。没有改视频生成或添人执行器；没有新的渲染。

## 标签与实验纪律

[冻结标签与门槛](entity_judge_calibration_20260905/labels.json)在候选提示修改和新 VLM 判决之前写入，绑定 packet SHA-256：`da0fcf31b5e6b7846a9fbbd3135fec5a44d46708d0d5c809ead0eef5099cea05`。

- 开发集：ns190/ns290 的 7 个已知案例。
- 留出集：预先指定的 ns000、ns040、ns090、ns130、ns210、ns310，共 11 个 baseline 镜头。原计划每场景镜头 0/1；ns090 镜头 1 没有归档 baseline 帧，明确记录排除，没有拿修复视频替代。
- 每案例检查 chunk 0/3/7 的原始枚举候选框。校准单位是“角色×采样帧”，不是完整镜头；采样时缺席不能冒充全镜头缺席证明。
- 角色标签来自直接查看带候选编号的原帧。外观 crop 使用复核后的 ID 裁剪，不使用待测分配器选出的 ID，因此两阶段的错误可分开测量。
- 开发集有历史参考时冻结当时的参考；留出外观采用 script-only，避免把旧裁判错误注册的参考当作真值。留出检查不覆盖所有参考身份漂移形式。
- 标签是独立于被测 Qwen 的 Codex 视觉审阅，**不是人工 ground truth**。开发样本已经见过旧结果，不是盲法试验；留出标签在本轮新判决之前冻结且不用于继续调提示。
- 无法可靠复核的 Miles（ns210 镜头 1）有 3 个角色标签留空；Clara 的一个运动模糊 crop 外观留空。歧义镜头中的 Clara 不计外观成功或失败。其余已有明确标签时，模型弃权算未判对，保留分母。
- 失败类别冻结为 appearance_drift、sampled_true_absence、script_ambiguous、assignment_unresolved、normal_control。它们只覆盖本次小样本，尚未完成全体 33 场景的 baseline 缺陷分类。

关键独立复核结果：

1. ns190 的白发 Iris 三个 crop 都违反脚本的红发要求，不是只在中间一帧走样。
2. ns290 镜头 1 的 baseline Clara 可见部分为栗红/棕红发、浅棕外套。衣摆在画面外，不能据此认定“不够长”；旧的缺席标签错误，但没有足够证据将它直接改标为外观失败。
3. ns290 镜头 2 两个案例的采样帧只有已建立的 Nora/Ethan，没有第三个 Clara，适合作为分配后的缺席检查。

## 角色分配结果

| 版本 | 开发集 | 留出集 | 决策 |
| --- | --- | --- | --- |
| 上轮双字段 status/index | 45/51 | 本轮未重复测 | 开发门槛未通过 |
| 单索引协议候选 | 51/51 | 54/57 | 留出门槛未通过 |

旧 schema 允许 `status=assigned, subject_index=-1`，实际回复一边说 Clara 不可见，一边给出该组合。校验器正确地拒绝矛盾输出，却把可解释的缺席变成了 UNKNOWN。

候选让 VLM 只返回一个索引：非负数选候选框、`-1` 缺席、`-2` 无法分配；状态由代码生成，冲突分配继续拒判。另实现了保守的同镜头空间连续性与候选耗尽规则，但本次开发和留出输出均未触发这些补充规则；改善来自索引协议本身。

留出的三个实错保留如下：

| 样本 | 正确候选 | 模型候选 | 问题 |
| --- | --- | --- | --- |
| ns040 镜头 0，chunk 7，Ethan | 2 | 1 | 把经过的人认作已建立的交谈主体 |
| ns210 镜头 1，chunk 0，Clara | 1 | 5 | 文字描述前排 Clara，却选择背景人物编号 |
| ns210 镜头 1，chunk 7，Clara | 1 | 3 | 再次选中背景人物编号 |

前排角色描述与返回编号不一致，说明“框是紧的”仍不等于“角色分配正确”。[ns040 对照帧](../videos/entity_judge_calibration_20260905/review/sheets/ns040_s00_a0.png)、[ns210 对照帧](../videos/entity_judge_calibration_20260905/review/sheets/ns210_s01_a0.png)。

本轮不会根据这些留出错误继续修改提示后重报同一批样本。若下一轮改成“全景 + 每个候选的独立编号 crop 卡片”，应将这些错误转入开发集，并选择新的场景级留出样本。

## 外观候选

旧 comparer 的开发集外观为 39/41，其中白发 Iris 只检出 1/3，正常对照 38/38。

第一版先盲描述 crop，再拿描述和原图对照目标：白发检出提高到 3/3，但总体降到 36/41，正常对照降到 33/38。它把“棕色/栗红色”“米色/浅棕色”、看不见的衣摆，甚至脚本没有规定的内搭，当成了冲突。**该候选明确失败，不接入生产。**

第二版将输出改为可见兼容证据、不可见/未规定项、明确矛盾三类，并由代码汇总。开发集最终为 **40/41**，正常对照 **38/38**，白发 Iris 检出 **2/3**；它恢复了 Clara 对照的通过，但仍漏一个白发 crop。

| 外观版本（开发集） | 总体正确 | 白发错误检出 | 正常对照接受 |
| --- | --- | --- | --- |
| 原 comparer | 39/41 | 1/3 | 38/38 |
| 盲描述 + 整体判断 | 36/41 | 3/3 | 33/38 |
| 盲描述 + 可见证据分类 | 40/41 | 2/3 | 38/38 |

单索引角色结果与第二版外观结果的[开发集阶段合并](entity_judge_calibration_20260905/development_combined.json)达到原定开发门槛，但这不是端到端视频实验。角色留出门槛已经失败，外观留出评估未继续开展；不能据开发集 40/41 宣布整条裁判已校准。完整来源、哈希和门槛决策见[校准摘要](entity_judge_calibration_20260905/summary.json)。

三个 crop 仍共享同一个模型及其偏差，票决一致率不是校准后的正确概率。

## 工具、缓存和复现

- [校准工具](../scripts/calibrate_entity_judge.py)：prepare 不调用 VLM，run 分阶段测试，merge 只允许相同 packet/标签/采样配置的完整结果合并。保存实现源码快照和哈希，禁止覆盖同名结果。
- [单索引分配候选](../utils/agent_role_assignment.py)、[盲描述候选](../utils/agent_appearance_judge.py)、[证据候选](../utils/agent_appearance_evidence.py)：均未接入 live demo。
- [缓存实现](../utils/agent_vlm.py)：按请求指纹加跨线程/进程文件锁，同一请求只产生一个响应；原子发布 transcript，异常不破坏旧的完整 JSON。温度 0、seed 0、max_tokens 1024 不变；缓存锁不改变提示或原缓存指纹。
- [测试](../tests/test_entity_judge_calibration.py)：包括冻结数据校验、UNKNOWN 分母、门槛失败关闭、编号互斥、并发缓存及异常写入保护。

本地实验结果（`videos/` 下产物不保证随 Git 分发）：

- [旧版本与外观第一版](../videos/entity_judge_calibration_20260905/development_v1/)
- [分配开发集](../videos/entity_judge_calibration_20260905/roles_v2_development/baseline_development.json)
- [分配留出集](../videos/entity_judge_calibration_20260905/roles_v2_holdout/baseline_holdout.json)
- [外观第二版开发集](../videos/entity_judge_calibration_20260905/evidence_v2_development/evidence_development.json)
- [输入 packet](../videos/entity_judge_calibration_20260905/review/packet.json)

初次 packet 构建的未标注中间产物保留在输出根目录；正式冻结和计分只使用 `review/packet.json`，不混用中间产物。

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/calibrate_entity_judge.py run \
  --output videos/entity_calibration_recheck_roles \
  --cache videos/entity_judge_calibration_20260905/cache_v2_roles \
  --split holdout --role_variant indexed --variants baseline --assignment_only --gpu 1
```

`assignment_only` 输出中的外观计数不是外观实验结果；阶段标记为 assignment_only，整体验收 gate 为 null。必须与完成的 comparer 实验核对版本后合并才能判断整条裁判是否过关。

## 验证与下一轮

相关裁判、约束、放置、运行时、分支及新增校准测试共 **111 passed，9 subtests passed**；静态错误检查与 `git diff --check` 通过。不是全仓库测试声明。本轮启动的所有 VLM 服务已退出，GPU 无残留任务。

下一步建议先处理拥挤画面中的“描述正确、编号错误”：提供全景以及每个候选的独立编号 crop 卡片，避免编号与重叠框混淆；本轮三个留出错误转为下一轮开发反例，另选新留出场景。外观方面仍需补充独立留出和参考身份负例，不能只看平均正确率或三票注册成功率。完成这些检查后，才能冻结全体 33 场景 baseline 缺陷分类、重评 A，并运行 `--place_absent never` 的 C。
