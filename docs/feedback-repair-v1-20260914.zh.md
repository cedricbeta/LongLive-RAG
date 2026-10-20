# 失败反馈驱动的视频修复 v1

> 本文描述 9 月 14 日冻结实验。当前工作区已固定力度为 1.0，并新增范围预览、区域选项和重复动作控制，见 [9 月 16 日修改说明](feedback-repair-v2-20260916.zh.md)。

## 这版改了什么

新增 `--repair_policy feedback`，使用现有 Qwen3-VL-8B 分别进行画面检查和修复规划。规划者读取原图、上次候选、可信人物参照，以及已实际检查的后续失败画面；每轮从同一父版本的合法区域中选择重画范围、强度和 caption。默认 `legacy` 继续使用旧动作选择方式。

模型只能选择程序提供的 mask，不能编造路径或增加工具。缺席人物的整镜头动作固定为 sigma 1.0；属性区域可选 0.63、0.83、0.94、1.0。caption 必须保留所有 required 角色名字，并满足长度限制。文字校验不等同于动作、身份等视觉要求已经满足，结果仍经过原验收流程。

当前镜头通过、后续不通过时，回退状态但暂不交付，在剩余预算中继续规划。所有候选失败时只交付父版本，明确记为未解决。比较程序或服务错误不会被当成画面问题再花一次渲染。第一版遇到这类错误会保存证据并停止该镜头，没有自动修复评测服务。

这是明确的离线修复模式：日志区分 `target_shot` 与 `observed_through_shot`，所有反馈和图片经过相应时间范围的输入审计。不会把未来图片藏进当前镜头的普通上下文。

同时修正约束适配器：把固定比较目标与实际观察到的 `comparison_basis` 分开，避免人物重新出现后被报为“要求定义改变”。此前已成立的出场证据丢失仍会拒绝。新模式下前瞻不会调用 caption 改写器，也不会修改改写历史。

主要源码：

- [规划器与离线循环](../utils/agent_repair_planner.py)
- [已有验收和回退接口](../utils/agent_runtime.py)
- [实体证据和合法区域](../utils/agent_entity_judge.py)
- [约束适配器](../utils/agent_constraints.py)
- [运行入口](../scripts/run_agent_edit_demo.py)

## 固定的六个案例

案例在新版结果产生前固定，并抽取了 baseline 画面核查。它们是开发展示案例，不是独立留出测试集；失败案例也必须保留。

| 案例 | 人类镜头编号 | 看什么 |
| --- | --- | --- |
| `ns310_at_the_flower_shop` | 第 1 镜头 | Clara 应穿 tan coat，baseline 为深酒红衣服；检查改色与身份、后续保持 |
| `ns090_iris_stands_at_the` | 第 1、2 镜头 | Iris 应为红卷发、苔绿外套；检查头发与衣服修复、副作用 |
| `ns020_at_the_studio_equipment` | 第 2 镜头 | Luna 应加入 Clara 和 Miles，baseline 仍主要是两人；检查补人是否挤掉原有人物 |
| `ns050_at_the_market_produce` | 第 2、3 镜头 | Dario 加入、Luca 递苹果、Milan 在场；旧修复曾因后续回退，检查反馈能否改变下一轮 |
| `ns190_in_the_hotel_lobby` | 第 3 镜头重点 | Iris 和 Kai 看笔记本；旧版曾接受有身份副作用的修复，需人工检查脸与动作 |
| `ns130_at_the_studio_equipment` | 全片 | 剧本人物描述较宽泛、旧裁判无需编辑；检查新流程能否保持不动。不是宣称全片没有其他瑕疵 |

执行时跑完整场景以建立人物参照和检查连续性；表中的镜头只是观看重点。

配置：[固定案例清单](../configs/feedback_repair_cases_20260914.json)。

## 三组比较怎样保证可复查

- **baseline**：复用已审计的原始 A 视频，验证 SHA-256；不重新随机生成一份。
- **original**：修改前工作区的源码快照，包含当时已有的未提交改动；使用同一 baseline 和当前实验参数重跑。
- **feedback**：本次源码快照，包含约束修正和新规划循环。

两组修复均使用 color 比较器、indexed 角色分配、sparse 抽帧、属性 mask、缺席时 regenerate、每镜头最多 3 次编辑、最多检查 2 个受影响未来出场。所有输入、源码和 baseline 哈希在场景间复核。

对比的是完整版本变化，不能单凭结果把收益全部归因于规划模型。候选次数上限相同，但新版多了规划调用，**不声称总计算成本完全相等**。报告保留每例墙钟耗时、规划调用和规划耗时。模型权重目录与 Python 环境共享，没有复制进快照。

## 产物和运行方式

实验目录：`videos/feedback_repair_20260914/`。

- `manifest.json`：冻结案例、预算、源码与输入哈希。
- `source_original/`、`source_feedback/`：运行源码快照。
- `baseline_case_preview.jpg`：六个案例的原始画面预览。
- `runs/{original,feedback}/<scene>/`：每个版本的私有输出、缓存、状态和完整日志。
- `comparisons/<scene>.mp4`：结果齐备后的三列视频，顺序 baseline / original / feedback。
- `index.html`、`comparison.json`：全部案例的状态和对比入口。
- `validation.log`：CPU 测试结果。
- `live_planner_smoke.json`：若真实模型调用完成，记录从归档失败画面得出的合法规划；它本身不证明视频改善。

准备、运行、汇总脚本：[run_feedback_repair_comparison.py](../scripts/run_feedback_repair_comparison.py)。本次已执行 prepare 并启动运行。已有状态会保留，重复 run 不会覆盖或偷偷重试失败样本。

```bash
# 只刷新已有结果和对比页面，不调用模型。
venv/bin/python scripts/run_feedback_repair_comparison.py report \
  --output videos/feedback_repair_20260914
```

新实验应放在新目录，先保存其修改前源码，再 prepare。不要改本次冻结目录后续跑，否则哈希审计会停止实验。

## 验证和解读

本次 CPU 验证通过 **104 项测试与 9 项 subtests**。另用花店案例的真实失败画面完成了一次本地 8B 规划调用：返回了合法的衣服区域重画动作、sigma 1.0 和保留 Clara/Miles 的 caption。该检查证明模型接口可用，不证明动作能改善画面；六例完整视频对比已启动，结果以实验目录状态为准。

测试覆盖：后续失败回到第二次规划；只有最终候选提交；预算用完只交付父版本；失败身份参照回滚；正确对照零规划调用；评测错误不触发再渲染；未来证据在正确的观察范围内审计；不允许更换已提供 mask 或使用非法强度；前瞻无改写历史副作用；两组输入与预算一致；冻结源码变更会停止运行。

验收仍主要覆盖人物出场、身份、脚本外观与有条件的人数检查，动作、深度关系等尚未成为完整硬约束。最终要看三列视频，核对“目标改善且没有新问题”。裁判接受数只作为运行诊断，不能单独当作画质提升结论。
