# 人工指定目标的小样本修复实验

## 状态

2026-09-05：已准备 5 个候选、每例 8 个 chunk 的明确选框、SAM2 实际遮罩和原视频预览。**尚未获得逐例人工确认，尚未重画，也没有人评结果。**

查看 `videos/oracle_entity_repair_20260905/index.html`。黄色是拟选人物框；视频中青色是 SAM2 遮罩预览。候选由助手查看原画面后提出，不冒充人工真值。人工可在页面下载确认 JSON，也可以在对话中明确确认或指出哪例有问题。

暂停继续改自动裁判；本次只回答：给定正确人物和修复区域，现有 arm C 的文字条件 masked-renoise 能否修正明确外观错误，而且不破坏动作、其他人物及后续镜头？

## 固定候选

| 场景 | 人物 | 待人工确认的缺陷 |
| --- | --- | --- |
| ns090 | Iris | 深色头发、深色上衣；设定红色卷发、苔绿色外套 |
| ns180 | Noah | 蓝色上衣；设定深绿色工作衬衫；不评价发长 |
| ns190 | Iris | 白／银色头发；设定红色卷发 |
| ns220 | Clara | 左侧坐着的女人穿灰色外套；设定棕黄色长外套 |
| ns310 | Clara | 左侧女人穿酒红／紫色外套；设定棕黄色长外套 |

全部选择第一镜头，人物明确在场；不是随机抽样。这是**设定外观违反的能力探针**，不是跨镜头身份漂移测试，也不是总体成功率或自动系统有效性的证据。ns190 是已见过修复结果的开发案例，其余也从既有素材筛选，不能称为独立盲测集。

## 固定执行

- 输入来自既有 baseline；ns190 使用 09-03 smoke，其余使用 08-27 三臂实验。复制各自的 caption 和配置，不混用不同来源。
- 直接使用已归档 enumerate 的候选框；每例 8 个选框 ID 显式列在 `packet.json`。不调用 VLM 分配、comparer、自动 guard 或 CLIPSeg，不处理 unassigned，不主动添人。
- 只给目标人物加 SAM2 prompt，沿用现有 `_soft` 转为 `[8,22,40]` 的逐 chunk 遮罩。不改生成实现。
- 每例先做一次无编辑重放，要求与人工确认的旧 baseline 的完整解码像素相同。若不同，停止并检查重放混杂因素，不拿这个差异算修复收益。
- 通过后调用现有 `EditRenderBackend.edit` 一次：`entry_sigma=1.0`，实际噪声种子 `4243`，caption 加原始人物 spec，KV-RAG 关闭。完整重放后续镜头。没有 best-of、失败重抽或自动接受结果。
- 人工确认绑定整个 packet 的 SHA256；运行前校验视频、输入文件、遮罩及相关源码指纹。每例独占 `started.json`，不覆盖既有运行。若出错，保留 `failure.json` 和日志，需先排查，不能悄悄多抽一次。

准备时 SAM2 报告本环境缺少 `_C` 扩展，因此跳过填洞后处理；传播完成、遮罩非空。此环境事实没有被当作通过质量检查的理由，实际区域仍需人工查看。原始 chunk PNG 与源 MP4 对应帧已逐一核对相同。

检查结果：packet 内所有文件／源码指纹一致；40 个逐 chunk 遮罩的形状、有限值、范围和非空检查通过。相关测试共 72 项通过（pilot、edit branching、entity judge、runtime、placement 五个测试文件）；不是全仓库测试。当前 packet SHA256 为 `8db8a41fb1f1fee650181ed189bac8bf0b7bd603e944c327070886e7736df8a4`。

## 如何运行

入口：`scripts/run_oracle_entity_repair.py`。已经准备好的包不要再次执行 `prepare` 覆盖。

```bash
python scripts/run_oracle_entity_repair.py run \
  --output videos/oracle_entity_repair_20260905 \
  --review /absolute/path/to/human_review.json \
  --case ns090 --gpu 0
```

其他病例同样调用，`--case` 为 ns180 / ns190 / ns220 / ns310。可分配不同 GPU 执行不同病例，但每例只调用一次。未获得真实人工确认，不应伪造 `reviewer_kind=human` 来越过检查。

全部五例完成后：

```bash
python scripts/run_oracle_entity_repair.py export \
  --output videos/oracle_entity_repair_20260905
```

导出 `blind_review/`：随机案例顺序、随机 A/B 侧别、每侧目标镜头及全部后续镜头，附独立 A/B 评分模板。答案只在包外 `blind_key.private.json`；不要把它发给评审。缺少任一结果就拒绝导出，不能只展示成功样例。

## 人评和解释

由人分别给 A、B 的外观、动作、其他人物、后续镜头标 `pass / fail / uncertain / not_applicable`。外观稍有改善但仍违反设定不能记为 pass。对于后续问题，另外记录具体镜头和问题，解盲后区分原有缺陷与新增退化，不能把“两边都失败”自动视为无退化。

最好让未参与目标筛选的人评价；看过 baseline 的人可能认出哪边是原视频，只能叫匿名顺序评价，不能宣称完全盲评。助手的视觉检查也不能代替人评。

逐例报告：外观是否真正改正，以及动作／旁人／后续有没有新增损坏。不设置一个任意百分比作为研究成败门槛；有明显可复现收益再考虑自动目标分配，正确目标也修不动则优先研究执行器。
