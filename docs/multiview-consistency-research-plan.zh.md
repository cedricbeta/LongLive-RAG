# 研究计划 — Long-Video Multi-Shot Consistency with KV-RAG

> 草稿 v2 · 2026-06-04 · 负责人: cdwang
> 本计划替代旧的“跨视频多视角一致性”目标。当前数据形态是一个 scene 文件夹内的 `0.json`,
> `1.json`, ... 顺序拼成一个长视频,所以研究对象应是**同一长视频内跨 shot 的场景/主体一致性**。

---

## 1. 研究目标

**KV-RAG 能否让一个长视频里的多个 shot 保持同一个事件、同一个主体和同一组关键物体,而不是在
shot 边界重新采样成另一个故事?**

以 `frying_egg_same_event` 为例,目标不是“同一主题的两个独立视频”,而是:

- Shot 0: 微距看到同一个深色平底锅里的完整 sunny-side-up egg。
- Shot 1: 中景看到一个男人继续煎**同一个锅里的同一个完整荷包蛋**。
- Shot 2-5: 继续从侧面、俯拍、中景和最终特写描述同一个锅里的同一个蛋,总计 6 个连续 shot。

旧的 `frying_egg_closeup/1.json` 写成了人在 wok 里炒碎蛋,因此模型生成“人在炒蛋”是 prompt 本身
导致的,不是单纯 KV-RAG 失败。

## 2. 数据协议

推荐目录:

```text
example/long_multishot_prompts/<scene>/
  global.json
  0.json
  1.json
  ...
  shot_durations.txt
```

- `global.json`: 整个长视频必须保持不变的 scene/subject/object 锚点。
- `<i>.json`: 第 `i` 个 shot 的局部镜头描述,只写相机角度、人物动作和局部变化。
- `shot_durations.txt`: 每个 shot 占多少 latent chunk。

`MultiTextConcatDataset` 会把 `global.json` 的 caption prepend 到每个 shot caption,并在 shot
边界保留 scene-cut prefix,这样 prompt 约束和 KV-RAG 的边界检测都能同时工作。

## 3. KV-RAG 干预

Baseline:

- 使用同一份修正后的 long-video prompt。
- `kv_rag.enabled=false`。

Modified:

- shot 0 的 clean recache 存入 persistent scene memory。
- 后续 shot 边界 force-inject persistent anchors。
- 检索使用更适合“同一个主体/物体”的 `retrieval_key_mode=subject_identity`。
- value 保持 `raw`,避免把细节过度压缩。

当前推荐设置:

```yaml
scene_memory_enabled: true
boundary_inject_anchors: 2
scene_score_bonus: 0.15
retrieval_key_mode: subject_identity
retrieval_value_mode: raw
```

## 4. 评测

主指标是同一视频内跨 shot 的一致性,而不是跨视频多视角:

- `cross_shot_scene_consistency`: shot 间颜色/布局/外观签名一致性。
- prompt adherence: 每个 shot 是否仍满足自己的 shot caption。
- motion/diversity guard: 防止靠冻结、复制或过度锚定来假装一致。

旧脚本里的 `cross_perspective` 名称保留为兼容 alias;新实验应使用:

```bash
python scripts/run_kv_rag_ablation.py --mode long_multishot ...
```

## 5. 当前最小验证 case

新增:

```text
example/long_multishot_prompts/frying_egg_same_event/
```

这个 case 是 6 个 shot 的完整 5B 长视频设置:`num_output_frames=128`,`num_frame_per_block=8`,
总计 16 个 latent block,`shot_durations.txt = 3 3 3 3 2 2`。它用于生成 baseline 与 KV-RAG
新版视频,验证后续 shot 是否能保持 shot 0 建立的完整荷包蛋,而不是变成炒蛋或另一个烹饪场景。
