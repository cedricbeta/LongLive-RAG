# 2026-09-10：消除评分的全场景等待，补上后续队列

## 实际状态（08:58 PDT 左右）

原 A、B、旧 C 的 33 场景 × 11 指标已完成；当前 C 为 29 个生成完成、3 个生成失败、最后 1 个运行中。完成生成不等于通过实体裁判，也不等于已完成独立评分。

| GPU | 当前任务 | 后续队列 |
|---|---|---|
| 0 | 当前 C 最后一个场景的生成 / 跟踪 | 原生成进程组完全退出后，领取失败场景的未编辑基线重放诊断 |
| 1 | 已完成的 29 个当前 C 视频，7 个特征类指标 | 最后场景若生成完成，自动补入同一评分协议 |
| 2 | 独立评分 VLM 服务，供 4 个条件一致性 / 指令指标调用 | 服务于当前 C 后续评分批次 |
| 3 | 当前 C 的控制器 VLM | 原生成进程组完全退出后，与 GPU 0 分担 3 个重放诊断 |

GPU 0/3 在生成与 VLM 判定之间轮流忙碌；评分也有 CPU 与缓存阶段，因此不宣称四张卡每个时刻都达到 100% 利用率。

## 本次修正

旧评分调度器在 A/B/旧 C 已完成后，要求当前 C 全部成功才启动评分，两个评分分支因此空等。这是调度问题。

已核实旧评分两个分支都在等待、没有运行中的指标任务，然后停止其专用进程组 378429，释放闲置评分服务显存。原生成进程组 324175 未被中断。旧结果和内容寻址缓存全部保留；旧 campaign.json 加入交接说明，避免仍显示为正在计算。

新调度器 `scripts/run_ready_scene_evaluation.py` 已启动（PID 667885）：

- 只领取 `driver_completed` 的场景，不再要求 33 个全部成功。
- 已启动 29 场景的两个评分分支；日志显示两个分支都在推进。
- 保持原 33 场景 sample ID，即使中间有失败也不重新编号。
- 部分结果只与 A/B/旧 C 的相同 sample ID 配对，不能与基线全 33 平均数直接对比。
- `population.json` 保留全部 33 个场景，失败不会被删掉，也不会偷偷替换成 A 视频。
- 复用前核对原评分器源文件哈希；生成和裁判策略均未修改。

状态与输出：

- `videos/benchmark_20260910/ready_evaluation/queue.json`
- `videos/benchmark_20260910/ready_evaluation/population.json`
- `videos/benchmark_20260910/ready_evaluation/comparison.md`
- `videos/benchmark_20260910/ready_evaluation/logs/batch00_features.log`
- `videos/benchmark_20260910/ready_evaluation/logs/batch00_judged.log`

## 已实际排队的 3 个诊断

`scripts/run_prefix_replay_diagnostics.py` 已启动（PID 688821），状态为 `waiting_for_generation_release`。队列已落盘，不只是待办描述。

| 场景 | 失败候选 | 检查未编辑的前缀 |
|---|---|---:|
| ns060_at_the_office_meeting | edit_shot02_attempt1 | chunk 0–15 |
| ns110_in_the_library_reading | edit_shot03_attempt1 | chunk 0–23 |
| ns200_at_the_flower_shop | edit_shot05_attempt2 | chunk 0–39 |

三项均为 branch-0000 的 decoded-frame 前缀校验失败。已有失败候选的 latent 哈希，因此每项只重新生成一次未编辑基线，保存 latent 与逐 chunk 哈希，然后进行两项检查：

1. 新基线与已有候选在编辑之前的 latent 哈希是否完全一致。
2. 新基线与存档 A 的全部 384 个解码帧是否完全一致，并记录逐帧误差。

任务只有在原生成状态终结、原进程组退出、对应 GPU 空闲显存至少 75000 MiB、磁盘剩余至少 50 GiB 时才能开始。GPU 0/3 两个执行槽共享三项队列，不抢占运行任务，不重跑直到成功，不改变阈值。

存档 A 没有 latent 证据，所以新重放不能冒充原始 A 的位级 latent 认证。诊断结果不会自动把原先的 `driver_failed` 改成成功。排队范围是入队时已记录的三个失败，后续新失败仍留在原 manifest，不暗中追加重试。

状态和报告：

- `videos/benchmark_20260910/prefix_replay_diagnostics/queue.json`
- `videos/benchmark_20260910/prefix_replay_diagnostics/<scene>/report.json`

## 验证与范围

就绪调度、缺失指标、原始 ID、GPU 交接条件、latent 精确比较、失败识别及原有审计 / 冻结逻辑相关测试共 **21 passed**；`git diff --check` 通过。

当前仍是 33 个 16 秒视频的迭代诊断，且本地 8B VLM 评分有 fallback，不能作为完整、可靠的论文质量结论。IAMFlow 官方权重权限、正式长视频协议和等预算开环对照尚未解决；它们不在这次已启动的队列里。内部 B 也不冒称为外部论文 LongLive-RAG。
