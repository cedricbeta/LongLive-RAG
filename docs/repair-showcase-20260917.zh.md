# 可观看的修复实例

已从已有真实生成结果制作四个开发实例和一个失败对照，无新增生成或验收调用。

- [中文播放页](../videos/repair_showcase_20260917/index.html)
- [21.33 秒四例速览](../videos/repair_showcase_20260917/overview_4_examples.mp4)：各例完整首镜头，0.5× 慢放。
- [全部材料 ZIP](../videos/repair_showcase_20260917/repair_showcase.zip)

| 实例 | 来源 | 可见变化 | 全片 |
|---|---|---|---|
| Clara 花店 | 历史自动纠错 original | 酒红／紫色外套改为棕褐色 | [16 秒](../videos/repair_showcase_20260917/01_clara_flower_shop.mp4) |
| Iris 花店 | 历史自动纠错 original | 深色头发变偏红；衣服要求仍未解决 | [16 秒](../videos/repair_showcase_20260917/02_iris_flower_shop.mp4) |
| Clara 图书馆 | 固定目标探针，绕过自动验收 | 灰衣变棕黄色；纸张与动作变化 | [16 秒](../videos/repair_showcase_20260917/03_clara_library.mp4) |
| Iris 酒店 | 固定目标探针，绕过自动验收 | 白发变红卷发；姿态／脸部变化，后期漂移 | [16 秒](../videos/repair_showcase_20260917/04_iris_hotel.mp4) |
| Iris 整人区域失败 | 固定目标探针 | 新增花盆遮挡与边缘异常 | [16 秒](../videos/repair_showcase_20260917/05_failure_iris_full_person.mp4) |

全部为历史本地 5B 的已暴露案例。可展示局部属性编辑能力，不能证明最新官方 1.3B progressive、完整自动验收或反馈机制的净收益，也不用于计算成功率。各全片保留全部 384 帧、六镜头和原始 24 fps；左原片、右修复，仅相同等比缩小和添加说明，未裁剪画面或修正颜色。速览中的慢放仅重复原帧，没有补帧。

来源哈希和全案例记录在播放页的 manifest/evidence 链接中。六个 MP4 已完整解码核对帧数、H.264/yuv420p 与哈希；实际 Chromium 已验证六个视频的播放、后续镜头跳转与手机宽度。检查记录在同目录 `video_check.json` / `browser_check.json`，不代表视频质量评测。

已向 `codex-vg:0.0` 正在追求原 goal 的 CLI 会话传达用户提速要求，并得到明确回复：保留原 submission goal，优先真实视频与有界效果实验，减少不影响近期决策的审计和报告；当前 context OFF/ON 最多两次按原方案收尾。具体近期顺序见 [用户新增优先级](fast-result-priority-20260917.zh.md)。
