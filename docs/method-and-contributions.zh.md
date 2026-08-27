# 方法与贡献整理:写入式记忆维护

2026-08-27 整理。这份文档把过去两个月的实验收敛成论文骨架:我们的方法到底是
什么、每条 claim 站在哪个实验上、相对同期工作(尤其 IAMFlow)的贡献在哪。
风格是台账不是宣传稿;每个数字都能在引用的文档里找到出处。

## 一、问题与定位

自回归长视频生成里,角色/物体跨镜头的身份漂移是公认痛点。主流解法是往注意力
上下文里**追加**记忆:LongLive 的 sink token、LongLive-RAG 的同场景检索、
MemFlow 的动态记忆帧、IAMFlow 的身份感知 KV 检索注入——机制上全是同一族,
差别只在"选哪些 KV 追加进去"。

我们的位置:这一族方法有一个共同的未检验假设——**追加进上下文的 KV 会被
注意力读取并作用于主体**。我们用探针数据证伪了这个假设,并给出一条不与
softmax 竞争的替代通道:**写入式记忆维护**——不往上下文里塞记忆,而是在
验收环里直接改写画错的内容,让修好的内容作为下一个 chunk 的历史被自然读取。

## 二、机理发现:为什么追加式有天花板

出处:[attention-downweighting-study.zh.md](attention-downweighting-study.zh.md)。
只读探针,离线重放与真实窗口对到 1e-5,全部分解在 logit 空间。

1. **recency 垄断**。镜头边界处当前 chunk 拿走 66%(L0)–84%(L29) 的注意力
   mass,对任何旧内容有 1.5–2.5 nat 的 logit 优势。去偏置(λ=0)后的真实
   降权是 2.3×(L0) 到 8–10×(L14/L29)。这不是"外来内容被歧视"——L0 把注入
   块/sink/pinned 压到完全相同的份额,内容折扣只在中层出现且定量可解释。
2. **竞争解释被排除**。RoPE 几何衰减凑不出缺口(44/128 旋转维,32 帧只掉
   16% vs 3 nat 缺口);槽位特权杠杆最多 +47%;剂量抬 mass 不抬一致性
   (Round 14)。垄断是局部窗口训练出来的结构性先验。
3. **层错配与无主体感读取**。身份最可解码的 L14(G1 探测)恰是无偏降权最重
   的层之一;且没有任何层的帧内读取模式描出主体轮廓(L14 是枢纽 token 式,
   亮点不在主体上)。就算列被读了,读的也不是身份。
4. **自洽性**。该机制同时解释:我们 Round 13–18 追加式注入的全部 null、
   LongLive-RAG 同场景检索为什么有效(同场景内容天然相似,不吃跨镜头折扣)、
   修复传播半衰期为什么只有 1–2 块。

推论:任何追加进上下文的记忆列都要和一个自带 ~2 nat 优势的在位者打 softmax,
选得再准也赢不了。这直接命中 IAMFlow 一族"检索层做到头"的方法链条。

## 三、方法:验收环里的写入式记忆维护

三个组件,每个都有独立实验背书:

**对比式裁判(何时修)**。VLM 拿参照帧问"相对开场变了没有"——绝对式判断只
抓灾难,对比式才抓漂移(凤冠鸠案例);负池 margin 信号预测漂移 AUC 0.787,
只用正池 0.605。出处:vlm_agent_sameshot 台账。

**masked renoise(怎么修)**。CLIPSeg mask 内加噪、换修复 prompt 重去噪、
mask 外逐步钉回、修复结果进 clean recache。σ=1.0 满噪声档收回 66% 的 DINO
身份差距(0.567→0.744);中档 σ 弱是因为错误旧内容残留在低频里。失败分型学
调度升级路径:强度不足→σ=1.0/双遍;文本带宽不足→TI2V/VACE/V-edit 辅助;
先验缺失→只能改 prompt。出处:[editing-repair-report.zh.md](editing-repair-report.zh.md)。

**逐块修复 + 复核提交(在哪修、修完怎么办)**。传播半衰期 1–2 块,修复必须
直接作用于每个坏块,不能指望自由传播;蝴蝶效应(修完的鱼在下个镜头被烧掉)
证明编辑不能脱离验收环。所以形态是 shot-buffered 状态机:provisional shot →
裁判 → (通过 commit | 漂移 → edit_shot 修复 → 复核 → commit 修复版 | 降级
restore+retry / accept-with-quarantine),全程 ledger 可回放。出处:agentic
runtime v1 + 本轮 edit_shot 接线。

辅助通道:in-place V-edit(capsule pilot 的幸存遗产)有验证过的因果力
(±0.09 vs 0.01 门槛),作为分型 b 的方向约束执行器保留。

关键机理论证:修好的内容就是下一个 chunk 的历史,**继承 recency 垄断而不是
对抗它**——这是写入式相对追加式的结构性优势,不是工程调参优势。

## 四、Contributions(论文式表述)

1. **机理**:首次定量刻画自回归视频 DiT 中记忆注入失效的根因——recency
   垄断(66–84% mass、无偏降权 2.3–10.6×)、层错配(身份最可解码层降权最重)、
   无主体感读取模式;并据此对追加式记忆一族(sink/检索/动态记忆帧/身份感知
   注入)给出统一的天花板解释。证据:只读探针 + λ=0 离线重放 + 头普查。
2. **通道**:验证两条不与 softmax 竞争的写入通道的因果力——in-place V-edit
   (±0.09 摆动 vs 0.01 门槛)与 masked renoise(σ=1.0 收回 66% 身份差距),
   并测清其约束(传播半衰期 1–2 块、蝴蝶效应、σ-保真/协调 trade-off、失败
   三分型)。
3. **系统**:把修复器嵌进带审计的验收环(对比式裁判、AUC 0.787 漂移预警、
   retry budget、ledger 回放),形成 training-free 的写入式记忆维护闭环;
   在 NarraStream-Bench 上与追加式机制类对照(预注册,含对我们自己机理结论
   的可证伪预测)。
4. **负结果(诚实资产)**:低秩线性身份 capsule 双估计器判死(身份在 memory-V
   空间弥散);追加路线的位置/选择/剂量/偏置杠杆逐一排除;最后一搏(时机/
   头定向/相位匹配)预注册进行中——赢了是新杠杆,输了是"穷尽式排除"的完备性。

## 五、与相关工作的差异

| 工作 | 机制层级 | 身份表示 | 记忆进入模型的方式 | 我们的差异点 |
|---|---|---|---|---|
| LongLive | 位置锚 | 无 | sink token 常驻 | 静态锚不更新,其 baseline 地位被 IAMFlow 和我们共用 |
| LongLive-RAG | 同场景检索 | 隐式(AE embedding) | 追加列 | 只在同场景(同质内容)有效,恰是垄断机制的边界条件 |
| MemFlow | 语义检索 | 帧级特征 | 追加列 | feature conflation,IAMFlow 已批判 |
| IAMFlow | 符号注册表 | LLM 实体 ID + 属性 JSON | 追加列(KV concat) | 解决"选什么",未检验"会不会被读";其 concat 注入正是我们测到被 2.3–10.6× 降权的机制 |
| GroundShot | agent 验收 | 实体记忆 | prompt/重生成 | 无模型内部通道;我们的 novelty 在 model-native 写入 |
| 我们 | 机理 + 写入 | 模型自身 KV/latent | **改写将被读的内容** | 唯一不与 softmax 竞争的路线,机理与因果双背书 |

IAMFlow 与我们是正交层级:它管 what to keep,我们管 whether it takes effect。
三臂对照实验(预注册见 [laststand-and-narrastream-prereg.zh.md](laststand-and-narrastream-prereg.zh.md))
把这句话变成一张表:concat 注入臂同时报注入列实际 attention mass 和实体级
一致性——机理预测它 mass 被压、指标不涨;修复闭环臂预测在 CLC/EG 上 ≥2σ 胜
baseline。若 concat 臂意外胜出,机理结论修正——输赢都有信息量。

## 六、证据地图(claim → 实验 → 状态)

| claim | 实验 | 状态 |
|---|---|---|
| recency 垄断 + 降权定量 | attn_probe 两场景 52GB 探针 + λ=0 重放 | 完成,已提交 |
| 身份弥散,低秩线性不可得 | capsule G1 + G1b 双估计器 | 完成,永久关闭 |
| V-edit 因果力 | channel test 4 臂 | 完成,通道有效 |
| renoise 修复通道 + σ=1.0 上限 | 8+N 臂 + MovieGenBench 抽查 | 完成 |
| 传播不可依赖 | 方案 B 传播测试 | 完成,半衰期 1–2 块 |
| 对比式裁判 + margin 预警 | sameshot + Track W | 完成,AUC 0.787 |
| VLM 收益 vs 机制收益 | 七臂静态对照 | 预注册,未跑 |
| 追加路线穷尽 | 最后一搏 4 renders | 预注册,实现中 |
| 写入 vs 追加 主表 | NarraStream 三臂 | 预注册,依赖 edit_shot 接线 |
| 实体级指标 | NarraStream EG/CLC 采纳 | 接入完成,部分偿还 M0 |

## 七、诚实的短板

- 修复方向性:renoise 对分型 b(文本带宽不足)只能靠升级执行器,V-edit 方向
  控制还没和 renoise 合流成型;
- benchmark 只测人物实体,非人主体(鱼、怀表)靠内部 frozen hard benchmark;
- 与 IAMFlow 的对照用我们的 KV-RAG 臂做机制类代理而非复现其全套 LLM 注册表
  ——论文里要写清楚这是 mechanism-class 对照不是 system 对照;
- 60s 终版渲染和 30B judge 的算力账还没付。
