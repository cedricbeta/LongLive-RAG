# 约束层 + 成对前瞻 + 分支回退:smoke test 复盘(2026-09-03/04)

这轮改动把 agent runtime 从"实体裁判直接决定接受与否"改成了三层:通用约束层
(`utils/agent_constraints.py`)、成对前瞻和回退(`utils/agent_runtime.py`)、实体裁判
只做适配器(`utils/agent_entity_judge.py`)。渲染侧改成累积编辑计划
(`inference.edit_renoise_plans`)和分支管理(`scripts/run_agent_edit_demo.py` 的
`RenderBranchState`)。产物在 `videos/narrastream_constraint_smoke_20260903/`,图在
`narrastream_constraint_smoke_20260903/`。这份笔记记的是复核结果、两个真实场景暴露的
问题,以及 09-04 据此做的修正。

## 这轮改了什么

runtime 不再判断"这个人对不对",只比较通用约束有没有变差。修复通过当前镜头的绝对
判定之后,不立即提交:把修复版和未修复版在受影响的后续镜头上各切一份,逐约束比较,
只有修复造成清楚退化才撤销。多次修复累积成一个 plan 列表,子分支 replay 父分支的
全部计划再加当前这个。修复失败时交付最近稳定分支上的当前镜头,不再"失败候选里挑
分数最高的"。

## 两个场景跑出来什么

ns190(hotel lobby,单主角 Iris,剧本要求红卷发绿外套)。baseline 镜头 0 里 Iris 是
白发,裁判判 absent + impostor。修一次就过了;前瞻检查了镜头 1、2(all_future 实体
只查最近两次出场),父分支两个镜头都是 identity 违反,候选都满足,提交。镜头 1 到 4
直接通过。镜头 5 又 drift,再修一次也过了,此时 plan 列表是 [0-7] 和 [40-47] 两段,
最终分支 branch-0002 没有隔离镜头。

![ns190](narrastream_constraint_smoke_20260903/ns190_contact_shot_mids.png)

ns290(restaurant prep,Nora/Ethan 在场,Clara 镜头 1 才出场)。镜头 1 baseline 里
Clara 位置站着一个浅发女人,判 absent + impostor,修一次通过(auburn 发、tan 外套都
出来了)。前瞻看镜头 2、3:父分支两个镜头都是 Clara 缺席带两个 impostor;候选镜头 2
一个 impostor,镜头 3 零 impostor 但 Clara 仍缺席。约束比较的结果是 count 改善、
其余不变、Clara 的 attribute 两边都 unknown,提交。然后镜头 2 本身修了三轮全失败,
加一次普通重试也失败,交付 branch-0001 的 attempt 0,隔离。镜头 3 无 mask 可修,
隔离。镜头 4、5 通过。

![ns290](narrastream_constraint_smoke_20260903/ns290_contact_shot_mids.png)

## 复核

四个测试文件在 venv 下 56 passed + 9 subtests。分支测试原本只能在 conda 的 torch 2.9
下跑,因为它为了拿一个纯函数把整个 wan_5b 栈 import 进来了(venv 的 torch 2.1 没有
`torch.nn.attention`)。两条 ledger 逐事件过了一遍,和上面的描述一致。

分支 replay 的前缀一致性是代码里没检查的隐含假设,单独量了一下。ffmpeg 解码成灰度,
按 chunk 算 MAE:

| 对比 | chunk 范围 | 逐帧 MAE 均值 | 最大 |
|---|---|---|---|
| ns190 branch-0001 vs 0002(新编辑在 40-47) | 0-31 | 0.000 | 0.000 |
| | 32-39 | 0.334 | 2.753 |
| | 40-47 | 16.8 | 26.7 |
| ns290 baseline vs branch-0001(编辑在 8-15) | 0-7 | 0.804 | 2.601 |
| | 8-15 | 16.2 | 21.7 |
| | 16-31 | 31-40 | 50 |

差异只出现在编辑点之前 64 帧以内,量级 <3,和 x264 rc-lookahead / mb-tree 的行为
一致,不像 latent 级别的非确定性。但这只是推断,代码里没有任何断言,子分支直接继承
父分支的验证状态。

## 暴露的问题

ns290 镜头 2 不是"修没修好",是没有东西可修。看
`ns290_shot2_mid_parent.png`:镜头中段 Clara 根本不在画面里,唯一的 mask 来自镜头开头
从镜头 1 带过来的残留轨迹。在那块小区域上 renoise 只能长出一个背景小人(见
`ns290_shot2_mid_edit1.png`),于是变成新的 unassigned,三轮都是这样。masked renoise
只能改写已有主体,不能凭空添加实体。剧本里"standing to Nora's left"其实给了布局提示,
但 relation 没有适配器。这是修复工具的覆盖缺口,也是现在最影响指标的一件事。

前瞻配对比较有偏。原实现让父分支和候选分支都从候选的 post-current 状态出发判未来
镜头,也就是父分支的未来是拿候选刚注册的 reference crop 来判的。ns190 里父分支镜头 1
的白发 Iris 被判成"drift",可父分支从来没有红发这个 reference。后果是被修实体的
identity 约束几乎不可能在父分支这边胜出,guard 实际只在检查"有没有弄坏别的实体、
有没有引入 impostor"。

`rollback_to_stable` 和 `_stable_branch_checkpoints` 从未被主循环调用,只有测试覆盖。
ns290 实际的"回到最近稳定分支"走的是 edit_failed 路径的 `stable_parent_fallback`
restore 加交付 attempt 0。但每个镜头都在为它做 memory 快照。

confidence 门槛没起作用。两条 ledger 共 50 次 comparer 调用,confidence 全部是 1.0,
0.75 的注册阈值一次都没触发。VLM 自报置信度不能当门槛用。

命名和字段:`verified_through_shot` 的语义是"最大被接受的镜头号",ns290 branch-0001
显示 5,但镜头 2、3 未解决。`summary.passed` 仍是 true。`preview()` 硬编码
`edited=True`,父分支预览文件名也带 `_edited`。约束比较在空集合时 fail closed,但
legacy 路径完全没有证据时返回 allow_commit=True,两条路径方向不一致。

## 09-04 的修正

前瞻改成从当前候选被判定之前的 verifier 状态出发(`_stage_and_judge` 已经存了这个
快照)。被修实体在两边都没有 reference,于是都和剧本描述比,identity 偏差消失;
已有 reference 的实体照旧。ledger 的 `commit_guard.lookahead.verifier_basis` 记录
用的是哪个基准。在 ns190 这个例子上结论不变(白发对剧本仍是违反),但理由是对的。

删掉 session 层的 `rollback_to_stable` 和每镜头 checkpoint;后端 `rollback_to` 保留
为单元测试覆盖的原语,docstring 写明未接入。跨镜头回滚需要先定策略。

`edit()` 里加前缀一致性检查:渲染子分支时通过 `LLV2_DUMP_LATENT_DIR` 让 pipeline
把输出 latent 落盘,和父分支的 latent 在编辑点之前逐 chunk 比,要求逐位相等;没有
latent(旧渲染)时退回解码帧,阈值 4.0 MAE(实测编码噪声 2.5-2.6,真实差异 16
以上)。不一致直接 raise,因为那意味着确定性 replay 的前提坏了(参考 inductor cache
那条经验)。检查结果记在分支的 `prefix_check`。

latent 路径在 GPU 1、2 上用 ns190 的 edit_shot00 / edit_shot05 两个配置重跑验证过:
dump 形状 [1, 384, 48, 44, 80] bf16,384 个 latent 帧正好 48 chunk 各 8 帧;chunk
0-39 两边逐位相等(max_abs_diff 0),从 chunk 40 起分歧;今天的两个 mp4 和 09-03 的
渲染逐帧 MAE 为 0,说明这台机器当前的 inductor cache 下 replay 是可复现的。一个 dump
130 MB,所以渲染完立刻压成逐 chunk 的 sha256(`chunk_digests.json`,几 KB)并删掉
张量;比较只需要相等与否。

`verified_through_shot` 改名 `last_accepted_shot`,分支上新增 `quarantined_shots`。
删掉 `summary.passed`,退出码改看 `run_completed`。`preview()` 沿用候选自己的
`edited` 标记。legacy 比较在完全没有证据时改为 fail closed。plan 归一化函数挪到
`utils/edit_renoise_plans.py`,分支测试在 venv 下也能跑,四个文件 67 passed + 9 subtests。

## 09-04 下午:实体缺席的放置 action

ns290 镜头 2 那类"没有东西可修"的情形,加了一条新路径,不改生成机制,只改"往哪画"。
贴片家族(cut-paste / cpfreeze)08-20 已经扫完噪声旋钮出局,所以执行器仍然是 mask 内
σ=1.0 全重画加剧本描述;新加的只是在 absent 且没有 impostor 轨迹时合成一个放置区域:

1. caption 里的显式关系。"Clara stands to Nora's left" 解析成 Clara left Nora,锚点是本
   镜头已定位的实体,框取锚点的高和宽、隔一个小间隙放到指定一侧,逐 chunk 跟着锚点走。
   beside / next to 不带方向的选历史所在的一侧,没有历史选空位大的一侧;behind 缩 0.85
   并上移。"to Nora's left" 按画面左理解(benchmark caption 基本是观众视角),几何验证
   用同一约定。
2. 上一镜头的轨迹。实体上次被接受时的框,直接放到同一位置。ns290 镜头 2 走的就是这条。
3. 空位兜底,只在 `--place_absent always` 下启用,默认 `relation_or_history`。

合成框减掉其他已定位实体的 SAM2 轨迹,不碰锚点。mask 进现有的 `edit_shot`,runtime、
pipeline、分支管理都没动。裁判状态多了 `last_tracks`,随 snapshot/restore 一起回滚。
reason 用 `reject_entity_absent_place`,metrics 里 `placement` 记来源和逐 chunk 的框。

顺手把空间关系补成了约束:每条解析出的关系在两个实体都定位到的 chunk 上做几何判定
(left/right 看框中心,beside 看间隙和纵向重叠,behind/front 不可判定记 unknown),
作为 `relation` 约束进入 constraint_assessments。暂不参与 accept,但进入前瞻的
逐约束比较,所以修复把人放错边会被 guard 看到。subset33 的 198 条 caption 里 21 条有
实体间空间关系(beside 8、next to 7、behind 4、to X's left/right 2),所以历史轨迹那条
才是主力。

测试 78 passed。

### ns290 真实运行(09-04 14:00-16:30,GPU 2 渲染 / GPU 3 裁判)

同一条 armC 配置、同一个 baseline,`--place_absent relation_or_history`,每镜头最多 3 轮。
镜头 0、1 和 09-03 一样:镜头 1 的 impostor 重画成 Clara 一次通过,新加的几何关系检查
在修复后判 Clara left Nora 为 satisfied。前瞻看镜头 2、3 后提交。这次的前缀检查在
7 个子分支上全部走 latent 摘要,逐 chunk 相等。

镜头 2 走了放置路径,来源是历史(镜头 1 的 Clara 轨迹,x 0-320 的整列,mask 覆盖 16%)。
关系路线没给出框:Nora 站得靠左,"Nora 左边"的框被画面边缘裁掉超过一半,当时的规则
直接拒绝。已改成贴边放一个 0.6 宽的框,靠锚点自身的 SAM2 轨迹排除来保护锚点,下次
运行生效。

放置的结果:第一轮 σ=1.0 把整列重画成了厨房背景(烤箱、台面),没有人
(`ns290_place_shot2_round1_history_box.png`);第二轮换噪声种子,出来一个 enumerate
认为是主体但三个采样帧上都看不见的框,转成 impostor 重画,第三轮还是没有 Clara。
四次都失败,交付 branch-0001 的 attempt 0,隔离。

![ns290 放置运行:baseline 对最终交付,每镜头中帧](narrastream_constraint_smoke_20260903/ns290_place_contact_shot_mids.png)

镜头 3 是另一种情况:父分支里 Clara 的位置上站着一个蓝色花纹上衣的女人,标准的
impostor,走的是镜头 1 成功过的那条路。但重画之后人被抹掉了
(`ns290_place_shot3_parent_impostor.png` → `ns290_place_shot3_round1_painted_out.png`)。
镜头 3 的 caption 是"Clara, having found the spice jar, steps away to prepare another
dish nearby",模型照着文本让她走开了,而 entity spec 里 Clara 的 shots 是 [1, 2, 3]。
这是 spec 和 caption 之间的冲突,不是修复失败;三轮都一样。

结论比设计时预想的更明确:masked renoise 用的是和 baseline 同一条文本通道。baseline
在镜头 2 的 prompt 切换点就把 Clara 丢了,虽然 caption 明写着她站在 Nora 左边;在同一
区域用同一 prompt 重新采样,成功率大致就是 baseline 保住她的概率,这里很低。镜头 1
能成是因为那里本来就站着一个人形主体,只需改外观。所以"往哪画"不是瓶颈,
"文本通道能不能凭空长出一个人"才是。

顺带一个观察:comparer 的 confidence 依然全是 1.0,以及 enumerate_subjects 会在没有
人的地方报主体(镜头 2 第二轮),metrics 里现在记了 `unassigned_boxes`,下次能对上。

对照:`--placement_prompt spec_only`,放置重画只给实体描述("Clara, a middle-aged
woman with auburn hair and a long tan coat, stands in the scene."),不给整条 caption。
复用同一个 baseline,镜头 0、1 结果相同,镜头 2 第一轮放置同样重画成背景,第二轮又
冒出一个"impostor",第三轮失败,镜头 3 三轮同样把人抹掉。两次运行加起来,放置重画
4 个种子 0 次长出人。prompt 写法这个变量可以关掉。

第二轮那个"impostor"这次能对上了:`unassigned_boxes` 显示它只在 chunk 7,框是
[212, 99, 579, 616],就是 Nora 本人。原因在 locator:8B 模型给 Nora 画的框经常是
[126, 71, 1120, 616],横跨 Nora 和 Ethan 占画面 78% 宽,SAM2 拿这种框做 prompt 得到的
轨迹不一定盖住 Nora 自己,于是 enumerate 出的紧框判成无主。同一个根因也让关系路线的
放置框算不出来(锚点框太大,"左边"没有空间),已经改成用 SAM2 轨迹的紧 bbox 做几何,
单测覆盖。但 locator 松框本身还在:identity 比较用的 crop 也是从这种框裁的,mask 方法
笔记里说过 VLM 指框几何最差,建议改成让 VLM 在 enumerate 出的紧框里选,而不是自己画。
第三轮于是把 Nora 的区域当 Clara 重画,当前镜头裁判把它拒了,没有交付出去,只是浪费
一轮。

两次运行的前缀检查:14 个子分支全部 latent 摘要逐位相等。

## 边界

这次证明的是事务和回退机制正确,不是指标提升。两个场景里只有 ns190 是"有主体可改"
的情形,ns290 恰好暴露了工具的上限,n=2 对 33 场景 arm C 的预测力有限。通用框架
声明支持 presence / identity / attribute / count / relation / action / state,真实
适配器只实现了前四个。

下一步。放置 action 的机制部分(关系解析、框合成、历史轨迹、relation 约束、回退)
留着,它们对任何执行器都需要;执行器要换。候选按代价排:

1. 先把 locator 的松框修掉(在 enumerate 的紧框里选),这同时影响 identity crop、
   impostor 判定和放置几何,是所有实体裁判结论的地基。
2. 把 entity spec 的 shots 和 caption 动词对一遍("steps away / leaves / out of
   sight"),避免 ns290 镜头 3 这种把剧本意图判成缺席的情况。
3. 插入需要比文本更强的条件通道。候选是从相邻 chunk 取 donor 的贴片(pipeline 的
   collage / paste_pin 现成),08-20 在身份修复上出局是因为输给了 σ=1.0 文本重画,而
   插入场景里文本重画是零,基线不同;或者 KV 级写入。这一步先别在 33 场景上跑。
4. confidence 门槛换成 crop 间一致性或 reference 互换比较。
