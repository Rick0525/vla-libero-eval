# STATUS（开发期工作文档，W7 转公开前决定翻译或移除）

> 更新规则：进度一变就更新本文件（CLAUDE.md 约定）。

## 当前状态：W4 D4（2026-08-06）

**头条：leaderboard v2 定稿（[results/leaderboard_zh.md](results/leaderboard_zh.md)，8/06，Rick 拍板同秤升主表）**：主表四行统一 MuJoCo 3.2.7——OFT 特训 **97.4** = 合训 **97.4** ＞ π0.5 **94.4** ＞ SmolVLA **87.4**；v1 的 3.8.1 数据移入新设「版本敏感度」节作为独立发现（三幕：悬空 init 语义外包 → 3.4.0 box-box bugfix 打破落座前提 → 同物理 A/B 实测根治；版本伪影=官方协议不覆盖的坏前提压力测试，SmolVLA vs π0.5 拉开 58pp）。**init3 ×10 探针 @3.2.7 开牌（8/06）：SmolVLA 9/10、π0.5 10/10——「毒布局」定谳为版本伪影**，健康物理下 init3 不难；W3 的 0/13 硬零 / 上采样破零 / 5/10 真难等叙事按 attribution 8/06 更新框重读（原文保留，六处更新框）。π0.5 复现差账更新：z=1.07 p=0.28（更稳）。

（历史头条：v1 榜 93.8/83.8 @3.8.1（7/31）；SmolVLA 30k/82.0 @n=100（7/20）。）

### W1（7/14–7/20）：基线与重训

- [x] 仓库脚手架：git init、MIT、双语 README、.gitignore（07-14）
- [x] smolvla_base zero-shot baseline：**N/A，结构性不可评**，证据见 `results/smolvla_base_zero_shot.md`（07-14）
- [x] 首次微调（lerobot 文档示例配方）：43%，三重根因定谳——①评测口径 n_action_steps=50 砍 ~35pp ②batch 4 vs 论文 64 ③lr 调度 30k 触底后 70% 步数白跑（07-19）
- [x] 重训（论文对齐配方：全局 batch 64 双卡 DDP / bf16 / 30k 步与 lr 调度对齐）：4h02m，**82.0%**；25k=84.0 统计平手、有 winner's curse 不报（07-20）
- [x] 成功率曲线（评测 daemon 独立 GPU 出分，n=10 × 50 eps/点）：5k=52 → 15k=72 → 30k=74，15k 后平台

### W2（7/21–7/26）：数据路径破案 + 100k 对照 + 失败归因

- [x] **dataloader 50× 退化破案与修复（patch A）**：delta 查询被 datasets ≥4.4 的 custom-format 闸门打回整行取+全量 PNG 解码（白解 100 张 PNG 只为拿 50×7 个 float）；缓存 select_columns 列视图修复，200.7→4.4 ms/样本（46×），逐 tensor `torch.equal` 零失配；生产验证 data wait 38%→~1%，30k 训练 4h02m→≈1.9h。补丁 `patches/lerobot-delta-query-column-views.patch`
- [x] **compile 四格矩阵（各 600 步 DDP 探针）**：default mode 胜出（4.33 step/s vs eager 3.49）；max-autotune 复现 W1 崩溃指纹，定谳为 torch 2.11 inductor 在 sm_80 上的真 bug（pytorch#95335 家族），本栈无解、升 torch 前不碰（W7 重估）
- [x] **b64_100k 对照跑（7/22）**：6h31m 零故障，官方口径终点 **70.0%** vs 30k 的 82.0——「更长余弦=更高终点」不成立，23 epochs 疑似过拟合倒贴；负结果入 leaderboard 证据。task4 新型崩塌（1/10，30k 时代非弱项）10 段视频留档归因素材
- [x] **失败归因标注定谳（7/24，Rick 标注拍板 + Claude 抽帧核验悬案）**：task5/8 全 12 段失败——**H1 指错碗 = 0/12，语言 grounding 出局（全表最硬结论）**；task5 主因 H2 抓取（5/8，含 2 例「双碗同抓」新子模式：下爪过深把 bowl+ramekin 当整体夹走），task8 主因 H3 放置（3/4，盘缘放置）。LIBERO `On` 谓词（直接接触 + 圆心 xy<3cm）比直觉严，放宽计数则 task5 2→3、task8 6→8。正式口径 n_action_steps=1 每步重推理 → **失败不是开环伪影，是「看见了也不会改」**，修正缺口在策略本身（无失败恢复演示，失败态即 OOD）。结论已回填 report Failure analysis 节；标注表 `results/failure_videos/ANNOTATION_zh.md`
- [~] **三层归因体系（感知/规划/控制）**：R/F/G/T 四实验 7/27 一日收官，全档 `results/attribution_framework_zh.md`。**头条：task5 归因定谳=控制层（抓取几何先验）**——「布局→抓取侧→在手偏心→落点中心→3cm 谓词」因果链三层证据闭环（观察 17/17 + 天然对照 + 干预 n=20/组、Fisher p≈0.0098；同布局换抓取侧成功率 5%→40%，P-R≡O6 布局效应 100% 中介）。副产物：可复现性定谳（三跑 82/84/84 聚合磐石、逐集同栈同 seed 也仅 84/100 非确定）；接管仪器五轮排障（get_state 不含 ctrl/current_action，v5 反解恢复，社区无先例可回馈 LIBERO #16）；接管税结案=源状态质量（T3 双源 10/10 vs 3100 控制 5/10）；**抓姿质量指标化（Exp V）**：碗倾角/腕轴偏角双指标样本内分档干净，但 3101 预注册注入 10/10 **样本外证伪**（其自然落点 1.92cm 实为好档——「源姿=落点档位」四连中，档位的状态空间编码仍是悬案，新指标准入门槛=分开 3100/3101 对）。待办：Rick 盲评 3101 源视频、`--deterministic` 模式、标注工具化、task4 实弹、修复实验设计（数据侧，且已定向**内容覆盖而非数量**：10 任务 demo 数 vs 成功率 Spearman≈0——最少 demo 的 stove 任务 35 条 90%，task5 39 条 23%）**→ 8/06 Rick 拍板：修复实验已随 Exp D/E 收档，其余四条（盲评 3101 / --deterministic / 标注工具化 / task4 实弹）全部销账不做——科学线已闭环，做了不改变任何结论**

### W3（7/28–）：修复实验（Exp D/E）+ 并行评测判别

- [x] **Exp D 训练数据审计（7/28）**：task5 全 39 条演示摆位/抓法审计——深负带 demo 密度正常（12/50≈均匀）、人类无条件抓右缘、「负 y→左缘」系策略自产分布外行为 → **已测维度数据无病灶**，修复选型转难例加权上采样（非补稀疏）
- [x] **Exp E 双臂对照（7/28–29）**：K=4 boost（27 集深负带上采样，帧份额 0.39%→1.54%）vs 完全复刻对照臂，顺序双卡 DDP 各 ~2h + 全自动评测链过夜。**开牌：靶点修中——init3 毒布局历史 0/13 首次破零 → 4/10、task5 8/30→11/30；护栏击穿——其余 9 任务合计 −14 集（t3 −9）**。行为学三机制定谳（接触几何漂移 / 目标锚定漂移「指令 cookie box 身体去 ramekin」/ 闭环退化，Rick 全片裁决）：上采样放大的是演示的整条轨迹分布，经共享表征伤及「碗在底座上」近亲任务。预注册两注一比一（task5 差分 +3 压线；init3 4/10）
- [x] **Exp E-C 配平臂（7/29）→ Exp E 三臂收档**：锚点帧份额拉平（t3+34/t7+22 盲选复制）重训——**t3 如预言修复（−9→−1），但 t7 反而炸穿（−4→−10）、t5 收益蒸发（init3 4/10→1/10）**；预注册两共识注全错、地鼠注字面 Claude 胜/精神 Rick 胜（塌在被干预任务内部而非雷区）。**终局结论：0.45B 上逐集复制式定向上采样是钝器，干预不可局部控制，打地鼠是路线内在属性；出路在训练侧或更多样演示**（limitation：每变体 n=1 重训彩票）。全档见 `results/attribution_framework_zh.md` Exp E/E-C 节
- [x] **并行评测判别三连（7/28–29）**：①asyncb10 结果漂移出 sync-b1 噪声带 → **不采纳，官方口径维持 sync-b1**（六样本均 82.2 vs 84.0，统计上未证有偏，采纳闸门未过）；②批量本身不赚钱（syncb10 1609s ≈ sync-b1 1444s，锁步惩罚吃掉提前退出收益）；③**EGL 渲染串行化证伪**——摊卡判别实验（渲染 per-worker 轮转到 GPU1/2/012）仅 1.2× 且剂量立即饱和、GPU util 24%/8% 远不饱和 → 真瓶颈在 CPU 侧锁步编排（每步 IPC×10 + max-of-10 等待）。旋钮工具化入 repo：`EVAL_USE_ASYNC`、`EVAL_ENTRY`、`EVAL_EGL_DEVICES`（白捡 1.2×）；大加速真旋钮 = n_action_steps 或 max_parallel_tasks（待验，W4 RLinf 铺垫）
- [ ] 顺手上游素材：HuggingFaceVLA/libero meta per-episode 指针 1690/1693 全错（官方 dataset_tools 在该集全废，split 首崩），可报 lerobot / HF dataset repo
- [x] **leaderboard 扩模型（7/30 发射，7/31 开牌收档 → [results/leaderboard.md](results/leaderboard.md)）**：口径定稿=统一套件/init 布局/集数（50/任务），推理配置各用官方（SmolVLA n_action_steps=1 / π0.5 chunk50 执行 10 / OFT chunk8 开环）逐列注明。四局零事故：**OFT 特训 97.4 = 合训 97.4（合训税 0，逐任务差 ≤±2）、π0.5 93.8、SmolVLA 83.8（CI ±3.2pp）**。踩坑入档：PaliGemma tokenizer gated 需 token；`hf download` 手工缓存缺 `.no_exist` 标记致离线假报连接错（SOP：在线真加载一次）；OFT 官方脚本改写本地 checkpoint 目录（有备份）；OFT 栈 spatial 每集 220+10 步 vs lerobot 栈 280（如实注明，OFT 在更短预算下仍领先）
- [x] **π0.5 复现差定谳（7/31）**：97.5 系四套件平均，spatial 目标实为 **97.0@n=100**；z=1.26 p=0.21 不显著 + 官方协议只访问 init states 0–9（t7 十个毒布局 8 个编号 ≥10，官方结构性看不见）→ 复现差不存在；社区 #2114 报 77%，我方 93.8 在复现分布高端
- [x] **π0.5 失败归因收档（7/31，29/29 全片裁决 ×2 独立 run）**：**t7/t9 = 「on the stove」↔「on the wooden cabinet」对称目标混淆**（t7 帧证 10/10 拿柜顶碗、放置动作无瑕、谓词永不亮；失败集跨 run 逐集重合=布局锁定）；**t5/t4 = 抓取失败**（13/13 抓不起或偏心抓，跨 run 低重合=采样波动）。**失败可复现性即机制指纹**（离散决策错误→锁定；接触几何边缘→波动）。跨模型：容量改变失败机制构成而非单调修复——0.45B 短板接触几何、3.3B 换成空间指代混淆、7.5B 两者皆罩。工具入库 `lerobot_eval_fullvideo.py`（EVAL_RENDER_ALL_N 解 10 集录像上限）
- [x] **init3 毒布局跨模型探针（7/31）**：SmolVLA 0/13 → +定向上采样 4/10 → π0.5 **5/10**（探针 seeds 2000–2009）→ OFT 特训/合训各 **1/1**（官方 eval 按集序号索引 init states，ep3 即 init3，免费开牌）。口径不对称入档：随机策略=命中率、确定性策略（OFT L1 头 + cudnn.deterministic）=0/1 覆盖，不可同格比较（→ 8/02 修订：OFT「确定性」证伪，1/1 降级 n=1 口径，见下）
- [x] **OFT 确定性证伪 + 榜单修订（8/02，Rick 起题「把失败集重跑一遍」）**：同权重/config（md5 核对）/同调用整套重跑——特训（同卡 GPU2）487→490 翻 7 集、合训（GPU1，卡因素未单独排除）487→489 翻 6 集 → **「OFT 是确定性的」撤回**、v1「两臂 t5 失败不重叠=权重差异翻盘」撤回（同臂自己就换失败集 {33,39}→{8,33}；仅 t5 集 22/33 跨臂反向 n=2 存活降级待复核）；**前向逐位探针**（固定合成观测,进程内 ×10 + 跨进程哈希全同）洗清策略头 → 非确定性住闭环 env/渲染链路（W2 lerobot 栈同 seed 非确定同款,两栈共同点 robosuite/MuJoCo+EGL）；合训税=0 升级 2×2 稳健账（跑内 ≤±1、跨臂 ≤±2）。**t7/t9 全失败看片（Rick 裁决,两臂×两跑）**：OFT 无一例指代混淆,全属接触几何家族 → 对称目标混淆定谳 π0.5(3.3B) 特有;「偏心抓→盘缘」家族 0.45/3.3/7.5B 三容量点集齐;成败边界穿过同一行为内部（成功跑也瞬时松爪,差在是否掉碗）。**指纹框架修正**：重合率以扰动强度为标尺（π0.5 采样=强扰动下锁定才是布局决定;OFT 数值噪声=弱扰动,接触失败也能锁定）。**松爪-块边界假说带动作日志检验（oft_action_trace.py）**：字面证伪——运送松爪系物理滑移非指令（成功 trace 指令零翻转）;意外①失败 trace 指令反转向块边界聚集（12 次瞬时张开 7 次在新 chunk 首步,「上 chunk 末 CLOSE→新 chunk 首 OPEN」反转对 2/3 出现,重抓场景伪影）;意外②单集协议下双跑锁定失败集 e14/e36 翻成功——跨集 env 上下文敏感（候选 warmstart 携带）,**单集探针≠全套协议**新协议教训。leaderboard_zh 八处修订（英文待 Rick 审毕）,裁决档 failure_videos/oft_t7|t9/,净化脚本 oft_bitstability_probe.py + oft_action_trace.py 入库。四项拍板（8/02 深夜落定）：剩余失败全拉（31 段入 oft_rest/ 含 t5 翻转对成功对照,待 Rick 看）、首分叉探针做、GPU2 复测不做、头条表不动
- [x] **首分叉探针定谳：非确定源=EGL 离屏渲染（8/02 深夜）**：两同调用进程各跑 t7 全 50 集,逐步哈希五量（MuJoCo 全状态/策略输入/主相机/腕相机/动作,oft_divergence_probe.py + oft_div_analyze.py 入库）——**50/50 集分叉,首分叉 100% 含渲染字段（腕相机先漂 45/50、主相机 9/50）,物理与策略在输入相同处零先漂**;两探针跑结局零翻转,t7 失败集四跑全同 {14,36,49}（两官方+两探针）→ 像素噪声全域存在、仅边缘布局翻结局;t7 三集系顺序协议下对噪声鲁棒的真·深失败,单集协议翻 e14/e36 属系统性跨集上下文效应（warmstart 候选未验证）,e49 两协议皆败最深。lerobot 栈 W2 同 seed 非确定疑同根,**可回馈上游（robosuite/LIBERO/lerobot）**。leaderboard_zh 局限节收口（分叉位置从「未定位」改「已定位」）
- [x] **warmstart 候选开庭定谳：边界 warmstart=协议效应携带者（e14 因果闭合）+ 第二携带者立案（8/03）**：先免模型双预检——env.reset() 实测系**硬重置**（MjSim 逐集销毁重建，「上集求解器记忆直接残留」最简版本出局）；两种历史抵同一 e14 边界逐字段对账：qpos/qvel/ctrl/sensordata/sim_state **逐位全同**、唯 `qacc_warmstart`（=qacc，reset 链 forward 以上次 warmstart 为求解种子重算）随历史变——状态侧唯一活通道。2×2 干预开牌（{顺序 50 集, 单集镜像 14/36/49}×{自然, 清零}，清零臂各 ×2，`oft_warmstart_probe.py`）：**e14 同钉死状态下，自然单集边界 ws→成 ×2、清零 ws→败 ×2、清零后两协议归一（都败）——边界 warmstart 两取值两结局，因果闭合**。机制画像：顺序臂两跑 50 边界 ws 哈希逐位全同 + 单集臂两跑 3 边界全同 + e14 边界与预检独立复测同值 → **边界 ws 是「进程内第几次 reset」的确定性位置函数，不携带前集 rollout 内容**（硬重置切断内容通道），协议效应 4/4 vs 2/2 的系统性由此解释。**尾巴=e36**：两臂同清零同钉态仍臂内稳定、臂间分歧（顺序败 ×2 / 单集成 ×2）→ **第二确定性携带者立案**（候选：RNG 流位置、渲染/进程历史；n=2 未铁，按独立硬币算 2-2 恰按臂分布概率 ~1/8）。副产物：**e31** 清零顺序臂内翻转（run1 败/run2 成）=新边缘布局入 7/500 翻转家族；**e49 十跑全败**（自然 6 + 清零 4）对协议/ws/噪声全鲁棒=真·最深失败；**渲染噪声底实测**：同状态背靠背两次渲染即 ~9% 像素不同、幅度达 182/255（物体边缘反走样翻转）——像素噪声「全域存在」的量化注脚。工具入库 `oft_warmstart_probe.py` + `oft_warmstart_precheck2.py`；logs：`oft_ws_{seq,single}_t7[_run2]_20260803.log`（服务器）。**两案白话版全案卷：[results/oft_nondeterminism_case_zh.md](results/oft_nondeterminism_case_zh.md)**（渲染鬼+草稿纸鬼,写给复习与面试）
- [x] **第二携带者定谳：e36 跨集通道=全局 RNG 流位置（8/03 傍晚,Rick 拍板追一轮封顶）**：RNG 钉住干预（清零基础上每集边界 `set_seed_everywhere`,`oft_ws_rngpin_probe.py`）×两协议×2 跑——**顺序臂 e36 从稳定败（自然 ×4+清零 ×3）翻成 ×2,与单集臂归一** → 定谳。消费者定位（免模型 tracer v2）：**全在 env.reset() 内**——摆位采样器 x/y/quat（~35 draws/reset）+ 机器人初始位 randn（single_arm.py:176）;步进/set_init_state 零消费（tracer v1 假阴性系 MT19937 每 624 draws 才翻搅状态数组、哈希未含 pos 指针,precheck2 rng 行同此误读——实为每次 reset 都消费）。**末跳悬置（封顶线）**：抽样值被 set_init_state 全覆盖、控制器/robot Python 态 30 字段 dump 全 PINNED（含 init_qpos,LIBERO robot 初始化实测确定性）——RNG 位置的存活表达不在 mjData/控制器,余下嫌疑=重置期以随机摆位渲染过的帧经驱动层影响后续渲染（与案一同层）,按约定封顶不追;工程上钉随机数已足以消掉该效应。**e14 修订**：zero-seq run3 翻成（F,F,S）——e14 在顺序上下文贴边到渲染噪声可翻;单集臂因果对照更铁（自然 S×2 vs 清零 F×3）,「清零后协议归一」降级为统计性。**边缘布局群体量化**：各扰动政权各出一批失败集（rngpin-seq 新增 {23,32}×2 稳定 + {19,42} 单跑）,累计边缘集 {14,19,23,31,32,36,42} ≈14% 布局微扰可及——**协议任何变更（含修 bug）都会换一批边缘集,逐任务数字天生 ±数集摆动**,跨协议版本比榜要留此心眼。e49 16/16 全败仍最深。上游修复建议成型：边界清 warmstart + 逐集定种子。工具入库 `oft_ws_rngpin_probe.py`+`oft_rng_consumer_tracer2.py`+`oft_ctrl_state_dump.py`;logs `oft_wsrng_*`/`oft_ws_*run3*`（服务器）

- [x] **跨栈 MuJoCo 版本混杂立案（8/04 深夜，Rick 起题「task5 当时环境是什么」；8/05 补跑 + 8/06 v2 定稿后销账）**：盘点两栈——lerobot 栈 `.venv` = mujoco **3.8.1**；OFT 官方栈 `.venv-oft` = mujoco **3.2.7**（Python 3.10.20 / torch 2.2.0+cu121 / transformers 4.40.1 / robosuite 1.4.1；官方依赖链**无显式钉版**，`robosuite>=2.3.0` 下 W3 安装期 resolver 解得，恰落健康区）。姊妹项目 vla-rl-post-training 8/04 干预四跑定谳版本效应（同一 GRPO ckpt 贪心 500：3.2.7→92.8/task5 98%、3.3.0→94.2/98%、3.8.1→85.0/12%、3.9.0→86.0/19%，边界 (3.3.0, 3.8.1]；且满血 OFT @3.9.0 task5 亦塌 52%、剔 task5 98.2≈harness 97.4，三模型家族同塌 ⇒ env 侧效应与策略无关）→ **task5 容量阶梯 28→86→96 与头条排序是跨秤比较**（SmolVLA/π0.5 在加难区 3.8.1、OFT 在健康区 3.2.7），OFT 一级领先中版本成分未分离；同秤比较全保留（SmolVLA↔π0.5、OFT 特训↔合训、一切栈内干预/侦破结论）。钩沉修正：`failure_videos/ckpt030000_n1/` 系 7/20 原跑（旧 conda venv，确切 mujoco 不可考，重放 L1/L2 对账提示同难度档）；`attr_f_task5_init3/` 系 uv venv 3.8.1。已披露入 leaderboard_zh（日期戳+协议+容量轴+三行账+局限首条+栈差异条）与 attribution_framework_zh / ANNOTATION_zh 头注。**补跑已定（Rick 8/04）：SmolVLA + π0.5 健康区版本各 500 集；版本选 3.2.7 还是 3.3.0 待拍板**（3.2.7=与 OFT 现数同秤、OFT 四跑全复用、官方环境零改动；3.3.0=与姊妹项目同秤、但 OFT 两臂须重跑且动官方环境）。GPU 纪律：补跑用 GPU2（GPU0/1 让给姊妹项目）
- [x] **mj3.2.7 补跑开牌（8/05 凌晨，GPU2 双跑全绿，venv 已自动还原 3.8.1）**：π0.5 **94.4**（472/500；task5 43→45，六任务逐集持平，总 +3 集≈跑间噪声）——**π0.5 对版本效应近乎免疫**；SmolVLA 30k **87.4**（437/500）——**task5 14→40/50（28%→80%，+26 集），塌方主体定谳为 3.8.1 版本伪影**，其余任务 ±5 内摆动（t0 −5 / t4 −3 / t8 +2，本栈逐任务噪声带内）；task5 init3 本跑成功（n=1，「硬零档」结论限 3.8.1 物理）。**同秤（3.2.7）新账**：task5 阶梯 80→90→96%、总分 87.4→94.4→97.4——排序不变、缝隙大幅收窄，SmolVLA 距论文 ~90 的缺口从 6.2pp 缩到 2.6pp。**版本敏感度=策略依赖**（OFT 家族强敏感 / SmolVLA 强敏感 / π0.5 免疫）→ 项目 2「env 效应对任意策略成立」需限定，列为跨项目回馈素材。输出 `eval_runs/{pi05_libero,smolvla_spatial_b64_30k_ckpt030000_n1}_official500_mj327_20260804`；披露已回写 leaderboard_zh（局限首条量级实测化+容量轴 8/05 行+三行账+日期戳）与 attribution/ANNOTATION 头注。**待 Rick 拍板**：①leaderboard v2 呈现（建议 3.2.7 同秤数字升主表、3.8.1 老数移入「版本敏感度」节作为发现展示）；②init3 ×10 探针 @3.2.7 复测毒布局命中率（~15 分钟）；③attribution 文档 task5 章节修订深度（→ 8/06 三项全拍板并落地，见 W4 节）

### W4（8/03–）：收尾与产出转化（主战场移姊妹项目 vla-rl-post-training）

- [x] **OFT 非确定性两案封顶收案（8/03）**：见 W3 末两条与 `results/oft_nondeterminism_case_zh.md`
- [x] **leaderboard v2 定稿（8/06，Rick 三项拍板全落地）**：①主表升 3.2.7 同秤（87.4/94.4/97.4，CI/z 检验全部重算），3.8.1 老数移入新设「版本敏感度」节（三幕事故链+跨模型鲁棒性表+方法论三条）；②**init3 ×10 探针 @3.2.7 开牌：SmolVLA 9/10（历史 0/13）、π0.5 10/10（3.8.1 探针 5/10）——毒布局=版本伪影定谳**，毒布局探针节整体改写（连锁改判：上采样「破零」=坏前提下捡救能力、「容量阶梯在最难布局单调」撤回）；③attribution 原文不动+六处 8/06 更新框（R/F/G/D/E 各节+附录收官注脚），头注「待复测」销账。探针跑档：`run_probe_init3_mj327_20260806.sh`（GPU2，降版自动还原已验证）
- [x] **mj 探针脚本入库（8/06，Rick 拍板归 harness）**：`mj_settle_probe.py`（六版本落座矩阵，边界=3.4.0）、`mj_reseat_probe.py`（重生成 pre-settled init + 座姿验证）、`run_smolvla_task5_reseat_20260805.sh` / `run_oft_reseat381_20260805.sh`（同物理 A/B 两臂）、`run_leaderboard_mj327_rerun_20260804.sh`（v2 主表补跑）、`run_probe_init3_mj327_20260806.sh`；姊妹项目 findings 交叉引用
- [x] **收官范围拍板（8/06 下午，Rick）**：①Exp T 尾巴四条销账（见 W2 三层归因条目尾注）；②**套件范围选 (a) 不扩**——Object/Goal/Long 与 GR00T 不纳入，简历口径改「LIBERO-Spatial 深度评测：leaderboard + 版本敏感度发现 + 三层归因」（深挖故事优于裸广度；SmolVLA 行扩套件需逐套重训，结构上不可行）；③EGL 非确定性两案是否上游回馈**待拍板**（Claude 建议不发、素材转 W7 博客，详解已给 Rick，8/06 晚定）；④剩余看片两件排上：oft_rest 31 段（`results/failure_videos/oft_rest/`，已在库）+ SmolVLA task9（失败集从未录像，8/06 已发射 t9 全录像重跑 @3.2.7 GPU2，`run_smolvla_t9_fullvideo_mj327_20260806.sh`）
- [ ] **上游 issue 包（靶向 8/05 终拍，草稿在途）**：huggingface/LIBERO 主 issue + Lifelong-Robot-Learning/LIBERO #141 回帖 + lerobot 交叉引用 + RLinf 钉版；mujoco issue 已撤销。**1690/1693 meta 指针案有新情况（8/06 核查）：HF dataset repo 已有他人报案（HuggingFaceVLA/libero discussion #10，2026-08-04，open 未获官方回应，内容与我方发现完全一致但只举 object 套件例证）——不开重复案，改为补充评论（全量统计 1690/1693 + split 首崩 + 手工 parquet workaround），并入 issue 包待 Rick 审后亲发**

## 上游贡献（lerobot #2895）

- [x] **根因评论发布（7/23）**：定谳 custom-format 闸门机制 + 四方案同机基准（现状 165ms / select() 185ms 更慢 / 列视图 0.8ms ≈200×）+ 集体误判来源考古（#2549 时代 <4.0 形态的先例）→ [评论](https://github.com/huggingface/lerobot/issues/2895#issuecomment-5060174507)
- [x] **lhoestq（datasets 负责人）16 分钟内回帖实质认可**；追帖（7/24）接其 column-based transform 长期解、给出 patch 退役路径、向 assignee 要 PR 绿灯 → [回帖](https://github.com/huggingface/lerobot/issues/2895#issuecomment-5066001307)
- [x] **PR 材料全备**（留档 VLA_Lab `upstream_prep/lerobot_2895/`）：patch A 正式版（基 main a0eb860，缓存正装 `__init__` + 删 try/except + timestamp 查询顺手修）+ 两枚测试（全帧等价 + spy transform 单列契约）；服务器 dev 环境（lerobot main editable）pytest 12/12、pre-commit 全 hook 通过
- [x] **开 PR（8/3 夜，火线过期 2 天补射）**：[PR #4314](https://github.com/huggingface/lerobot/pull/4314)，Rick 网页亲发。预检（8/3 晚）：main 自 a0eb860 走 110 commits 但两目标文件零人碰、`hf_transform_to_torch` 未动；rebase 至 f1efa588 干净、pytest 12/12、pre-commit 全绿。checklist 的 community review 已先行发布于 #3558（Comment 型，三段：hoisting 正确/为何仅 22%/与列视图正交可叠加）。首发 CI 仅 Label PR 绿，主工作流待维护者放行（首次贡献者常态）
- [ ] **守楼**：CI 放行与全绿后手动补勾 checklist；被 @ 尽量 48h 内回；#3558 若先合则按承诺重放 rebase

## 脚本清单

- `scripts/train_smolvla_spatial.sh` — 首次微调用（文档示例配方，留档对照）
- `scripts/train_smolvla_spatial_b64.sh` — **论文对齐配方**（accelerate DDP + bf16）。W2 起默认：workers 32 / prefetch 8 / `COMPILE_MODE=default` / dynamo 开 / NCCL P2P 开（W1 两处 workaround 均已摘除，见环境事实节）；新增 `SCHED_DECAY_STEPS` 旋钮（余弦 decay 长度与总步数对齐，W1 教训固化）
- `scripts/eval_checkpoint_spatial.sh` — checkpoint 评测（n_action_steps / batch / episodes / task_ids 全部参数化）
- `scripts/eval_daemon_spatial.sh` — 评测 daemon：盯 checkpoint 目录自动出曲线（训练/评测分卡解耦）；100k 跑首次全程追平训练（~8 分钟/点）
- `scripts/eval_smolvla_base_spatial.sh` — zero-shot 尝试留档

## 环境事实（服务器侧，2026-07-21 大版本切换后）

- **uv venv `~/vla_lab/.venv`**：Python 3.12.13 / lerobot 0.6.0（钉住）/ torch 2.11.0+cu130 / mujoco 3.8.1 / peft 0.19.1；旧 conda `py312_lerobot` 已随账号切换弃用
- **uv venv `~/vla_lab/.venv-oft`（OFT 官方栈，W3 建）**：Python 3.10.20 / torch 2.2.0+cu121 / transformers 4.40.1 / flash-attn 2.5.5 / robosuite 1.4.1 / **mujoco 3.2.7**（无显式钉版，安装期 resolver 所得；⚠️ 8/04 起为 task5 版本混杂关键事实，勿升级）；装法留档 `~/vla_lab/scripts/oft_install_step1–4.sh`，配套 `LIBERO-oft/` fork
- 驱动 595.71.05 / CUDA 13.2；**IOMMU 关闭后 GPU0+1 P2P 修复（19.7 GB/s）**——W1 的 `NCCL_P2P_DISABLE=1` 与 `TORCHDYNAMO_DISABLE=1` 两个 workaround 均已摘除并探针验证；max-autotune 仍禁用（torch 2.11 inductor sm_80 真 bug，见 W2 矩阵）
- ⚠️ lerobot #3814 补丁 + patch A 都打在 site-packages——重装/升级 lerobot 会同时丢补丁与 409M LIBERO 场景资产，须重打补丁 + 重下资产
- ⚠️ 服务器上跑任何 lerobot 命令前先 `source scripts/env.sh`（HF 缓存/数据集路径重定向；含机器路径，不入库）
- 评测注意：`--eval.batch_size` 在 LIBERO 上无实际加速（环境串行）；`--env.max_parallel_tasks>1` 线程池共享 policy 状态，勿用

## 决策记录

- 2026-07-24：失败标注定谳 Rick 拍板；机制口径自纠——正式评测 `n_action_steps=1` 每步重推理，失败非开环伪影，「eval 调小 n_action_steps」宣判为伪对策
- 2026-07-22：100k 对照定谳 Rick 拍板——对外数字维持 30k/82.0；更长余弦无增益（70.0，−12pp 处噪音边界但至少无增益）；compile 数值与 W1 eager 时代 loss 不再逐位可比，成功率口径不受影响
- 2026-07-21：数据路径 Rick 拍板——patch A + workers 32 + prefetch 8 都上；lerobot 0.6.0 钉住（W1 基线可比）；热数据不搬 SSD，vmtouch 预热替代
- 2026-07-20：对外数字 Rick 拍板——报 **30k checkpoint 的 82.0%**（n=1×100 官方口径）；25k 的 84 不报：同一次评测既选优又报数天然偏高（winner's curse），除非换 seed 复测坐实
- 2026-07-19：重训配方 Rick 拍板——batch 64（32×2 DDP）、30k 步（与 lr 调度对齐）、bf16；训练/评测分卡；best checkpoint = 全量保留周期 checkpoint + daemon 曲线事后选优
- 2026-07-19：评测口径 Rick 拍板——正式数字用 n_action_steps=1 × 100 eps（论文仿真协议）；曲线用 n=10（论文 Table 13 示 n=10≈n=1）
- 2026-07-14：repo 今日启用 git；GitHub 私有仓先行，W7 打磨后转公开；MIT；commit 由 Claude 代打、Rick review
- 2026-07-13：baseline 从 libero_spatial 开刀（论文参考值最高、最易复现），libero_10 最后碰
