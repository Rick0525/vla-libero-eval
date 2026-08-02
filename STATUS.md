# STATUS（开发期工作文档，W7 转公开前决定翻译或移除）

> 更新规则：进度一变就更新本文件（CLAUDE.md 约定）。

## 当前状态：W3 D5（2026-07-31）

**头条：LIBERO-Spatial leaderboard v1 开牌（[results/leaderboard.md](results/leaderboard.md)，500 集/模型、同布局集）**：OFT 特训 **97.4** = 合训 **97.4**（合训税 0）＞ π0.5 **93.8** ＞ SmolVLA（我方微调）**83.8**。task5 容量阶梯 28→86→96%；init3 毒布局阶梯 0/13→4/10→5/10→1/1；π0.5 「复现差」定谳为不存在（官方 97.0 系 n=100 且只看前 10 布局，z=1.26 p=0.21）；π0.5 全部 29 个失败集看片定谳=目标混淆（t7/t9 布局锁定）+ 抓取失败（t5/t4 采样波动）两机制。

（历史头条：SmolVLA 30k/82.0 @n=100，现由 83.8 @n=500 取代，CI ±7.5→±3.2pp。）

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
- [~] **三层归因体系（感知/规划/控制）**：R/F/G/T 四实验 7/27 一日收官，全档 `results/attribution_framework_zh.md`。**头条：task5 归因定谳=控制层（抓取几何先验）**——「布局→抓取侧→在手偏心→落点中心→3cm 谓词」因果链三层证据闭环（观察 17/17 + 天然对照 + 干预 n=20/组、Fisher p≈0.0098；同布局换抓取侧成功率 5%→40%，P-R≡O6 布局效应 100% 中介）。副产物：可复现性定谳（三跑 82/84/84 聚合磐石、逐集同栈同 seed 也仅 84/100 非确定）；接管仪器五轮排障（get_state 不含 ctrl/current_action，v5 反解恢复，社区无先例可回馈 LIBERO #16）；接管税结案=源状态质量（T3 双源 10/10 vs 3100 控制 5/10）；**抓姿质量指标化（Exp V）**：碗倾角/腕轴偏角双指标样本内分档干净，但 3101 预注册注入 10/10 **样本外证伪**（其自然落点 1.92cm 实为好档——「源姿=落点档位」四连中，档位的状态空间编码仍是悬案，新指标准入门槛=分开 3100/3101 对）。待办：Rick 盲评 3101 源视频、`--deterministic` 模式、标注工具化、task4 实弹、修复实验设计（数据侧，且已定向**内容覆盖而非数量**：10 任务 demo 数 vs 成功率 Spearman≈0——最少 demo 的 stove 任务 35 条 90%，task5 39 条 23%）

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
- [x] **OFT 确定性证伪 + 榜单修订（8/02，Rick 起题「把失败集重跑一遍」）**：同权重/config（md5 核对）/同调用整套重跑——特训（同卡 GPU2）487→490 翻 7 集、合训（GPU1，卡因素未单独排除）487→489 翻 6 集 → **「OFT 是确定性的」撤回**、v1「两臂 t5 失败不重叠=权重差异翻盘」撤回（同臂自己就换失败集 {33,39}→{8,33}；仅 t5 集 22/33 跨臂反向 n=2 存活降级待复核）；**前向逐位探针**（固定合成观测,进程内 ×10 + 跨进程哈希全同）洗清策略头 → 非确定性住闭环 env/渲染链路（W2 lerobot 栈同 seed 非确定同款,两栈共同点 robosuite/MuJoCo+EGL）；合训税=0 升级 2×2 稳健账（跑内 ≤±1、跨臂 ≤±2）。**t7/t9 全失败看片（Rick 裁决,两臂×两跑）**：OFT 无一例指代混淆,全属接触几何家族 → 对称目标混淆定谳 π0.5(3.3B) 特有;「偏心抓→盘缘」家族 0.45/3.3/7.5B 三容量点集齐;成败边界穿过同一行为内部（成功跑也瞬时松爪,差在是否掉碗）。**指纹框架修正**：重合率以扰动强度为标尺（π0.5 采样=强扰动下锁定才是布局决定;OFT 数值噪声=弱扰动,接触失败也能锁定）。**松爪-块边界假说带动作日志检验（oft_action_trace.py）**：字面证伪——运送松爪系物理滑移非指令（成功 trace 指令零翻转）;意外①失败 trace 指令反转向块边界聚集（12 次瞬时张开 7 次在新 chunk 首步,「上 chunk 末 CLOSE→新 chunk 首 OPEN」反转对 2/3 出现,重抓场景伪影）;意外②单集协议下双跑锁定失败集 e14/e36 翻成功——跨集 env 上下文敏感（候选 warmstart 携带）,**单集探针≠全套协议**新协议教训。leaderboard_zh 八处修订（英文待 Rick 审毕）,裁决档 failure_videos/oft_t7|t9/,净化脚本 oft_bitstability_probe.py + oft_action_trace.py 入库。遗留待 Rick 拍板：剩余零散失败看片范围（建议只补 t4/t5）、首分叉探针、合训臂 GPU2 复测、头条表是否就近注重跑值

## 上游贡献（lerobot #2895）

- [x] **根因评论发布（7/23）**：定谳 custom-format 闸门机制 + 四方案同机基准（现状 165ms / select() 185ms 更慢 / 列视图 0.8ms ≈200×）+ 集体误判来源考古（#2549 时代 <4.0 形态的先例）→ [评论](https://github.com/huggingface/lerobot/issues/2895#issuecomment-5060174507)
- [x] **lhoestq（datasets 负责人）16 分钟内回帖实质认可**；追帖（7/24）接其 column-based transform 长期解、给出 patch 退役路径、向 assignee 要 PR 绿灯 → [回帖](https://github.com/huggingface/lerobot/issues/2895#issuecomment-5066001307)
- [x] **PR 材料全备**（留档 VLA_Lab `upstream_prep/lerobot_2895/`）：patch A 正式版（基 main a0eb860，缓存正装 `__init__` + 删 try/except + timestamp 查询顺手修）+ 两枚测试（全帧等价 + spy transform 单列契约）；服务器 dev 环境（lerobot main editable）pytest 12/12、pre-commit 全 hook 通过
- [ ] **开 PR**：assignee imstevenpmwork 点头即开火；静默至 **8/1** 也开火。checklist 需附一条对他人 open PR 的 review（拟 #3558，已深度分析）

## 脚本清单

- `scripts/train_smolvla_spatial.sh` — 首次微调用（文档示例配方，留档对照）
- `scripts/train_smolvla_spatial_b64.sh` — **论文对齐配方**（accelerate DDP + bf16）。W2 起默认：workers 32 / prefetch 8 / `COMPILE_MODE=default` / dynamo 开 / NCCL P2P 开（W1 两处 workaround 均已摘除，见环境事实节）；新增 `SCHED_DECAY_STEPS` 旋钮（余弦 decay 长度与总步数对齐，W1 教训固化）
- `scripts/eval_checkpoint_spatial.sh` — checkpoint 评测（n_action_steps / batch / episodes / task_ids 全部参数化）
- `scripts/eval_daemon_spatial.sh` — 评测 daemon：盯 checkpoint 目录自动出曲线（训练/评测分卡解耦）；100k 跑首次全程追平训练（~8 分钟/点）
- `scripts/eval_smolvla_base_spatial.sh` — zero-shot 尝试留档

## 环境事实（服务器侧，2026-07-21 大版本切换后）

- **uv venv `~/vla_lab/.venv`**：Python 3.12.13 / lerobot 0.6.0（钉住）/ torch 2.11.0+cu130 / mujoco 3.8.1 / peft 0.19.1；旧 conda `py312_lerobot` 已随账号切换弃用
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
