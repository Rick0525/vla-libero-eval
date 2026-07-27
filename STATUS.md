# STATUS（开发期工作文档，W7 转公开前决定翻译或移除）

> 更新规则：进度一变就更新本文件（CLAUDE.md 约定）。

## 当前状态：W2 D7（2026-07-26）

**头条：SmolVLA(0.45B, VLM-init) 微调 libero_spatial 成功率 82.0%**（30k checkpoint，n_action_steps=1 × 100 episodes，官方协议，seed 1000；论文参考值 90，95% CI ±7.5pp）。W2 的 100k 对照跑（70.0%）证明更长调度无增益，**30k/82.0 维持对外数字**。

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
- [~] **三层归因体系（感知/规划/控制）**：R/F/G/T 四实验 7/27 一日收官，全档 `results/attribution_framework_zh.md`。**头条：task5 归因定谳=控制层（抓取几何先验）**——「布局→抓取侧→在手偏心→落点中心→3cm 谓词」因果链三层证据闭环（观察 17/17 + 天然对照 + 干预 n=20/组、Fisher p≈0.0098；同布局换抓取侧成功率 5%→40%，P-R≡O6 布局效应 100% 中介）。副产物：可复现性定谳（三跑 82/84/84 聚合磐石、逐集同栈同 seed 也仅 84/100 非确定）；接管仪器五轮排障（get_state 不含 ctrl/current_action，v5 反解恢复，社区无先例可回馈 LIBERO #16）；接管税结案=源状态质量（T3 双源 10/10 vs 3100 控制 5/10）；**抓姿质量指标化（Exp V）**：Rick 视觉判据（碗前倾 vs 平）→ 碗倾角/腕轴偏角双指标，好/边缘档无重叠（`scripts/bowl_tilt_analysis.py`）。待办：3101 边缘源确认注入、`--deterministic` 模式、标注工具化、task4 实弹、修复实验设计（数据侧，且已定向**内容覆盖而非数量**：10 任务 demo 数 vs 成功率 Spearman≈0——最少 demo 的 stove 任务 35 条 90%，task5 39 条 23%）

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
