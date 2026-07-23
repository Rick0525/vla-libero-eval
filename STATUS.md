# STATUS（开发期工作文档，W7 转公开前决定翻译或移除）

> 更新规则：进度一变就更新本文件（CLAUDE.md 约定）。

## 当前状态：W1 D7（2026-07-20 凌晨）

**头条：SmolVLA(0.45B, VLM-init) 微调 libero_spatial 成功率 82.0%**（n_action_steps=1 × 100 episodes，官方协议，seed 1000；论文参考值 90，95% CI ±7.5pp）

- [x] 仓库脚手架：git init、MIT、双语 README、.gitignore（07-14）
- [x] smolvla_base zero-shot baseline：**N/A，结构性不可评**，证据见 `results/smolvla_base_zero_shot.md`（07-14）
- [x] 首次微调（lerobot 文档示例配方：batch 4 / 100k 步 / 内联评测 n=50）：跑完但仅 **43%**（07-14 启动，07-19 归因），三重根因：①评测口径 n_action_steps=50（论文 Table 13：该口径砍 ~35pp）②batch 4 vs 论文 64（样本量差 16 倍）③lr 调度 30k 步即触底、后 70% 步数近乎白跑
- [x] 基线重评（n=1×100，官方口径）：ckpt60k=38% / ckpt100k=**43%**（07-20 凌晨）
- [x] **重训（论文对齐配方：全局 batch 64 双卡 DDP / bf16 / 30k 步与 lr 调度对齐 / 无内联评测）**：4h02m 完成，loss 0.434→0.237，**82.0%**（07-20 04:07 训毕，04:47 评毕）
- [x] 成功率曲线（评测 daemon 于独立 GPU 出分，n=10 × 50 eps/点）：5k=52 → 10k=48 → 15k=72 → 20k=68 → 25k=74 → 30k=74
- [x] 最佳 checkpoint 甄别：25k 正式评测 **84.0%** vs 30k 82.0%（n=1×100，差 2 集=统计平手，07-20 05:20 完成）；**对外数字拍板为 30k 的 82.0%**（终点惯例；25k 的 84 未经换 seed 复测有 winner's curse，不报）
- [~] 结果正式化：微调 report 已写入 `results/smolvla_spatial_finetune_report.md`（07-20，含双配方对照/曲线/逐任务表/CI 论证）；**失败归因标注待 Rick**——task5/8 全部 20 段视频已拉到本地 `results/failure_videos/`，标注表 `ANNOTATION_zh.md`（四分类 + 赛前预测已登记，先预测后看片），结论回填 report

## 上游贡献（lerobot）

- [x] **#2895 根因评论已发布（2026-07-23）**：定位 delta 查询 40× 退化的真根因（lerobot `set_transform` 触发 datasets ≥4.4 的 custom-format 闸门，列优先语法静默退化为整行取+全量 PNG 解码），四方案同机基准（现状 165ms / select() 185ms 更慢 / 缓存列视图 0.8ms ≈200×），并解释了此前"升级 datasets 即愈"集体误判的来源（#2549 时代 <4.0 形态的先例）→ [评论链接](https://github.com/huggingface/lerobot/issues/2895#issuecomment-5060174507)。本地补丁 `patches/lerobot-delta-query-column-views.patch`（生产验证 46×/样本，b64 训练 data wait 38%→~1%）。
- [ ] PR：待维护者回应或静默一周后推进（rebase 到 main + 等价性/单列契约测试）。

## 脚本清单

- `scripts/train_smolvla_spatial.sh` — 首次微调用（文档示例配方，留档对照）
- `scripts/train_smolvla_spatial_b64.sh` — **论文对齐配方**（accelerate DDP + bf16；含两处环境 workaround，注释内有据：NCCL_P2P_DISABLE、TORCHDYNAMO_DISABLE）
- `scripts/eval_checkpoint_spatial.sh` — checkpoint 评测（n_action_steps / batch / episodes / task_ids 全部参数化）
- `scripts/eval_daemon_spatial.sh` — 评测 daemon：盯 checkpoint 目录自动出曲线（训练/评测分卡解耦）
- `scripts/eval_smolvla_base_spatial.sh` — zero-shot 尝试留档

## 环境事实（服务器侧，已验证）

- conda `py312_lerobot`：lerobot 0.6.0 / torch 2.11.0+cu126 / mujoco 3.8.1 / peft 0.19.1，EGL headless 渲染冒烟通过
- ⚠️ 服务器上跑任何 lerobot 命令前先 `source scripts/env.sh`（HF 缓存/数据集路径重定向）——env.sh 含机器路径，不入库
- ⚠️ 2026-07-19 服务器驱动 535→595 后：GPU0↔GPU1 PCIe P2P 挂死（NCCL 须 `NCCL_P2P_DISABLE=1`）；Triton/inductor kernel 训练期间歇性 illegal memory access（须 `TORCHDYNAMO_DISABLE=1`，lerobot SmolVLA 有硬编码内部 compile）——均已固化进脚本/env.local.sh，详见 VLA_Lab notes/W1.md 踩坑表
- 评测注意：`--eval.batch_size` 在 LIBERO 上无实际加速（环境串行）；`--env.max_parallel_tasks>1` 线程池共享 policy 状态，勿用

## 决策记录

- 2026-07-20：对外数字 Rick 拍板——报 **30k checkpoint 的 82.0%**（n=1×100 官方口径）；25k 的 84 不报：同一次评测既选优又报数会天然偏高（winner's curse），除非换 seed 复测坐实
- 2026-07-19：重训配方 Rick 拍板——batch 64（32×2 DDP）、30k 步（与 lr 调度对齐，论文明言步数可大减）、bf16；训练/评测分卡（GPU0+1 训、GPU2 评）；best checkpoint = 全量保留周期 checkpoint + daemon 曲线事后选优
- 2026-07-19：评测口径 Rick 拍板——正式数字用 n_action_steps=1 × 100 eps（论文仿真协议）；曲线用 n=10（论文 Table 13 示 n=10≈n=1）
- 2026-07-14：repo 今日启用 git；GitHub 私有仓先行，W7 打磨后转公开；MIT；commit 由 Claude 代打、Rick review
- 2026-07-13：baseline 从 libero_spatial 开刀（论文参考值最高、最易复现），libero_10 最后碰
