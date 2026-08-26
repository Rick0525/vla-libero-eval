# VLA × LIBERO：评测、归因与上游问题发现

[English version / 英文版](README_en.md)

对于 LIBERO-Spatial 任务统一评测了三个 VLA 模型（SmolVLA 0.45B / π0.5 3.3B / OpenVLA-OFT 7.5B），对每一次失败看仿真视频做归因分析。过程中发现并定位了三个上游问题，均已报告。

> **定位**：一次完整的评测实践 + 一套探针工具，不是通用评测框架。各模型跑在各自官方栈上（lerobot 0.6.0 / OFT 官方仓库），本仓库提供的是统一协议、逐失败归因，以及解释数字背后隐藏变量的探针。

## 主表

| 模型 | 参数量 | 推理配置（官方） | 成功率 (n=500) | 95% CI | 已发表参考值 |
|---|---|---|---|---|---|
| OpenVLA-OFT（特训） | 7.5B | action chunk 8，开环 | **97.4**（487/500） | [95.6, 98.5] | 97.6（论文） |
| OpenVLA-OFT（合训） | 7.5B | action chunk 8，开环 | **97.4**（487/500） | [95.6, 98.5] | — |
| π0.5 | ~3.3B | chunk 50，执行 10 | **94.4**（472/500） | [92.0, 96.1] | 97.0 @ n=100 |
| SmolVLA（我方微调） | 0.45B | n_action_steps=1，每步重规划 | **87.4**（437/500） | [84.2, 90.0] | ~90（论文） |

统一协议：10 任务 × 50 集，完全相同的 500 个 benchmark 钉死初始布局，MuJoCo 3.2.7，各模型用自己发布的推理配置。详细的逐任务拆分、checkpoint 来源、与公开参考值对账见 **[results/leaderboard_zh.md](results/leaderboard_zh.md)**。

## 失败机制分析

通过看仿真视频判读失败原因（覆盖率在报告中逐模型披露）：

| 失败家族 | SmolVLA 0.45B | π0.5 ~3.3B | OFT 7.5B |
|---|---|---|---|
| 接触几何（抓取/运送/放置） | 已判读失败全属此族 | 发生率低 | 发生率低，主要失败来源 |
| 目标指代混淆（奔向错的碗） | 未见 | **系统性**混淆 | 单布局孤例（1/500） |

几个有意思的发现：

- **失败可复现性本身就是诊断信号**：π0.5 的 stove ↔ cabinet 混淆是锁死的决策错误——同样 10 个布局在两次独立跑中失败集完全相同（10/10），且模型端端正正地把*错的*碗放上盘子。跑间失败集重合率可以把「锁死的决策错误」和「差几毫米的接触运气」分开。
- **容量降低发生率，但不消灭机制**：偏心抓、双碗同抓在 0.45B 和 7.5B 上都有实证；系统性指代混淆只在 π0.5 上观测到。

完整分析见 leaderboard「失败机制」节和 `results/failure_videos/` 看片台账。

## 发现的三个上游问题

### 1. MuJoCo bugfix 静默改写了考卷

LIBERO task 5 的初始状态存的是黑碗悬在 ramekin 上方约 11 cm 的「前状态」，真正的布局语义外包给了物理引擎，在 `reset()` 内 10 个空转步里落座完成。MuJoCo 3.4.0 修复了 box-box 碰撞距离计算后，碗落座姿态改变，考卷在无人察觉中被改写（如下图所示），导致分数大幅变动：

- SmolVLA task 5：80% → 28%
- OFT 家族 checkpoint：98% → 12%（姊妹 RL 后训练项目）
- π0.5：几乎不受影响

通过同物理 A/B 实验（钉住物理版本、只换重落座 init），100% 归因到初始前提破坏。探针免策略、免 GPU、每版本秒级运行（`scripts/mj_settle_probe.py`）。

**上游报告**：[lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)、[LIBERO#141](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/141#issuecomment-5231993900)、[RLinf#1460](https://github.com/RLinf/RLinf/issues/1460)；单文件探针 [gist](https://gist.github.com/Rick0525/6e6db2d1fe5f4358c980b569b123fde8)



![task5_init_drift_grid_labeled](./results/failure_videos/task5_init_drift_grid_labeled.png)

### 2. 46× 数据加载退化

`datasets` ≥ 4.4 新增的 custom-format 闸门把 lerobot 的 delta-timestamp 查询打回整行读取，导致为了取 50×7 个 float，每样本白解约 100 张 PNG。修复后 200.7 → 4.4 ms/样本（46×），30k 步训练从 4h02m 缩到约 1.9h。

**上游报告**：[lerobot#2895](https://github.com/huggingface/lerobot/issues/2895)，获 Hugging Face `datasets` 负责人回帖认可；补丁与测试在 `patches/`

### 3. 评测链路不能逐位复现

同权重、同 config（md5 核对）、同种子，OFT 两个 checkpoint 各整套重跑一遍，仍有数集换结局（特训臂 7/500、合训臂 6/500；正反翻转相抵后即「局限」中的净差 3/500 与 2/500）。通过逐步哈希五个量（仿真全状态、本体、主相机图、手臂相机图、动作），定位到两个独立的非确定性来源：

1. **EGL 离屏渲染**：首个分叉 50/50 全部落在相机图上，此刻物理状态仍逐位相同
2. **跨集位置效应**：同一集的结局取决于它在 500 集序列中的位置（MuJoCo warmstart 缓存 + 全局随机数流位置，均经干预实验因果验证）

实用结论：评测执行方式的任何变更，例如集序位置（连跑 vs 单跑）、warmstart 处理、随机数流位置等，都会换一批边缘布局的成败，LIBERO 逐任务数字因此天生带 ±几集的摆动。在穷举探测的 task 7 上，累计已有 7/50 = 14% 的布局被某种微扰翻转过结局。

**全案卷**：[results/oft_nondeterminism_case_zh.md](results/oft_nondeterminism_case_zh.md)

## SmolVLA 微调

- **首次按 lerobot 文档示例命令训练：43%**。定谳三重根因：`n_action_steps=50` 打分本身吃掉约 35pp（纯评测期参数）、batch 4 vs 论文 64、lr 调度 30k 步触底致后续 70% 步数近乎空转。
- **论文对齐重训**：全局 batch 64（2×A800 DDP、bf16），30k 步，墙钟 4h02m（打上数据加载补丁后约 1.9h）→ **87.4% @ n=500**，与论文 ~90% @ n=100 的缺口统计不显著（z=0.73, p=0.47）。
- 全报告：[results/smolvla_spatial_finetune_report.md](results/smolvla_spatial_finetune_report.md)

## 复现

```bash
pip install mujoco==3.2.7    # ≥3.4.0 会打破 task 5 的初始前提（见发现 1）

# SmolVLA / π0.5：lerobot-eval（lerobot 0.6.0）+ 主表配置
# OFT：官方 openvla-oft 仓库 eval 脚本，协议未改

# 版本效应探针（免策略、免 GPU、秒级）
python scripts/mj_settle_probe.py

# 钉死布局复测
python scripts/attribution_probe.py --mode rollout --task-id 5 --init-index 3 \
    --model-path <ckpt> --n-action-steps <官方值>
```

## 局限

- 只测 LIBERO-Spatial；Object / Goal / Long 不在范围内。
- π0.5 评的是纯动作模式，论文的分层子任务推理不在本次评测范围。
- 主表数字为单次跑；重跑抖动在有实测处如实披露（OFT 整套重跑净差 3/500 与 2/500）。
- SmolVLA 行是我方按论文配方的微调，非官方 checkpoint。

## 目录结构

```
scripts/   训练与评测脚本 + 探针（落座、分叉、warmstart、RNG、动作日志等）
results/   榜单（双语）· 微调报告 · 归因报告 · 看片台账 · 非确定性案卷
patches/   lerobot 补丁：46× 数据加载修复（#2895）· LIBERO env 复用（#3814）
```

## 协议

MIT
