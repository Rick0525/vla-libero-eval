# STATUS（开发期工作文档，W7 转公开前决定翻译或移除）

> 更新规则：进度一变就更新本文件（CLAUDE.md 约定）。

## 当前状态：W1 D2（2026-07-14）

- [x] 仓库脚手架：git init、MIT、双语 README、.gitignore（2026-07-14）
- [ ] SmolVLA（smolvla_base，微调前）LIBERO-Spatial baseline 评测
- [ ] SmolVLA LoRA 微调（libero_spatial）+ 第一条成功率曲线

## 环境事实（服务器侧，已验证）

- conda `py312_lerobot`：lerobot 0.6.0 / torch 2.11.0+cu126 / mujoco 3.8.1 / peft 0.19.1，EGL headless 渲染冒烟通过
- 权重与数据集已全部就位并逐文件验收（2026-07-14 晨）
- ⚠️ 服务器上跑任何 lerobot 命令前先 `source scripts/env.sh`（HF 缓存/数据集路径重定向）——env.sh 含机器路径，不入库

## 决策记录

- 2026-07-14：repo 今日启用 git；GitHub 私有仓先行，W7 打磨后转公开；MIT；commit 由 Claude 代打、Rick review
- 2026-07-13：baseline 从 libero_spatial 开刀（论文参考值最高、最易复现），libero_10 最后碰
