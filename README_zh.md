# VLA 统一评测框架

[English version / 英文版](README.md)

面向开源 VLA（视觉-语言-动作）模型的 LIBERO 统一评测框架——标准化 leaderboard、失败三层归因（感知/规划/控制）、轨迹可视化、一键复现。

> 🚧 **开发中**（2026-07-14 启动）。计划覆盖模型：SmolVLA、OpenVLA-OFT、π0/π0.5、GR00T。第一个里程碑：SmolVLA 微调前的 LIBERO-Spatial baseline。

## 为什么还需要一个评测

各论文发布的 VLA 数字之间几乎不可比：episode 数、随机种子、最大步数、相机设置都不一致，社区复现结果与论文报告也可能相差悬殊。本框架用同一套固定协议评测所有模型，并在成功率这个标量之外，提供逐 episode 的失败归因。

## 目录结构

```
scripts/   # 评测与复现脚本
results/   # leaderboard 表格与曲线（只放小文件）
```

内容随项目推进补充。

## 协议

MIT
