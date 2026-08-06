# OFT 零散失败看片：t1–t8 全部 31 段（待 Rick 裁决）

来源：OFT 官方栈（MuJoCo 3.2.7 健康物理，与 v2 主表同秤，无版本混杂）。`run0730` = 7/30 leaderboard 官方跑；`run0802` = 8/02 确定性复核重跑。文件名：`spatial_`=特训臂 / `combined_`=合训臂；`tXeY`=任务 X 第 Y 号布局；`_SUCCESS`=翻转对里的成功对照跑。8/02 拍板「剩余失败全拉待看」的兑现（t7/t9 已于 8/02 单独裁决完毕，档在 `oft_t7/`、`oft_t9/`）。

**判读问题**：
1. **有无任何一例指代混淆**（拿错碗/放错目标）？t7/t9 已定谳全无——若这 31 段也无，则「OFT 全失败谱零指代混淆」完整闭环，跨模型全景（混淆系 π0.5 特有）终审落定。
2. 无混淆则逐段归家族：**H2 抓取**（抓空/偏心/碰倒）/ **H3 放置**（目标边缘、谓词不亮）/ **运送掉落**（t7 同款瞬时松爪）/ 其他。
3. **翻转对**四组：成败是否同 t7 翻转对一样穿过「同一行为内部」（都松爪、差在是否掉碗），还是行为定性不同？
4. **t5 e22/e33 跨臂反向**（leaderboard 局限里降级待复核的那条观察）：e22 合训两跑皆败而特训过、e33 特训两跑皆败而合训过——看片找两臂在这两个布局上的行为差异线索。

任务从句备忘：t1 next to the ramekin · t2 from table center · t3 on the cookie box · t4 in the top drawer · t5 on the ramekin · t6 next to the cookie box · t8 next to the plate

## A. 稳定失败对（两跑同败 = 弱扰动下布局锁定，两段对照看机制是否复现）

| 臂 | 任务/布局 | 文件（0730 / 0802） | 机制（Rick 裁决） | 两跑机制同款？ | 备注 |
|---|---|---|---|---|---|
| 合训 | t1 e47 | `combined_run0730_t1e47_ep98` / `combined_run0802_t1e47_ep98` | | | |
| 合训 | t4 e18 | `combined_run0730_t4e18_ep219` / `combined_run0802_t4e18_ep219` | | | |
| 合训 | t4 e19 | `combined_run0730_t4e19_ep220` / `combined_run0802_t4e19_ep220` | | | |
| 合训 | t5 e22 | `combined_run0730_t5e22_ep273` / `combined_run0802_t5e22_ep273` | | | ⚠️ 跨臂反向案：特训此布局过 |
| 合训 | t8 e31 | `combined_run0730_t8e31_ep432` / `combined_run0802_t8e31_ep432` | | | |
| 特训 | t3 e16 | `spatial_run0730_t3e16_ep167` / `spatial_run0802_t3e16_ep167` | | | |
| 特训 | t4 e20 | `spatial_run0730_t4e20_ep221` / `spatial_run0802_t4e20_ep221` | | | |
| 特训 | t4 e29 | `spatial_run0730_t4e29_ep230` / `spatial_run0802_t4e29_ep230` | | | |
| 特训 | t5 e33 | `spatial_run0730_t5e33_ep284` / `spatial_run0802_t5e33_ep284` | | | ⚠️ 跨臂反向案：合训此布局过 |

## B. 翻转对（一败一成，`_SUCCESS` 为成功对照；重点看成败分岔点）

| 臂 | 任务/布局 | 败跑 | 成跑（对照） | 失败机制 | 分岔点（同一行为内部？） |
|---|---|---|---|---|---|
| 合训 | t5 e42 | `combined_run0730_t5e42_ep293` | `combined_run0802_t5e42_ep293_SUCCESS` | | |
| 合训 | t5 e48 | `combined_run0802_t5e48_ep299` | `combined_run0730_t5e48_ep299_SUCCESS` | | |
| 特训 | t5 e8 | `spatial_run0802_t5e8_ep259` | `spatial_run0730_t5e8_ep259_SUCCESS` | | |
| 特训 | t5 e39 | `spatial_run0730_t5e39_ep290` | `spatial_run0802_t5e39_ep290_SUCCESS` | | |

## C. 单跑失败（另一跑成功但未拉片；弱扰动下的边缘布局）

| 臂 | 任务/布局 | 文件 | 机制（Rick 裁决） | 备注 |
|---|---|---|---|---|
| 合训 | t2 e49 | `combined_run0730_t2e49_ep150` | | |
| 合训 | t6 e30 | `combined_run0730_t6e30_ep331` | | 与特训同布局各败一次 ↓ |
| 特训 | t4 e32 | `spatial_run0730_t4e32_ep233` | | |
| 特训 | t6 e30 | `spatial_run0730_t6e30_ep331` | | 与合训同布局各败一次 ↑ |
| 特训 | t8 e36 | `spatial_run0730_t8e36_ep437` | | |

计数自检：A 组 9 对 18 段 + B 组 4 对 8 段（含 4 成功对照）+ C 组 5 段 = 31 段 ✓
