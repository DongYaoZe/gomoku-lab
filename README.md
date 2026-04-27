# 幸福五子棋 - 人工智能程序设计课程设计报告

![Python](https://badgen.net/badge/Python/3.8%2B/blue) ![Tkinter](https://badgen.net/badge/GUI/Tkinter/green) ![AI](https://badgen.net/badge/AI/Minimax%20%7C%20MCTS%20%7C%20AlphaZero/red)

[![GitHub stars](https://img.shields.io/github/stars/DongYaoZe/gomoku-lab?style=social)](https://github.com/DongYaoZe/gomoku-lab/stargazers)
[![Fork](https://img.shields.io/github/forks/DongYaoZe/gomoku-lab?style=social)](https://github.com/DongYaoZe/gomoku-lab/network/members)


本项目为《人工智能程序设计》课程的五子棋博弈期末课程设计。五子棋标准 $15 \times 15$ 棋盘状态空间庞大（约 $3^{225}$ 种状态），对搜索算法提出了极大挑战。本项目不仅实现了一个功能完善、界面美观的五子棋对战平台，还集成了从基础贪心策略到极小极大（Minimax）、蒙特卡洛树搜索（MCTS）及深度强化学习（AlphaZero）等多种异构人工智能博弈算法，并探讨了连环冲四绝杀（VCF）与各类启发式剪枝优化技术。

---

## 🚀 提交材料与运行指南

### 1. 运行环境依赖

本项目核心逻辑及图形界面基于 Python 内置库原生开发，确保了高度的可移植性。若需体验深度学习 AI (AlphaZero)，则需额外安装 PyTorch。

- **基础环境**: Python 3.8 或以上版本（内置 `tkinter`，无需额外依赖即可运行绝大部分功能）。
- **进阶环境 (AlphaZero可选)**: `pip install torch`。若不安装，AlphaZero 模块将提示缺失，但不影响其他模式和高级 AI 引擎的正常运行。

### 2. 一键执行命令

请在项目**源代码根目录下**（包含 `main.py` 的目录），打开命令行终端（CMD / PowerShell / Terminal），输入以下命令即可一键运行：

```bash
python main.py
```
> **注**：代码内部完全避免了绝对路径，支持跨平台一键执行。

---

## 📌 选题描述与项目背景

五子棋（Gomoku）是经典的完全信息博弈游戏。由于其规则简单但变化多端，一直是检验人工智能搜索算法和评估函数的优秀平台。
本课题旨在：
1. **构建一个规范的五子棋对战环境**，包含标准的15x15棋盘，并支持竞技级别的“禁手”规则判定（长连、三三、四四）。
2. **实现层次化的 AI 算法梯队**，从初级的启发式规则，到传统的基于 Alpha-Beta 剪枝的 Minimax 搜索，再到融合启发式局面评估的蒙特卡洛树搜索（MCTS）和基于神经网络的 AlphaZero 算法。
3. **提供友好的图形交互界面**，支持玩家对战（PvP）、人机对战（PvE）以及机器对决（EvE），并具备存盘记录、复盘打谱和 VCF（连续冲四胜）算杀等高级辅助功能。

---

## 💡 架构设计思想

本项目采用 **Model-View-Controller (MVC) 架构思想** 进行高度解耦设计：
- **数据层 (Model - `core.py`)**：负责维护棋盘矩阵表示、处理落子状态转移、$\mathcal{O}(1)$ 时间复杂度的胜负检测，以及基于贪婪字符串模式匹配的复杂禁手规则判定。它作为一个纯物理沙盒，对上层透明。
- **视图与控制层 (View & Controller - `gui.py`)**：基于 Tkinter 构建，运用异步非阻塞机制（如 `root.after` 打造平滑闪烁游标）。负责事件轮询、界面渲染与游戏状态机（PvP/PvE/EvE）的调度。
- **算法层 (AI Modules)**：各 AI 模块对外暴露统一的 `get_best_move(game)` 接口。通过高度封装，实现了 AI 思考过程与游戏物理规则的隔离。在多机互搏（EvE）中，更引入了对称性空间旋转变换（Symmetry Breaking）以增加对局混沌性。

---

## 📂 项目文件一览

| 文件 | 角色 |
|------|------|
| `main.py` | **程序入口** — 创建 Tkinter 根窗口，启动主事件循环 |
| `core.py` | **核心逻辑** — 棋盘状态、落子、五子连珠判定、禁手规则 |
| `gui.py` | **图形界面** — 棋盘绘制、点击交互、AI 集成、棋谱保存/回放 |
| `ai.py` | **初级 AI** (BaselineAI) — 贪心策略：赢→堵→随机 |
| `ai_advanced.py` | **高级 AI** (AdvancedAI) — Minimax + Alpha-Beta 剪枝 + 启发式评估 + VCF 强杀 |
| `ai_mcts.py` | **MCTS AI** — 蒙特卡洛树搜索 + UCB1 + 启发式评估替代随机模拟 |
| `alphazero_net.py` | **AlphaZero 神经网络** — PyTorch 残差 CNN，策略头 + 价值头 |
| `alphazero_mcts.py` | **AlphaZero MCTS** — 带神经网络先验概率的 PUCT 树搜索 |
| `ai_alphazero.py` | **AlphaZero AI 封装** — 组合网络 + MCTS，对外暴露 `get_best_move()` |
| `ai_alphazero_train.py` | **AlphaZero 训练管道** — 自我对弈 → 数据增广 → 网络训练 → 保存模型 |

---

## 🧩 代码模块与底层原理剖析

### 1. 核心逻辑裁判引擎 (`core.py`)
- **状态转移**：实现了 `make_move` 和 `undo_move` 支持博弈树快速遍历。
- **$\mathcal{O}(1)$ 胜负判定**：对最后落子点四个方向进行中心辐射扫描。
- **连珠禁手探测**：提取落子点周边长度为 9 的线段序列，通过 `"01111"`, `"011100"` 等字符串特征进行快速模式匹配，实现长连、四四、三三禁手拦截。

### 2. 传统极小极大搜索网络 (`ai_advanced.py`)
本项目中名为“高级/大师”级的 AI。
- **静态评估函数 (Heuristic Evaluation)**：赋予成五 $+100,000$ 绝对杀权，活四 $+10,000$，冲四 $+1,000$ 等评分。利用己方得分减去敌方得分评估整体盘面态势。
- **Alpha-Beta 剪枝优化**：
  - **启发式排序 (Move Ordering)**：对子节点按静态估分降序探索，大幅增强剪枝效果。
  - **记忆化转置表 (Memoization)**：利用 Python 字典缓存已搜索过局面的深度和分值，避免冗余探索。
- **VCF 算杀引擎 (Threat-Space Search)**：针对“连续冲四迫使对手防守”的战术，实现了剥离普通 Minimax 的独立深搜机制。为了避免“自杀式进攻”，算杀前会首先检查并拦截对手的一击必杀点，实现了极具压迫感的人类大师级算杀能力。

### 3. 启发式蒙特卡洛树搜索 (`ai_mcts.py`)
名为“深智”引擎的实现。
- **UCB1 分支抉择**：利用 $UCT = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N_i}{n_i}}$ 平衡探索（Exploration）与利用（Exploitation）。
- **价值函数置换 (Rollout Substitution)**：摒弃了五子棋中由于棋盘过大导致传统完全随机模拟（Random Rollout）效率低下且常平局的缺陷，**改用静态评估函数代替模拟过程**，并通过 Sigmoid 函数 $1 / (1 + e^{-x})$ 将分值反演压缩到 $(0, 1)$ 胜率空间，实现了极高效率的局面探查。

### 4. 深度学习 AlphaZero (`ai_alphazero.py`)
- **神经网络架构**：引入了 DeepMind 的 AlphaZero 架构思想，由 PyTorch 策略价值网络直接输出局面的 Value 和先验落子 Policy。
- **降维搜索**：利用神经网络的先验概率指导 MCTS，极大地缩减了搜索树宽度。模块内含 `ai_alphazero_train.py` 提供自我对弈与迭代训练脚本。

### 5. 用户交互与复盘系统 (`gui.py`)
- **异步渲染引擎**：界面实现了多套皮肤（木制、大理石、水晶）。运用 `root.after` 延时回调函数设计了焦点游标的红框闪烁。
- **棋谱序列化**：自研了标准化对局记录文本结构（例如 `{[C5][黑方][白方][赛果]...;B(J,10);W(K,9)}`），通过正则表达式解析实现对战打谱与逐帧动态回放（Replay）。

---

## 🎮 实现效果展示

### 1. 人机对战全程实录

玩家执黑对阵 AI（高级），被但电脑打败了（😭。

![玩家VS电脑全程](pics/玩家VS电脑全程.gif)

### 2. AI 对弈：大师 VS 高级 & 中级

机器互搏模式（EvE）下，不同等级 AI 之间的精彩对决。

大师（Depth=4）VS 高级（Depth=3）
![大师VS高级](pics/大师VS高级.gif)

大师（Depth=4）VS 中级（Depth=2） 
![大师VS中级](pics/大师VS中级.gif) 

### 3. 禁手规则判定

开启禁手后，黑棋触发三三、四四或长连禁手时系统立即弹窗判负。

![三三禁手动图](pics/33forbidden.gif)

| 三三禁手实例 |
|:---:|
| ![33禁手2](pics/33forbidden2.png) |
| ![33禁手](pics/33forbidden.png) |

| 四四禁手实例 |
|:---:|
| ![44禁手](pics/44forbidden.png) |

| 长连禁手实例 |
|:---:|
| ![长连禁手](pics/long_link_forbidden.png) |


### 4. 复盘打谱与回放

保存棋谱后可通过"打谱"功能打开，以动画逐帧重演整局对弈。

| 回放过程 |
|:---:|
| ![打开回放](pics/打开回放.gif) |
| ![回放动画](pics/回放2.gif) |

### 5. AI 思考过程日志

控制台实时输出各 AI 的决策耗时、搜索估值、缓存命中等信息，便于调试与性能分析。

![命令行日志](pics/cmd_log_print.png)

### 6. 特色功能一览

| AlphaZero 在缺乏训练时的落子，可以看成一幅画 |
|:---:|
| ![MCTS评估](pics/AI_deepmind1.PNG) |
| ![AlphaZero绘画](pics/Alphazero：随机落子绘画.png) |

| 用五子棋写字 |
|:---:|
| ![落子绘画](pics/落子绘画：“可”.png) |

---
## 致谢

本项目的 AlphaZero 实现参考了以下开源项目：

https://github.com/junxiaosong/AlphaZero_Gomoku

基于 PyTorch 的五子棋 AlphaZero 实现，为本项目的策略价值网络与自对弈训练管道提供了重要参考。

---
*本项目为2026《人工智能程序设计》dyz的课程作业。*

*欢迎在GitHub上star我！*
