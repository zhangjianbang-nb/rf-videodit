# RF-VideoDiT — 像素空间流匹配视频生成（本机训练版）

在 `/home/whu/vidgen` 实现的视频生成算法：**rectified flow（流匹配）+ 因子化时空注意力
DiT**，直接在 64×64×16 帧的像素空间建模 UCF-101 动作视频，类别条件生成，支持
classifier-free guidance。

算法细节见 [`DESIGN.md`](DESIGN.md)；训练结果见本文末尾"结果"一节。

## 目录结构

```
vidgen/
├── DESIGN.md              算法设计（中文）
├── prepare_data.py        解压 UCF-101 并构建缓存
├── vidgen/
│   ├── model.py           RF-VideoDiT：3D patchify + 空间/时间注意力 + 3D RoPE + AdaLN-Zero + EMA
│   ├── flow.py            流匹配损失、时间步采样、Euler/Heun ODE 采样器、CFG、reflow 配对
│   ├── data.py            视频解码 → 72×72×16 帧 uint8 memmap 缓存
│   ├── train.py           训练循环（bf16、EMA、断点续训、定期采样预览）
│   ├── sample.py          从 checkpoint 采样
│   ├── evaluate.py        生成 + 与真实留出集算 FVD
│   ├── metrics.py         R(2+1)D-18 特征 + Fréchet 距离
│   ├── reflow.py          可选第二阶段：直线化 ODE（未在本轮执行）
│   └── viz.py             mp4 拼图输出
├── scripts/               setup_env.sh / prepare_data.sh / train.sh / sample.sh / eval.sh
├── data/                  UCF-101.zip、ucf101/（解压）、cache/（memmap 缓存）
└── runs/                  日志、checkpoint、采样预览
```

## 快速开始

```bash
cd /home/whu/vidgen

# 1) 环境（已有 .venv：torch 2.14.0+cu130 + opencv）
scripts/setup_env.sh

# 2) 数据：下载 UCF-101（6.4GiB）→ 解压 → 构建缓存
scripts/prepare_data.sh

# 3) 训练（固定 GPU 1；GPU 0 是 Bonsai 服务，禁止占用）
scripts/train.sh

# 4) 采样 / 评测
scripts/sample.sh runs/base/ckpt_final.pt --classes "BasketballDunk,Surfing,PlayingGuitar" --per-class 3
scripts/eval.sh   runs/base/ckpt_final.pt
```

## 关键约定

- **GPU 1 专用**：所有脚本里 `CUDA_VISIBLE_DEVICES=1`。GPU 0 上的 `bonsai.service`
  （llama-server :8080）与 GPU 1 上的 wan-video（:8091）都不受本训练影响，训练进程
  与它们共存（GPU 1 空闲约 15 GB）。
- 训练是**可断点续训**的：`ckpt_last.pt` 含模型 + EMA + 优化器状态；每 2000 步落盘，
  收到 SIGTERM/SIGINT 也会先存盘再退出；`ckpt_final.pt` 为收尾权重（模型 + EMA）。
- 采样默认用 **EMA 权重**（`--no-ema` 可关）。

## 结果

### 训练

| 项 | 值 |
|---|---|
| 总步数 | 89,029（三段连续训练，累计 343.6 分钟 ≈ 5.7 小时） |
| 等效批大小 | 16（batch 8 × 梯度累积 2） |
| 吞吐 | 68–73 样本/秒（≈ 4.5 步/秒） |
| 模型 | 45.0M 参数，patch (2,4,4)，hidden 384，depth 12 |
| 数据 | UCF-101，过滤后 13,276 条（train 9,510 / val 3,766） |
| loss | 1.33（初始）→ **1.0141（最佳，step 36.4k）** → 1.077（step 88k） |
| 最终权重 | `runs/base/ckpt_best.pt`（step 88,000，尖峰前最佳）、`runs/base/ckpt_final.pt` |

**两次 loss 尖峰**（本机实测，重要教训）：

| 尖峰起点 | 当时学习率 | 峰值 loss | 结果 |
|---|---|---|---|
| step 37,200 | 2.74e-4 | 1.2009 | 从 1.018 回退到 ~1.10，未完全恢复 |
| step 88,400 | 1.53e-4 | 1.1516 | 训练随即收尾 |

两次尖峰发生在**不同学习率**下，说明不是学习率过高所致，而是架构层面的稳定性问题：
本项目 `Attention` **没有 QK-norm**（对 q/k 做 RMSNorm 或钳制注意力对数），这类"单批次
数值尖峰"在 DiT 训练中常见，标准做法就是加 QK-norm。下一步应优先补上。
另外，第一版 checkpoint 保留策略只留最近 2 个编号权重，导致尖峰前的最佳状态（34k–36k）
被删除、无法回滚；现已修正为保留最近 8 个（`--ckpt-keep`）并额外按窗口 loss 保存
`ckpt_best.pt`。

### 生成质量（step 88,000，EMA 权重）

FVD 用 R(2+1)D-18（Kinetics-400）特征，256 条生成样本 vs 256 条真实留出样本：

| 检查点 | 样本数 | 步数 | FVD | 噪声地板（真实 vs 真实） |
|---|---|---|---|---|
| step 30,000 | 64 | 25 | 958.8 | 274.8 |
| step 70,000 | 64 | 25 | 989.7 | 274.8 |
| **step 88,000** | 256 | 50 | **984.3** | 157.9 |

像素统计（生成 vs 真实）：

| 指标 | 生成 | 真实 |
|---|---|---|
| std（对比度） | 0.718 | 0.523 |
| 帧间差（运动量） | 0.824 | 0.140 |
| 饱和像素（钳到 ±1）比例 | 31.9% | 0% |

速度场诊断（在真实片段上测，t=0.2–0.95）：`|v| / |u| ≈ 0.65`（回归到后验均值导致的
收缩，属 MSE 训练的正常行为），`corr(v, u) ≈ 0.41`。

**结论（如实说明）**：这是一个**能跑通、能生成、但质量仍属早期**的检查点。生成片段
的对比度接近真实数据，但时间抖动是真实数据的 5.9 倍、约 1/3 像素饱和——说明模型学到
了部分结构（速度场与目标相关 0.41），但远未收敛。FVD 在 30k→88k 步之间没有改善，
与 loss 平台一致。参照公开工作，64×64 视频 DiT 通常需要 50 万–200 万步，本项目
8.9 万步仅约为其 1/5–1/10；按本机 4.5 步/秒计算，达到 50 万步还需约 30 小时。

要显著提升，建议按优先级：(1) 加 QK-norm 消除尖峰；(2) 延长训练到 30 小时量级；
(3) 引入预训练 VAE 做潜空间建模，把每步算力换到更多 token/更长视频上。

### 推理网页

```bash
scripts/webapp.sh                       # 默认加载 runs/base/ckpt_best.pt
# 浏览器打开 http://127.0.0.1:8093
```

页面支持：101 个动作类别（可搜索）或无条件生成、样本数 1–16、采样步数、CFG 强度、
随机种子、求解器（heun/euler）、帧数；带进度条、结果播放器与历史记录。
后端只用标准库（`vidgen/webapp.py`），单 GPU 串行执行任务。

样例产物：`runs/base/eval_final/generated.mp4`（生成 64 条）、`runs/base/eval_final/real.mp4`
（真实对照 64 条）、训练过程预览 `runs/base/samples/step_*.mp4`。
