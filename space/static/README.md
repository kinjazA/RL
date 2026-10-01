---
title: Qwen2.5-3B Interview Assistant
emoji: 🎤
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---

# Qwen2.5-3B 面试助手 · SFT + DPO 微调对比

同一道面试题，在 Qwen2.5-3B-Instruct 基座上依次经过 QLoRA SFT 与 DPO 对齐后的回答对比。

| Stage | 模型 |
|---|---|
| Base | `Qwen/Qwen2.5-3B-Instruct` |
| SFT | 基座 + `Shawnno/qwen2.5-3b-interview-sft-lora` |
| SFT + DPO | 基座 + `Shawnno/qwen2.5-3b-interview-dpo-lora` |

DPO adapter 由 SFT adapter 继续训练而来，已包含 SFT 权重，因此直接加载在干净基座上。

本 Space 为静态展示：若干道代表性面试题的三阶段回答已预生成（4-bit、贪心解码），
并附 64 题独立测试集的定量指标。想要现场输入任意问题运行，可打开仓库中的
`colab_demo.ipynb` 在 Colab 免费 T4 上运行。
