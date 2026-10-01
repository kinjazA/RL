# 面试助手微调项目

用 **LLaMA-Factory** 微调 `Qwen2.5-3B-Instruct`，做一个**面试回答助手**。
基于 3230 道带元数据的面试题，答案按「简洁、专业、口语化」风格用 DeepSeek 补全，
再走 **SFT → DPO** 两段训练（QLoRA 4-bit，本机 RTX 4060 8GB）。

## 在线 Demo

**https://huggingface.co/spaces/Shawnno/Interview_Assistant**

5 道代表性面试题的 Base / SFT / SFT+DPO 三栏回答对比（免费静态展示，秒开）。
想现场输入任意题，用 `space/colab_demo.ipynb` 在 Colab 免费 T4 上跑 live 版。

## 成果

64 题独立测试集（8 岗位 × 8，冻结），三阶段贪心解码对比：

| 指标 | Base | SFT | **SFT + DPO** |
|---|---|---|---|
| 中位长度（字） | 883 | 345 | **443** |
| 150–450 字命中率 | 1.6% | 84.4% | **53.1%** |
| 自然收尾率 | 100% | 100% | **100%** |
| 强 5-gram 重复率 | 90.6% | 9.4% | **10.9%** |

- **SFT** 把平均回答从 584 字压到 272 字，学成了目标风格（先定义 → 说现象 → 点本质，口语化）
- **DPO** 在保住简洁风格的同时，自然收尾修到 100%、重复率压回接近 SFT 水平

## 模型

| 阶段 | Hugging Face |
|---|---|
| SFT | [Shawnno/qwen2.5-3b-interview-sft-lora](https://huggingface.co/Shawnno/qwen2.5-3b-interview-sft-lora) |
| SFT + DPO | [Shawnno/qwen2.5-3b-interview-dpo-lora](https://huggingface.co/Shawnno/qwen2.5-3b-interview-dpo-lora) |

## 技术方案

- **数据**：3230 题 × 12 列（岗位 / 技能 / 难度 / 考察意图 / 期望要点…），DeepSeek 把「意图 + 要点」当大纲补全答案，保留 150 条人工金标
- **SFT**：QLoRA（4-bit），3 epochs，loss 2.66 → 1.54，显存峰值 ~3.8GB，约 2 小时
- **DPO**：805 组「长度匹配」偏好对（Skywork-Reward 打分），sigmoid，从 SFT LoRA 起训

## 关键问题与解决

1. **DPO 8GB 显存 OOM**：lm_head 在 bf16 下仍输出 fp32 logits，单张峰值约 855MB。补丁降为 bf16 —— `patches/llamafactory-dpo-bf16-logits.patch`
2. **长度偏置**：RM 分数与长度正相关，DPO 会学到「越长越好」。改为只选「长度相当但更差」的负例（长度比 ≤ 1.3）
3. **截断收尾**：`max_new_tokens` 太小会切掉结尾。放大到 600 + 触顶回退到最后一个句号

## 目录结构

```
RL/
├─ data/       数据（题库 → 训练数据）
├─ scripts/    管线脚本（生成答案 / 转格式 / 接入 LLaMA-Factory）
├─ configs/    训练配置（SFT / DPO）
├─ rm/         偏好数据管线（RM 打分 → preference pairs）
├─ eval/       独立 64 题验收
├─ verify/     训练后快速风格验证
├─ space/      前端 demo（Gradio + 静态展示）
└─ patches/    LLaMA-Factory 补丁
```

## 快速复现

```bash
# 1. 装 LLaMA-Factory（conda Python 3.11 + CUDA 版 torch）
# 2. 数据：生成答案 → 转 alpaca → 接入
python scripts/generate_answers.py --provider deepseek
python scripts/convert_llamafactory.py
python scripts/setup_llamafactory.py
# 3. 训练（在 LLaMA-Factory 目录下，DPO 前先打上面的补丁）
llamafactory-cli train ../RL/configs/sft_qwen3b.yaml
llamafactory-cli train ../RL/configs/dpo_qwen3b.yaml
```

更细的环境搭建、超参和验收方法见 [`eval/README.md`](eval/README.md) 和 [`rm/README.md`](rm/README.md)。
