# 面试助手微调项目（LLaMA-Factory 管线）

用 **LLaMA-Factory** 微调 `Qwen2.5-3B-Instruct`，做一个面试回答助手。
数据是 **3230 道带完整元数据的面试题**，答案按统一风格（**简单、专业、口语化，像人现场作答**）
用 DeepSeek 批量生成补齐。前端用 Gradio 展示。

> 相比旧版（TRL 手写 SFT→RM→PPO 管线）的改动：
> - 训练框架换成 **LLaMA-Factory**（不再自己写训练代码）
> - 数据换成带元数据的结构化题库（`interview_intent` / `expected_points` 双栏驱动答案生成）
> - 方向务实化：面试回答**主观**，不做难见效的 PPO，先把 **SFT 做扎实**

---

## 目录结构

```text
RL/
  data/
    rlhf_interview_all_questions_merged.csv   原始题库：3230 题 × 12 列（含 150 条金标答案）
    rlhf_answers_filled.csv                   生成答案后的全量数据（3230 行，训练主数据源）
    rlhf_questions.csv                         问题子集/衍生数据
    train.csv  sft_clean.csv  rm_train.csv    旧版遗留数据（弃用，可删）

  space/                                      前端演示（Gradio HF Space）
    app.py                                     主界面：Base/SFT/RLHF 三栏对比 + RM 打分
    colab_demo.ipynb                           Colab 版启动（share=True 出公网链接）
    README.md                                  HF Space 部署配置（sdk: gradio, hardware: T4）
    data/                                      测试爬的数据（可删）

  generate_answers.py                         答案生成工具（DeepSeek API，断点续跑）
  scrape_interview.py                         面试题爬虫（备用；牛客/知乎 JS 反爬抓不了）
  README.md                                   本文件
```

---

## 数据

### 结构（3230 行 × 12 列）

| 列 | 含义 |
|---|---|
| `id` | 唯一编号（ALG_xxx 等） |
| `role` | 岗位：算法工程师 / 数据科学家 / 数据分析师 / 软件开发 / 测试开发 / 产品经理 / 数据工程师 / 通用HR |
| `category` | 类别：机器学习、深度学习、SQL、系统设计、HR 六大类等 |
| `skill` | 具体技能点 |
| `question_type` | 题型：基础概念 / 业务场景案例 / 方案权衡开放题 / 实操任务 / 故障排查等 |
| `difficulty` | 初级 / 中级 / 高级 |
| `seniority` | 目标职级 |
| `question` | 问题正文 |
| `interview_intent` | 考察意图（出题目的） |
| `expected_points` | 期望要点（评分标准） |
| `source_type` | human_seed(100，金标) / self_instruct / evol_instruct |
| `answer` | 答案 |

覆盖：8 岗位 × 300+ 技能点，难度分布 中级 1488 / 高级 1099 / 初级 643。

### 答案风格 spec（金标 = human_seed 的 ALG_001）

```
1. 直接回答，不绕弯、不铺垫
2. 先给核心概念的定义/直觉 → 再说"过高/异常时"的现象（冒号具体展开）→ 最后一句点本质
3. 专业术语用得准，但用大白话解释清楚
4. 语气平实、口语化，像面试现场口头作答；不要"首先/其次/最后"模板、不要 AI 腔
5. 长度约 150-300 字
```

金标示例（偏差-方差）：
> 偏差表示模型的假设与真实规律之间的差距。偏差过高通常说明模型太简单，会出现欠拟合：训练集和验证集效果都比较差，而且两者差距不大。方差表示模型对训练数据波动的敏感程度。方差过高通常说明模型过度记住了训练数据，会出现过拟合：训练集效果很好，但验证集明显变差，换一批数据结果波动也比较大。本质上就是在模型复杂度和泛化能力之间做平衡。

### 答案生成（`generate_answers.py`）

- 模型：DeepSeek `deepseek-chat`；`temperature=0.45, top_p=0.9`（保事实又不失口语）
- 每道题把 `interview_intent` + `expected_points` 注入 prompt 当答案大纲
- 系统提示词内置上述风格 spec + ALG_001/L1L2 两个 few-shot 样本
- 保留源文件已有的 150 条金标答案，只补缺的 ~3080 条；断点续跑

```bash
export LLM_API_KEY=<DeepSeek key>
python generate_answers.py --provider deepseek --limit 5    # 先试 5 条
python generate_answers.py --provider deepseek              # 全量
# 其他模型: --provider qwen | moonshot | ollama(本地 llama)
```

---

## 训练管线（LLaMA-Factory SFT）

> 待做。LLaMA-Factory 版本/配置需按当时最新版核对（本项目搭建时网络受限未核实）。

```
1. 数据转格式
   data/rlhf_answers_filled.csv → LLaMA-Factory 的 alpaca JSON：
   {"instruction": question, "output": answer}（可带 system 列放风格提示）

2. 写训练配置（examples/train_lora 参照）
   base:  Qwen/Qwen2.5-3B-Instruct
   method: lora / qlora（4bit）
   dataset: 转好的 alpaca 数据
   cutoff_len: 1024
   建议: epochs 2-3, lr 2e-4, batch 16, bf16

3. 训练（云 GPU，如 RunPod，非本机——本机无 GPU）
   llamafactory-cli train sft.yaml

4. 评估
   从 3230 题里按 skill 分组抽出 ~5% 当测试集（组隔离，防泄漏）
   生成后人工按风格 spec + 要点覆盖打分
```

---

## 前端展示（`space/`）

- 当前是 **Zephyr 7B 的 RLHF 流程对比 demo**：同一道题 → Base / SFT / RLHF 三栏回答 + RM 打分，展示"微调→对齐"让回答变好的效果
- 部署到 HF Space 需：**T4 显卡 + 持久存储**（HF 现在要求 PRO 或预付费 credits）
- 本机无 GPU 时可用 `colab_demo.ipynb` 在 Colab 免费 T4 上临时跑
- 待办：训练出自己的模型后，把 `space/app.py` 换成加载自己 adapter 的版本

---

## 状态 & 待办

- [ ] 答案生成（后台运行中，~305/3230）
- [ ] 数据转 LLaMA-Factory 格式
- [ ] SFT 训练（云 GPU）
- [ ] 评估（留出测试集 + 人工/强模型打分）
- [ ] 前端接入自己模型
- [ ] 部署 Space（需付费）
