# 面试助手微调项目（LLaMA-Factory 管线）

用 **LLaMA-Factory** 在本机（RTX 4060 8GB）微调 `Qwen2.5-3B-Instruct`，做一个面试回答助手。
数据是 **3230 道带完整元数据的面试题**，答案按统一风格（**简单、专业、口语化，像人现场作答**）用 DeepSeek 批量生成补齐。

**✅ 训练已完成**：QLoRA SFT 3 epochs，loss 2.66 → 1.54，风格验证通过。LoRA adapter 已推上 Hugging Face：
**[Shawnno/qwen2.5-3b-interview-sft-lora](https://huggingface.co/Shawnno/qwen2.5-3b-interview-sft-lora)**

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
    sft_train.json                            转好的 alpaca 格式（LLaMA-Factory 直接用）
    dataset_info.json                         LLaMA-Factory 数据集注册文件

  space/                                      前端演示（Gradio）
    app.py                                     主界面：Base/SFT/RLHF 三栏对比 + RM 打分
    colab_demo.ipynb                           Colab 版启动（share=True 出公网链接）
    README.md                                  HF Space 部署配置（sdk: gradio, hardware: T4）

  generate_answers.py                         答案生成工具（DeepSeek API，断点续跑）
  convert_llamafactory.py                     CSV → alpaca JSON 转换工具
  setup_llamafactory.py                       LLaMA-Factory 数据接入一键脚本
  sft_qwen3b.yaml                             QLoRA SFT 训练配置
  verify/
    verify.py                                 加载 adapter 跑真题验证风格
    infer.yaml                                推理配置（指向 HF 上的 adapter）
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
# 其他模型: --provider qwen | moonshot | ollama(本地)
```

---

## 训练管线（LLaMA-Factory SFT）—— 已跑通

### 0. 前置：装 LLaMA-Factory（本机 conda 环境）

```bash
# 用 conda 建 Python 3.11 环境（系统 Python 3.14 太新，bitsandbytes 装不上）
conda create -n llama python=3.11 -y
conda activate llama

# 装 CUDA 版 torch（必须带 cu128，默认 pip 装的是 CPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# 装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e . --no-deps   # 关键：不要用 [torch]，否则会覆盖成 CPU 版 torch
# 再装它锁定的版本范围依赖
pip install "transformers>=4.55,<=5.8.0" "datasets<=4.0.0" "accelerate<=1.11.0" \
  "peft==0.18.1" "gradio<=5.50.0" "trl<=0.24.0" bitsandbytes "tyro<0.9.0" \
  "gradio-client==1.14.0"
```

### 1. 数据接入（`setup_llamafactory.py` 一键完成）

把 `data/sft_train.json` 拷进 LLaMA-Factory，并把 `sft_train` 注册进它的 `dataset_info.json`：

```bash
python setup_llamafactory.py    # 幂等，可重复跑
```

### 2. 训练配置（`sft_qwen3b.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| model_name_or_path | `Qwen/Qwen2.5-3B-Instruct` | 基座 |
| stage / finetuning_type | `sft` / `lora` | SFT + LoRA |
| quantization_bit / method | `4` / `bnb` | QLoRA，8GB 显存刚需 |
| dataset | `sft_train` | 3230 条 |
| template | `qwen` | ChatML |
| cutoff_len | `1024` | ⚠️ 不能设 2048，会 OOM（见下） |
| per_device_train_batch_size | `2` | |
| gradient_accumulation_steps | `4` | 有效 batch 8 |
| learning_rate | `2e-4` | Qwen 官方推荐范围 |
| num_train_epochs | `3` | |
| lr_scheduler_type | `cosine` | |
| lora_rank / alpha | `8` / `16` | |

### 3. 启动训练

```bash
# 国内网络：走 hf-mirror 镜像下载基座
$env:HF_ENDPOINT = "https://hf-mirror.com"
# 确保 llamafactory-cli 用的是 conda 环境的（而非 base）
$env:PATH = "C:\Users\leeze\anaconda3\envs\llama\Scripts;C:\Users\leeze\anaconda3\envs\llama;$env:PATH"

cd C:\Users\leeze\Documents\GitHub\LLaMA-Factory
llamafactory-cli train sft_qwen3b.yaml
```

**训练完产物**：`saves/Qwen2.5-3B-Instruct/lora/sft-cli/` 下的 checkpoint，含 `adapter_model.safetensors`（LoRA 增量权重）+ `training_loss.png`。

### 4. 效果验证（`verify/`）

训练完用 `verify/` 目录下的脚本，从你的题库里挑不同岗位真题测风格。

```bash
# 方法一: 跑内置 4 类真题（算法概念 / 后端GC / 数据分析编程 / 行为面试）
cd LLaMA-Factory
python C:\...\RL\verify\verify.py

# 方法二: 测自定义问题
python C:\...\RL\verify\verify.py "什么是偏差和方差？"

# 方法三: 交互对话（自己慢慢问）
llamafactory-cli chat C:\...\RL\verify\infer.yaml
```

`verify/infer.yaml` 指向 HF 上的 adapter（`Shawnno/qwen2.5-3b-interview-sft-lora`），克隆仓库即可直接用；要测本地 checkpoint 就改成你的 `adapter_name_or_path`。

**验证结果（4 类真题全部风格达标）**：

| 真题 | 效果 |
|---|---|
| 偏差-方差（算法概念） | ✅ 定义准确 → 现象冒号展开 → 点本质，完全对齐 human_seed 风格 |
| JVM GC（后端概念） | ✅ 口语化比喻开场，Minor/Major/Full 区分清楚，CMS/G1/ZGC 逐个讲 |
| Python CSV 排序（数据分析） | ✅ 完整代码 + try/except 异常处理 + 进阶优化思路 |
| 行为面试（开放题） | ✅ STAR 结构完整，有起因/行动/结果/反思 |

对比原始 Qwen-Instruct：开场不再「首先/其次/最后」模板、直接给定义、口语化叙述、收尾一句话点本质——微调成功学到了目标风格。

### 5. 独立评测集（`eval/`）

训练题库的 3230 条样本均已参与本轮 SFT，不能再从中抽题作为最终验收。`eval/sft_test_v1.json` 因此提供 64 条独立人工编写题：8 个岗位各 8 条，包含评分要点但不包含参考答案。

```bash
# 检查候选测试题与训练题的词面近重复，生成 overlap_report_v1.csv
python eval/audit_overlap.py
```

使用该集比较 Base 与 SFT 时，`expected_points` 只允许评分者或裁判读取，绝不能进入模型 prompt。Base 与 SFT 必须使用相同的模板、输出上限和确定性解码参数；完整使用方法见 `eval/README.md`。

`eval/colab_sft_acceptance.ipynb` 可在 Colab 中直接运行 `eval/compare_base_sft.py`。脚本会导出 64 条 Base/SFT 原始回答及长度统计；RM 训练产物可作为可选的成对量化裁判接入，但 RM 分数不能替代独立评分。

### ⚠️ 踩过的坑（重要）

1. **显存 OOM**：`cutoff_len: 2048` 会让 8GB 显存撑爆（激活值太大），跑到 ~160 步就崩。降到 **1024** 即可（数据中位数才 585 字符，仅 3.4% 被截断）。崩溃后 LLaMA-Factory 会从最近 checkpoint **自动续训**。
2. **预处理崩溃**：WebUI 默认 `preprocessing_num_workers: 16`，Windows 多进程 spawn 必挂。降到 **4**。
3. **不要用 Claude/后台任务起长训练**：会话清理会杀训练进程。用 `Start-Process` 起独立进程，或直接开个终端窗口跑。
4. **别让 Windows 睡眠**：笔记本合盖/长时间无操作会杀训练进程。训练时把电源计划设成「高性能」+ 永不睡眠。

---

## 训练结果

| 指标 | 值 |
|---|---|
| 基座 | Qwen/Qwen2.5-3B-Instruct |
| 方法 | QLoRA (4bit) |
| 数据 | 3230 条面试问答 |
| 步数 / epochs | 1212 步 / 3 epochs |
| 显存峰值 | ~3.8GB（8GB 卡安全） |
| 训练耗时 | ~2 小时（RTX 4060） |
| loss | 2.66 → 1.54 |

独立的 64 题 Base vs SFT 验收已完成。SFT 将平均回答长度从 584 字降至 272 字，150-450 字目标区间命中率从 26.6% 提升到 75.0%，自然结束率从 53.1% 提升到 89.1%。完整结论、逐题回答和原始摘要见 [`eval/results/sft_acceptance_v1/`](eval/results/sft_acceptance_v1/)。

验证样例（风格已达标）：
> **Q**: 请解释机器学习中的偏差和方差，它们分别过高时通常会出现什么现象？
> **A**: 偏差是指模型对训练数据的拟合程度，方差是指模型对训练数据的敏感程度。偏差过高时，模型拟合能力差，训练集和验证集误差都高，但两个误差之间的差距小，容易欠拟合；方差过高时，模型拟合能力好，但对训练数据的微小变化很敏感，训练集误差和验证集误差之间的差距大，容易过拟合。

---

## 前端展示（`space/`）

- 当前是 **Zephyr 7B 的 RLHF 流程对比 demo**：同一道题 → Base / SFT / RLHF 三栏回答 + RM 打分，展示"微调→对齐"让回答变好的效果
- 部署到 HF Space 需：**T4 显卡 + 持久存储**（HF 现在要求 PRO 或预付费 credits）
- 本机无 GPU 时可用 `colab_demo.ipynb` 在 Colab 免费 T4 上临时跑
- 待办：训练出自己的模型后，把 `space/app.py` 换成加载自己 adapter 的版本

---

## 状态

- [x] 数据生成（3230 条全部补齐）
- [x] 数据转 LLaMA-Factory 格式
- [x] SFT 训练（本机 RTX 4060，QLoRA）
- [x] 效果验证（4 类真题风格达标）
- [x] LoRA adapter 推上 HF（[Shawnno/qwen2.5-3b-interview-sft-lora](https://huggingface.co/Shawnno/qwen2.5-3b-interview-sft-lora)）
- [x] 独立测试集（64 条，等待 Base vs SFT 批量对比）
- [ ] 前端接入自己模型
- [ ] 部署 Space（需付费）
