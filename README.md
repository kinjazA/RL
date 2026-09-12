# 面试助手微调项目（LLaMA-Factory 管线）

用 **LLaMA-Factory** 在本机（RTX 4060 8GB）微调 `Qwen2.5-3B-Instruct`，做一个面试回答助手。
数据是 **3230 道带完整元数据的面试题**，答案按统一风格（**简单、专业、口语化，像人现场作答**）用 DeepSeek 批量生成补齐。

**✅ 训练已完成**：

1. **SFT**：QLoRA 3 epochs，loss 2.66 → 1.54，风格验证通过。LoRA adapter 已推上 Hugging Face：
   **[Shawnno/qwen2.5-3b-interview-sft-lora](https://huggingface.co/Shawnno/qwen2.5-3b-interview-sft-lora)**
2. **DPO**：在 SFT LoRA 基础上继续 sigmoid DPO（805 组**长度匹配**偏好对），1 epoch，train_loss 0.20。
   产物在 `LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/dpo-cli-v2/`。

**三方验收（Base / SFT / SFT+DPO，64 题贪心解码）**：DPO 保住了 SFT 的简洁风格，同时修掉了旧 DPO 的两个问题——
natural ending **82.8% → 100%**，5-gram 重复率 **25.0% → 10.9%**（回到 SFT 的 9.4% 水平）。完整过程见文末 [DPO 阶段](#dpo-阶段2026-09-12-完成)。

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

  rm/                                         偏好数据管线（RM 打分 → preference pairs）
    build_preference_data.py                  构造 chosen/rejected 对（含长度匹配，见文末）
    artifacts/full_v1/                        805 组 pair + 质量报告 + 审计表
      preference_pairs_llamafactory.json      LLaMA-Factory DPO 数据（805 组）

  eval/                                       独立 64 题验收
    compare_base_sft.py                       Base / SFT / SFT+DPO 三路对比脚本
    sft_test_v1.json                          64 条独立测试题（8 岗位 × 8）
    audit_overlap.py                          测试集 vs 训练集词面查重
    results/                                  历次验收结果
      sft_acceptance_v1/                      Base vs SFT
      dpo_acceptance_v1/                      旧 DPO（有长度回归，留档对比）
      dpo_acceptance_v2/                      修复长度偏置后 DPO
      dpo_acceptance_v3/                      最终（v2 + natural-ending 修复）

  patches/
    llamafactory-dpo-bf16-logits.patch        修 DPO OOM 的 LLaMA-Factory 补丁（**必打**，见文末）

  space/                                      前端演示（Gradio）
    app.py                                     主界面：Base/SFT/DPO 三栏对比 + RM 打分
    colab_demo.ipynb                           Colab 版启动（share=True 出公网链接）
    README.md                                  HF Space 部署配置（sdk: gradio, hardware: T4）

  generate_answers.py                         答案生成工具（DeepSeek API，断点续跑）
  convert_llamafactory.py                     CSV → alpaca JSON 转换工具
  setup_llamafactory.py                       LLaMA-Factory 数据接入一键脚本（SFT + DPO 两个数据集）
  sft_qwen3b.yaml                             QLoRA SFT 训练配置
  dpo_qwen3b.yaml                             DPO 训练配置（从 SFT LoRA 起跑）
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

- 同一道题 → Base / SFT / DPO 三栏回答 + RM 打分，展示“微调 → 对齐”让回答变好的效果
- `app.py` 的 SFT 栏已加载本项目 adapter（`Shawnno/qwen2.5-3b-interview-sft-lora`）；**DPO 栏目前仍是占位文案**，待接入 `dpo-cli-v2`
- 生成侧已加 natural-ending 兜底（`max_new_tokens=600` + 触顶回退到最后一个句号），保证回答不会说半句就断
- 部署到 HF Space 需：**T4 显卡 + 持久存储**（HF 现在要求 PRO 或预付费 credits）
- 本机无 GPU 时可用 `colab_demo.ipynb` 在 Colab 免费 T4 上临时跑

---

## 状态

- [x] 数据生成（3230 条全部补齐）
- [x] 数据转 LLaMA-Factory 格式
- [x] SFT 训练（本机 RTX 4060，QLoRA）
- [x] 效果验证（4 类真题风格达标）
- [x] LoRA adapter 推上 HF（[Shawnno/qwen2.5-3b-interview-sft-lora](https://huggingface.co/Shawnno/qwen2.5-3b-interview-sft-lora)）
- [x] 独立测试集（64 条，用于 Base / SFT / SFT+DPO 三路验收）
- [x] RM 偏好数据管线（17251 条候选评分 → 805 组**长度匹配** pair）
- [x] 修偏好数据长度偏置 + 重训 DPO（`dpo-cli-v2`）
- [x] 修 DPO 训练 OOM（bf16 logits 补丁，见文末）
- [x] 修 natural ending（解码上限 + 兜底回退，见文末）
- [x] 三路验收（Base vs SFT vs SFT+DPO，natural ending **100%**）
- [ ] 前端接入自己模型（`space/app.py` 的 DPO 栏位目前仍是占位）
- [ ] 部署 Space（需付费）

---

## DPO 阶段（2026-09-12 完成）

### 1. 偏好数据：消除长度偏置

**问题**：v1 的 1210 组 pair 里 `chosen` 100% 取 greedy SFT reference，而 RM 分数与长度正相关，DPO 因此学到“更长更好”，把 SFT 压下来的长度又拉回去。

**修复**（`rm/build_preference_data.py`）：新增 `--max_len_ratio 1.3`，要求同组内 `rejected` 与 `chosen` 长度比 ≤ 1.3，即只有“长度相当但更差”的答案才能当负例。效果：1210 → **805 组**，另有 405 组因找不到长度匹配的负例被丢弃（记录在 `data_quality_report.json` 的 `skipped.no_length_matched_negative`）。

| | v1 | v2 |
|---|---|---|
| pair 数 | 1210 | 805 |
| chosen/rejected 长度比 | 未约束（69% chosen 更长） | ≤ 1.3（中位 1.13） |

pair 构造规则其余不变：原始 SFT reference 作为 `chosen`；同题中 RM 分更低、通过基础质量过滤、分差在 `[0.5, 8.0]` 的候选作为 `rejected`。

### 2. 修 DPO 训练 OOM（关键，必打补丁）

**现象**：用 805 组重训时，sigmoid DPO 在**第 1 步就 OOM**；而 v1 的 1210 组当初却跑完了 152 步。配置、显存、数据长度分布都相同，所以不是“数据变长”导致的。

**根因**（逐层定位）：
LLaMA-Factory 在 bf16 训练下，lm_head 仍输出 **fp32 logits**。`[2, seq, 151936]` 的 logits 单张 fp32 最大约 **855MB**；sigmoid DPO 每步要算 policy + reference 两次前传，且 policy 的 logits 要留到反向传播。更关键的是 PyTorch 缓存分配器的 `reserved` 会随梯度累积**单调增长、从不回落**，实测涨到 **13354 MiB（超过 8GB 物理显存）**——显存碎片化，最终在 `log_softmax` 申请连续块时崩掉。v1 只是碎片没炸（非确定性），并非配置更好。

**修复**：把 logits 降为 bf16，峰值直接减半。补丁见 [`patches/llamafactory-dpo-bf16-logits.patch`](patches/llamafactory-dpo-bf16-logits.patch)，改 LLaMA-Factory 的 `src/llamafactory/train/dpo/trainer.py` 一行：

```diff
-        all_logits = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
+        all_logits = model(**batch, return_dict=True, use_cache=False).logits.to(torch.bfloat16)
```

```bash
# 应用补丁（在 LLaMA-Factory 目录下）
cd LLaMA-Factory
git apply ../RL/patches/llamafactory-dpo-bf16-logits.patch
```

> ⚠️ 不要用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 兜底——该选项在 **Windows 上不支持**，会被静默忽略（日志里会出现 `expandable_segments not supported on this platform` 警告）。

**结果**：打完补丁后 805 组重训一次跑完，**101 步 / 16.4 分钟，无 OOM**。

### 3. DPO 重训（`dpo-cli-v2`）

```bash
cd C:\Users\leeze\Documents\GitHub\LLaMA-Factory
llamafactory-cli train C:\Users\leeze\Documents\GitHub\RL\dpo_qwen3b.yaml
```

`dpo_qwen3b.yaml` 关键配置：

| 参数 | 值 | 说明 |
|---|---|---|
| `adapter_name_or_path` | `saves/Qwen2.5-3B-Instruct/lora/sft-cli` | 从 SFT LoRA 继续 |
| `pref_loss` | `sigmoid` | 标准 DPO；LoRA 时 `ref_model=None`，用 `disable_adapter()` 复用同一模型当参考，不额外加载基座 |
| `dataset` | `interview_preference_pairs` | 805 组长度匹配 pair |
| `cutoff_len` | `704` | 实测 prompt+answer 最大 655 tokens，704 全覆盖 |
| `optim` | `adamw_torch_fused` | 与 SFT 训练一致 |
| `learning_rate` | `1e-5` | DPO 学习率远小于 SFT |
| `num_train_epochs` | `1` | |
| `per_device_train_batch_size` | `1` × acc 8 | 8GB 保守；DPO 同时前传 chosen+rejected，压力 > SFT |

训练结果：**101 步 / 1 epoch / 16.4 分钟，train_loss 0.205，rewards/accuracies 0.99**。产物在 `LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/dpo-cli-v2/`。

### 4. 修 natural ending（解码侧，零重训）

**问题**：v2 验收时 DPO 的 natural ending 只有 85.9%。逐题排查发现：**未收尾的 9 题全部正好卡在 `max_new_tokens=384` 上限**（tokens=384~385），没有一题是模型自己“不会收尾”——纯粹是解码上限切掉了最后半句。SFT 之所以 100%，只是因为它写得短（mean 359 字），从没撞过上限。

**修复**（`eval/compare_base_sft.py` 与 `space/app.py` 同步）：
1. `max_new_tokens` 384 → **600**，给模型留出写完的空间；
2. 新增 `ensure_natural_ending()` 兜底：若生成触顶且末字不是 `。！？…`，回退到最后一个句号。

实测：修完后 **0/64 题触顶**（原 9/64 全部自然收尾，token 数落在 387~453），兜底逻辑根本没触发——病根就是上限太小。补出来的内容是每题的**结论句**（如“阈值策略必须跟业务成本挂钩”），不是水词。

### 5. 三路验收最终结果（2026-09-12）

冻结 64 题、贪心解码、`max_new_tokens=600`。完整数据见 [`eval/results/dpo_acceptance_v3/`](eval/results/dpo_acceptance_v3/)。

| 指标 | Base | SFT | **SFT+DPO（最终）** | 旧 DPO (v1) |
|---|---|---|---|---|
| median_chars | 883.0 | 345.0 | **443.0** | 477.0 |
| p90_chars | 1069.1 | 478.5 | **570.5** | 564.0 |
| pct_150_450 (%) | 1.6 | 84.4 | **53.1** | 42.2 |
| pct_le_550 (%) | 1.6 | 96.9 | **84.4** | 78.1 |
| natural_ending (%) | 100.0 | 100.0 | **100.0** | 82.8 |
| strong 5-gram repeat (%) | 90.6 | 9.4 | **10.9** | 25.0 |
| extreme 5-gram repeat (%) | 71.9 | 0.0 | **1.6** | 0.0 |

**结论**：

- ✅ **重复率修复**：strong 5-gram 重复 25.0% → **10.9%**，基本回到 SFT 的 9.4%——长度匹配消除了“越写越长、自我重复”的退化。
- ✅ **natural ending 修复**：82.8% → **100%**（解码上限 + 兜底，零重训）。
- ⚠️ **长度仍有残留**：median 443 vs SFT 345，DPO 仍比 SFT 长约 100 字。这是当前版本的已知取舍——多出来的是更完整的论证和结论句，不是重复。
- 注：Base 在 `max_new_tokens=600` 下更长（median 883）是因为它本来就爱啰嗦，抬高上限后更明显；这反而让 demo 里“SFT 简洁 vs 原版啰嗦”的对比更清楚。

### 后续可选方向

- [ ] 进一步压 DPO 长度：收紧 `max_len_ratio`，或在偏好对里加长度惩罚项
- [ ] `space/app.py` 的 DPO 栏位接入 `dpo-cli-v2`（目前仍是占位文案）
- [ ] 部署 HF Space（需付费）
