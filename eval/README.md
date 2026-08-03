# SFT Independent Evaluation Set

`sft_test_v1.json` 是现有 SFT 模型的冻结独立测试集，不能被加入任何 SFT、RM 或偏好数据训练。

## 构建原则

- 64 条人工编写的新题，覆盖当前项目的 8 个岗位，每个岗位 8 条。
- 每个岗位包含基础、场景、排障或方案权衡题，难度覆盖初级、中级和高级。
- 不增加 finance 等岗位外行业数据；所有题目均在现有岗位能力范围内。
- `expected_points` 是评分量表，不得放入模型 prompt 或作为模型可见上下文。
- 本测试集只用于 Base vs SFT 的最终对比。开发解码脚本、选择训练 epoch 和调提示词应另用验证集。

## 质量控制

运行以下命令检查候选测试题与 3230 条训练题的词面相似度：

```powershell
python eval/audit_overlap.py
```

脚本将生成 `eval/overlap_report_v1.csv`。它是初筛工具，不代表语义独立性证明：所有被标记的题目必须人工审阅，尤其是同一技术主题但表述不同的题目。冻结后不得根据 SFT 输出修改测试题或评分要点。

## 后续使用

对 Base 与 SFT 使用相同的模型模板、`do_sample=false`、输出上限和问题顺序。每次对比至少保存：模型和 adapter 版本、完整 prompt、生成参数、原始回答、自动统计及人工评分。评分维度为要点覆盖、事实正确性、简洁度、口语化、重复/模板化和岗位匹配度。

### Colab 对比

`colab_sft_acceptance.ipynb` 可直接上传到 Google Colab 运行。它调用 `compare_base_sft.py`，生成以下产物：

- `comparisons.csv`：Base/SFT 原始回答、prompt、题目元数据和评分要点。
- `summary.json` / `summary.md`：长度统计、模型版本、解码参数，以及可选 RM 统计。

默认命令只生成 Base/SFT 回答：

```bash
python eval/compare_base_sft.py \
  --base_model Qwen/Qwen2.5-3B \
  --sft_adapter Shawnno/qwen2.5-3b-interview-sft-lora \
  --output_dir eval/results/sft_acceptance_v1
```

若使用 RM，必须同时给出 RM 的实际基座或完整模型、可选 LoRA adapter，以及 RM 训练时完全相同的输入拼接格式：

```bash
python eval/compare_base_sft.py \
  --base_model Qwen/Qwen2.5-3B \
  --sft_adapter Shawnno/qwen2.5-3b-interview-sft-lora \
  --rm_model <RM_BASE_OR_FULL_MODEL> \
  --rm_adapter <RM_ADAPTER_IF_ANY> \
  --rm_input_format question_answer \
  --output_dir eval/results/sft_acceptance_with_rm
```

RM 的 `mean_delta` 和 `sft_win_rate` 只能作为成对偏好信号。它们不能单独证明答案正确或更适合面试，必须与评分要点的人工/独立裁判评分共同使用。
