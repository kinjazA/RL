# Interview Answer Assistant RLHF

Fine-tuning pipeline for an interview-answer assistant based on
`Qwen/Qwen2.5-3B-Instruct`. The current project focuses on the domains covered
by the available data: behavioral/HR, career, product-style interview answers,
ML, data science, software engineering, and general interview-style Q&A.

The project uses QLoRA for single-GPU training and keeps SFT, reward modeling,
and PPO code separated so each stage can be evaluated independently.

## Repository Layout

```text
RL/
  app.py                         Gradio demo
  data/
    train.csv                    Original SFT data
    rm_train.csv                 Existing RM preference data
    sft_train_v2.csv             Curated SFT train split
    sft_eval_v2.csv              Curated SFT eval split
    sft_test_v2.csv              Held-out SFT acceptance split
    sft_data_report_v2.md        Data audit report
  scripts/
    run_sft_v2.py                One-command SFT training launcher
    sft_prepare_data.py          Clean, deduplicate, and add metadata
    sft_data_report.py           Generate SFT data report
    build_sft_splits.py          Group-isolated train/eval/test split
  sft/
    chatml.py                    Shared prompt formatter
    train.py                     QLoRA SFT trainer
    compare.py                   Base vs SFT comparison
    inference.py                 Quick adapter smoke test
  rm/
    gen_reject.py                Existing RM negative generation script
    train_rm.py                  Existing reward model trainer
  ppo/
    train_ppo.py                 Existing PPO trainer
```

## Environment

Install the SFT dependencies:

```bash
pip install -r requirements_sft.txt
```

Training expects a CUDA GPU. The previous SFT runs used a RunPod A40; 16GB+
VRAM should be workable with 4-bit QLoRA depending on batch size.

## SFT v2 Workflow

For the next training round, run from the `RL/` directory:

```bash
python scripts/run_sft_v2.py
```

This command:

1. Cleans and deduplicates `data/train.csv`.
2. Writes `data/sft_clean_v2.csv` and `data/sft_removed_v2.csv`.
3. Generates `data/sft_data_report_v2.md`.
4. Builds group-isolated train/eval/test splits.
5. Verifies assistant-token loss masking.
6. Launches QLoRA SFT training.

Default training settings:

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | QLoRA, 4-bit NF4 |
| LoRA rank / alpha | `16 / 32` |
| Learning rate | `1e-4` |
| Epochs | `2` |
| Max sequence length | `1024` |
| Loss | assistant tokens only |
| NEFTune alpha | `5` |
| SE/DS oversampling | `2x`, train split only |

Alternative conservative run:

```bash
python scripts/run_sft_v2.py \
  --output_dir sft_output_lr5e5_e1 \
  --learning_rate 5e-5 \
  --num_train_epochs 1
```

## Curated SFT Data

Current SFT v2 data was generated from `data/train.csv` with:

- normalized-question cap of 3 rows
- removal of suspicious long-question/short-answer `general_oasst` rows
- group-isolated train/eval/test split by normalized question
- metadata columns: `domain`, `answer_type`, `normalized_question`

Current split sizes:

| Split | Rows | Normalized question groups |
|---|---:|---:|
| Train | 4,355 | 4,264 |
| Eval | 414 | 406 |
| Test | 411 | 406 |

Source distribution after cleaning:

| Source | Rows |
|---|---:|
| `career_qa` | 1,620 |
| `local_interview_qa` | 1,220 |
| `general_alpaca` | 1,160 |
| `ml_interview` | 482 |
| `general_oasst` | 368 |
| `se_interview` | 174 |
| `ds_qa_treasury` | 145 |
| `hr_interview` | 11 |

See `data/sft_data_report_v2.md` for the full audit.

## Stage Results and Notes

### SFT v1

Earlier SFT runs showed clear gains on behavioral/career prompts: the adapter
learned to answer as a candidate in first person and reduced base-model refusal
patterns such as "As an AI...".

Observed issues:

- repeated behavioral/product-style questions made answers feel templated
- technical domains were underrepresented
- some technical answers regressed compared with the base model
- random row splits made eval loss easier to trust than real generalization

### SFT v2 Preparation

The current SFT v2 update addresses the above before retraining:

- shared ChatML prompt helper used by training, inference, comparison, app, and PPO
- assistant-token loss masking retained
- prebuilt group-isolated train/eval/test splits
- duplicate question cap reduced to 3
- learning rate reduced from `2e-4` to `1e-4`
- epochs reduced from `3` to `2`
- SE/DS oversampling reduced from `3x` to `2x`

### SFT v2 Results

**Model:** [Shawnno/RL-sft-adapter](https://huggingface.co/Shawnno/RL-sft-adapter)

| Metric | Value |
|---|---:|
| Eval loss (best) | 0.983 |
| Eval token accuracy | 73.9% |
| Train loss (final) | 1.036 |
| Train token accuracy | 78.1% |
| Epochs | 2 |
| Train samples | 4,622 (oversampled from 4,355) |
| Eval samples | 414 |
| GPU | NVIDIA A40 (44 GB) |
| Training time | ~26 min |

### Reward Model

The existing reward model stage uses pairwise ranking with
`AutoModelForSequenceClassification` and QLoRA.

Previous result:

| Metric | Value |
|---|---:|
| Eval accuracy | 65.1% |
| Eval loss | 0.697 |
| Epochs | 3 |

Analysis note: the current RM data contains many easy negatives, including
truncated or much shorter rejected answers. This can bias the reward model
toward length and completeness rather than factual quality. RM data should be
reworked before relying on PPO results.

### PPO

PPO code is present but should be treated as experimental until the SFT adapter
and reward model are revalidated. The PPO prompt path now uses the same ChatML
helper as SFT to avoid prompt-format drift.

## Evaluation

After training, compare:

```text
base Qwen2.5-3B-Instruct
previous SFT adapter
new SFT adapter
```

Use held-out prompts from `data/sft_test_v2.csv` and manually inspect:

- behavioral/career specificity and first-person consistency
- reduced template wording
- ML/DS/SE factual correctness
- coherence on general prompts
- graceful behavior on out-of-scope prompts

Quick comparison:

```bash
python -m sft.compare --adapter_path sft_output
```

## Demo

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

## References

- [TRL](https://github.com/huggingface/trl)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DPO](https://arxiv.org/abs/2305.18290)
- [PPO](https://arxiv.org/abs/1707.06347)
