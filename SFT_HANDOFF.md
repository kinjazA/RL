# SFT Handoff Plan

This document defines the reproducible SFT-only workflow for the next training
round. The scope is intentionally limited to domains already represented in the
current data: behavioral/HR, career, ML, DS, SE, and general interview-style QA.

## Data Owner Tasks

Run these commands from the `RL/` directory before pushing the data version:

```bash
python scripts/sft_prepare_data.py
python scripts/sft_data_report.py
python scripts/build_sft_splits.py
```

Generated files:

```text
data/sft_clean_v2.csv
data/sft_removed_v2.csv
data/sft_data_report_v2.md
data/sft_train_v2.csv
data/sft_eval_v2.csv
data/sft_test_v2.csv
data/sft_split_manifest_v2.json
```

Review before training:

- `data/sft_removed_v2.csv`: confirm removed rows are acceptable.
- `data/sft_data_report_v2.md`: confirm source/domain distribution.
- `data/sft_split_manifest_v2.json`: confirm row counts and split settings.

Current default cleaning decisions:

- Cap each normalized question at 3 rows.
- Remove suspicious `general_oasst` rows where question is very long and answer is short.
- Keep technical-domain rows unless they match generic length/noise rules. Technical factual review is still manual.

## Trainer Tasks

Before training:

One-command training:

```bash
python scripts/run_sft_v2.py
```

This prepares data, builds splits, verifies the SFT collator, and starts training.

Conservative defaults are in `sft/config.py`:

Default run settings:

| Setting | Value |
|---|---:|
| `max_per_question` | 3 |
| `learning_rate` | `1e-4` |
| `num_train_epochs` | `2` |
| SE/DS oversampling | `2x`, train split only |
| Loss masking | assistant tokens only |
| NEFTune alpha | `5` |

If the first run overfits or copies training answers too strongly, retry with:

```bash
python scripts/run_sft_v2.py \
  --output_dir sft_output_lr5e5_e1 \
  --learning_rate 5e-5 \
  --num_train_epochs 1
```

## Trainer Deliverables

Return all of the following:

```text
sft_output/
trainer_state.json
training_args.json
adapter_config.json
train/eval logs
best checkpoint step
GPU type and training duration
git commit hash used for training
```

Do not train from a dirty or undocumented data version.

## Acceptance Owner Tasks

Evaluate using the same ChatML system prompt as training. The helper lives in:

```text
sft/chatml.py
```

Compare at least:

```text
base Qwen2.5-3B-Instruct
previous SFT adapter
new SFT adapter
```

Use held-out prompts from:

```text
data/sft_test_v2.csv
```

Suggested Colab comparison command:

```bash
python -m sft.compare \
  --adapter_path Shawnno/RL-sft-adapter \
  --question_set both \
  --samples_per_source 2
```

Outputs are saved under:

```text
compare_outputs/base_vs_sft.csv
compare_outputs/base_vs_sft.md
```

By default, CSV sampling uses target interview sources only. Use
`--csv_sources all` if you want to include `general_alpaca` and `general_oasst`.

Acceptance focus:

- Behavioral/career answers should be specific, first-person, and less templated.
- ML/DS/SE answers must not regress into obvious factual errors.
- General prompts should remain coherent and on-topic.
- Out-of-scope domains are not target domains, but the model should avoid word salad.
