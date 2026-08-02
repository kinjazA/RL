---
title: RLHF Pipeline Demo — Zephyr 7B
emoji: 🦙
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
hardware:
  accelerator: T4
---
# RLHF Pipeline Demo — Zephyr 7B

Shows the same prompt answered by the three stages of an RLHF pipeline,
then scores all three answers with a reward model.

| Stage | HF model | What it is |
|---|---|---|
| Base (pre-SFT) | `mistralai/Mistral-7B-v0.1` | before any fine-tuning |
| SFT | `HuggingFaceH4/zephyr-7b-sft-full` | after supervised fine-tuning (UltraChat) |
| RLHF (DPO) | `HuggingFaceH4/zephyr-7b-beta` | after DPO alignment (UltraFeedback) |
| Reward model | `OpenAssistant/reward-model-deberta-v3-large-v2` | scores each answer |

This demonstrates **SFT → DPO** alignment on the Zephyr family. The reward
model is a general-purpose public RM (the one used to train Zephyr itself
was not released standalone); swap `RM_MODEL` in `app.py` to use another.

## Requirements

- **GPU Space required** (`T4` minimum; the three 7B models are loaded
  sequentially in 4-bit to fit 16GB VRAM).
- **Persistent storage must be enabled** (Space settings → Persistent
  Storage) so the ~45GB of models download only once.

## Deploy

1. `huggingface-cli repo create rlhf-pipeline-demo --type space`
2. Upload `app.py`, `requirements.txt`, and this `README.md` to the repo.
3. Space settings → Hardware: **T4** (paid) or **zero-gpu** (free, queues).
   Persistent storage: **ON**.
4. Wait for the build, then open the Space.

**First run:** pressing *Generate* downloads and loads the models (several
minutes). Later runs are fast.

## Customize

To demo a different model family, edit the `MODELS` / `RM_MODEL` dict at the
top of `app.py`. `MODEL_FORMAT` controls per-model prompt formatting
(`chatml` for instruction-tuned models, `plain` for base models).
