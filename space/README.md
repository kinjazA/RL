---
title: Qwen2.5-3B Interview Assistant
emoji: 🤖
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
---

# Qwen2.5-3B Interview Assistant

Gradio demo comparing the same interview question across three project stages:

| Stage | Model |
|---|---|
| Base | `Qwen/Qwen2.5-3B-Instruct` |
| SFT | Base + `Shawnno/qwen2.5-3b-interview-sft-lora` |
| SFT + DPO | Base + `Shawnno/qwen2.5-3b-interview-dpo-lora` |

The DPO adapter was trained from the SFT adapter and already contains the SFT
weights. It is therefore loaded directly on the clean base model, not stacked
on top of the SFT adapter.

The app loads the 3B base model once in 4-bit and switches between the two LoRA
adapters. Requests are serialized to keep adapter switching safe. The generation
entry point uses `@spaces.GPU`, so the app can run on Hugging Face ZeroGPU as
well as a dedicated CUDA Space. CPU mode is only a compatibility fallback.

## Deploy

Create a Gradio Hugging Face Space, select ZeroGPU or a dedicated T4, and place
the contents of this directory at the root of the Space repository. Persistent
storage is recommended for dedicated hardware so model files remain cached
between restarts.

Optional Space variables can override the defaults:

| Variable | Default |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` |
| `SFT_ADAPTER` | `Shawnno/qwen2.5-3b-interview-sft-lora` |
| `DPO_ADAPTER` | `Shawnno/qwen2.5-3b-interview-dpo-lora` |

The public demo intentionally does not display a reward-model score. The old
prototype used an unrelated OpenAssistant reward model, whose absolute scores
were not valid evidence for this project's interview-answer quality. Frozen
64-question acceptance metrics are shown instead.
