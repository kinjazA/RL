# RLHF Interview Answer Assistant

End-to-end RLHF pipeline to align a language model as an **interview answer assistant** — the model learns to give high-quality, specific answers to interview questions across ML, software engineering, product management, finance, HR behavioral, and general career domains.

Built on **Qwen2.5-3B-Instruct** with **QLoRA** (4-bit NF4), runnable on a single GPU with 16GB+ VRAM.

---

## Pipeline Overview

```
                ┌──────────────────────────────────────┐
                │            Phase 1: SFT               │
                │                                      │
                │  Qwen2.5-3B-Instruct                  │
                │       + QLoRA (4-bit NF4)             │
                │       + 5,558 Q&A pairs               │
                │         · Career QA (1,620)            │
                │         · HR Behavioral (1,570)        │
                │         · Alpaca General (1,160)       │
                │         · ML/DL Interview (482)        │
                │         · OpenAssistant (396)          │
                │         · SE Interview (174)           │
                │         · DS Q&A Treasury (145)        │
                │         · HR Interview (11)            │
                │                                      │
                │  Output: Model learns to answer        │
                │  interview questions informatively     │
                └──────────────────┬───────────────────┘
                                   ↓
                ┌──────────────────────────────────────┐
                │         Phase 2: Reward Model         │
                │                                      │
                │  Same base model (Qwen2.5-3B)         │
                │       + QLoRA                         │
                │       + Reward head (Linear: d→1)     │
                │       + 5,558 preference pairs        │
                │         · chosen:  human answer       │
                │         · rejected: 0.5B model (70%)  │
                │           + rule-based (30%)          │
                │                                      │
                │  Loss: Pairwise Ranking               │
                │  L = -log(σ(score_good - score_bad))  │
                └──────────────────┬───────────────────┘
                                   ↓
                ┌──────────────────────────────────────┐
                │          Phase 3: PPO Alignment       │
                │                                      │
                │  4 models loaded (all 4-bit QLoRA):   │
                │  · Policy model  (trainable)          │
                │  · Reference model (frozen, KL anchor)│
                │  · Reward model   (frozen, scoring)   │
                │  · Critic model   (trainable, value)  │
                │                                      │
                │  For each cycle:                      │
                │  1. Policy generates answer           │
                │  2. Reward model scores it            │
                │  3. Reference model computes KL       │
                │  4. Critic estimates expected value   │
                │  5. PPO clip loss updates policy      │
                └──────────────────┬───────────────────┘
                                   ↓
                         Aligned Model
                  (gives specific, well-structured
                   interview answers)
```

---

## Data

### SFT Data: `data/train.csv` (5,558 rows)

| Source | Count | Domain |
|--------|-------|--------|
| career_qa | 1,620 | Job role responsibilities |
| local_interview_qa | 1,570 | HR behavioral (STAR format) |
| general_alpaca | 1,160 | General instruction-following |
| ml_interview | 482 | ML/DL/CV/NLP interviews |
| general_oasst | 396 | Conversational Q&A |
| se_interview | 174 | Software engineering |
| ds_qa_treasury | 145 | Data science fundamentals |
| hr_interview | 11 | HR behavioral |

Format: `question, answer, source`

### RM Data: `data/rm_train.csv` (5,558 pairs)

Format: `prompt, chosen, rejected`

Generated via `rm/gen_reject.py`:
- 70%: Qwen2.5-0.5B-Instruct answers (naturally lower quality)
- 30%: Rule-based degradation (truncation, shuffle, bullet removal)

---

## Project Structure

```
RL/
├── data/
│   └── train.csv                  # SFT training data (5,558 Q&A)
├── rm/
│   ├── gen_reject.py              # Generate rejected answers for RM
│   ├── train_rm.py                # Train reward model
│   └── RM_Training_Colab.ipynb    # One-click Colab notebook
├── SFT_Training_Colab.ipynb       # SFT training notebook
├── build_notebook.py              # Notebook build script
└── README.md
```

---

## Quick Start

### Phase 1: SFT

Open `SFT_Training_Colab.ipynb` in Google Colab (T4 GPU, free tier) → Run All.

Or run locally:
```bash
# SFT is best done via the Colab notebook — local training requires manual setup
# See SFT_Training_Colab.ipynb for the full pipeline
```

### Phase 2: Reward Model

Open `rm/RM_Training_Colab.ipynb` in Colab → Run All. This runs both steps:

**Step 1 — Generate rejected answers:**
```bash
python rm/gen_reject.py --batch_size 4 --model_ratio 0.7
```
Uses Qwen2.5-0.5B (70%) + rule degradation (30%) to create rejected answers.
Supports checkpoint resume on interruption.

**Step 2 — Train reward model:**
```bash
python rm/train_rm.py
```
QLoRA + reward head on Qwen2.5-3B. Saves adapter to `rm/rm_adapter/`.

### Phase 3: PPO (Coming Soon)

---

## Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Base Model** | Qwen2.5-3B-Instruct | Strong instruction-following, fits 16GB VRAM with QLoRA |
| **PEFT** | QLoRA (4-bit NF4) | Efficient: r=16, alpha=32, ~0.5% trainable params |
| **SFT Trainer** | TRL SFTTrainer | ChatML format, max_seq=1024 |
| **RM Trainer** | TRL RewardTrainer | Pairwise ranking loss, AutoModelForSequenceClassification |
| **Hardware** | Single GPU (16GB+ VRAM) | T4, RTX 4060+, A10G |

---

## References

- **TRL**: [HuggingFace Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
