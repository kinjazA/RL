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
│   ├── train.csv                  # SFT training data (5,558 Q&A)
│   └── rm_train.csv               # RM preference pairs (5,558)
├── rm/
│   ├── gen_reject.py              # Generate rejected answers for RM
│   ├── train_rm.py                # Train reward model (RunPod A40)
│   └── rm_adapter/                # Trained LoRA weights (gitignored, 115MB)
├── sft/                           # SFT training scripts
├── sft_output/                    # SFT LoRA weights (gitignored, 115MB)
├── app.py                         # Gradio demo app (SFT + RM side-by-side)
├── requirements.txt               # Dependencies for HF Spaces / local
└── README.md
```

---

## Quick Start

### Phase 1: SFT

**Training results** (RunPod A40, 43 min):
| Metric | Start | End |
|--------|-------|-----|
| Train Loss | 3.07 | 0.55 |
| Best Eval Loss | - | 0.85 (step 800) |

**SFT Adapter:** [Shawnno/RL-sft-adapter](https://huggingface.co/Shawnno/RL-sft-adapter)

Run command:
```bash
python -m sft.train --csv_path data/train.csv --output_dir sft_output
```

### Phase 2: Reward Model

**Generate rejected answers:**
```bash
python rm/gen_reject.py --batch_size 4 --model_ratio 0.7
```
Uses Qwen2.5-0.5B (70%) + rule degradation (30%) to create rejected answers.
Supports checkpoint resume on interruption.

**Train reward model:**
```bash
python rm/train_rm.py
```
QLoRA + reward head on Qwen2.5-3B. Designed for RunPod A40 (46GB VRAM) or similar GPU.

**Training results** (RunPod A40, ~2 hr):
| Metric | Value |
|--------|-------|
| Eval Accuracy | 65.1% |
| Train Loss (end) | 0.133 |
| Eval Loss | 0.697 |
| Epochs | 3 |

**RM Adapter:** [Shawnno/RL-rm-adapter](https://huggingface.co/Shawnno/RL-rm-adapter)

Adapter saved to `rm/rm_adapter/` (~115MB LoRA weights).

### Phase 3: PPO (Coming Soon)

---

## V2 Improvements (2026-07)

### Base vs SFT Evaluation

A side-by-side evaluation compared the base Qwen2.5-3B-Instruct against the SFT fine-tuned adapter
(QLoRA r=16, 3 epochs) on domain-representative interview questions from the trained domains
(ML/DL, SE, DS, HR, Career). Both models use the same 4-bit quantization — only the LoRA adapter is toggled on/off.

| Domain | Question | Base Model | SFT Model | Verdict |
|--------|----------|-----------|-----------|---------|
| ML | Explain gradient descent as a beginner | Mountain-hiking analogy, intuitive | Textbook definition, dry | **Base wins** — SFT lost the "beginner" framing |
| ML | What is overfitting? | Detailed explanation + symptoms | Condensed 5-point prevention list | **Near-verbatim from training data** — SFT overfit |
| HR | Time you failed? | ❌ Refused ("As an AI...") | ✅ First-person, E-commerce story | **SFT wins** — learned role-play |
| HR | Why leave current job? | ❌ Refused again | ✅ Interview persona with motivation | **SFT wins** — consistent persona |
| Career | Data Scientist day-to-day? | Informative bullet list | Conversational but less dense | Draw |
| SE | Linked list vs array? | ✅ Accurate (O(1) access, contiguous memory) | ❌ Factual error: O(log n) search | **SFT hallucinated** — base more reliable |
| Finance | Time value of money? | ✅ Correct explanation | ❌ Empty word-salad | **SFT failed on unseen domain** |
| PM | Feature prioritization? | Structured framework answer | Started OK, ended with off-topic question | **SFT drifted** |

### Key Findings

1. **HR/Career domains (57% of training data)** — Strong gains: SFT model learned to role-play as a candidate
   with first-person examples, overcoming the base model's "As an AI" refusal pattern.

2. **ML/SE/DS technical domains (15%)** — Mixed results: near-verbatim memorization when question matches
   training data; factual errors introduced for low-sample domains (SE: 174 samples → O(log n) hallucination).

3. **Untrained domains (Finance, PM: 0%)** — Catastrophic degradation: SFT outputs become circular word-salad
   or drift off-topic, significantly worse than the base model.

4. **Root cause** — Loss was computed on the full ChatML sequence (user prompt + assistant answer),
   wasting ~50% of training capacity on memorizing the prompt instead of learning to answer.

### V2 Training Improvements

| # | Change | Files | Expected Impact |
|---|--------|-------|-----------------|
| P0 | **Loss masking** — `DataCollatorForCompletionOnlyLM` restricts loss to assistant tokens only | `train.py` | Less memorization, better generalization to unseen phrasings |
| P1 | **NEFTune noise** (alpha=5) — embedding-level perturbation reduces overfitting | `train.py`, `config.py` | Smoother loss curve, less verbatim reproduction |
| P2 | **System prompt** — every training example now includes a role-defining system message | `dataset_utils.py`, `inference.py` | Consistent interview persona, no refusal drift |
| P3 | **Domain oversampling** — SE (174) and DS (145) duplicated 3× in training set | `dataset_utils.py`, `config.py` | Mitigates factual errors from low-sample domains |
| P4 | **`check_seq_lengths()`** — utility to verify truncation ratio before training | `dataset_utils.py` | Prevents silent answer truncation |

To run the v2 training:

```bash
# 1. Check if MAX_SEQ_LENGTH needs adjustment
python -c "from sft.dataset_utils import check_seq_lengths; check_seq_lengths('data/train.csv')"

# 2. Train
python -m sft.train --csv_path data/train.csv --output_dir sft_output
```

**Comparison notebook**: [`Compare_Base_vs_SFT.ipynb`](Compare_Base_vs_SFT.ipynb) — run in Colab (T4 GPU)
to reproduce the base vs SFT comparison with the current or v2 adapter.

---

## Demo App

Launch the Gradio app locally:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` — enter an interview question, the **left panel** shows the model's answer and the **right panel** shows the reward model's quality score with a visual gauge.

### Deploy to Hugging Face Spaces

1. Push this repo to GitHub
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
3. Choose **Gradio** SDK, point to your repo
4. **Hardware**: pick **T4 GPU** (recommended) or **CPU** (slower, no RM scoring on CPU)
5. HF Spaces auto-installs `requirements.txt` and launches `app.py`

| Space Hardware | Speed | RM Scoring |
|---------------|-------|------------|
| **T4 GPU** (paid) | Fast (~2-3s / answer) | Yes |
| **CPU** (free) | Slow (~15-30s / answer) | No (OOM risk) |

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
