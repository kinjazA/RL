# Interview Answer Assistant RLHF

Fine-tuning pipeline for an interview-answer assistant based on
`Qwen/Qwen2.5-3B-Instruct`. Domains covered: behavioral/HR, career,
product-style interview answers, ML, data science, software engineering,
and general interview-style Q&A.

Uses QLoRA for single-GPU training. SFT, reward modeling, and PPO stages
are separated so each can be evaluated independently.

---

## Repository Layout

```text
RL/
  app.py                         Gradio demo
  requirements.txt               Demo / general deps
  requirements_sft.txt           SFT + RM + PPO deps

  data/
    train.csv                    Raw SFT data (5,558 rows, 8 sources)
    rm_train.csv                 RM preference pairs (prompt, chosen, rejected)

    sft_clean.csv                After Phase 1.2: cleaned + deduplicated (5,202 rows)
    sft_removed.csv              Rows removed during dedup + reasons

    sft_train_v3.csv             Phase 1.3: training split
    sft_eval_v3.csv              Phase 1.3: eval split
    sft_test_v3.csv              Phase 1.3: held-out test split
    sft_split_manifest_v3.json   Split metadata & sizes

  scripts/
    sft_audit_raw.py             Phase 1.1: audit raw train.csv
    sft_clean_data.py            Phase 1.2: content-level cleaning (HTML, whitespace)
    sft_prepare_data.py          Phase 1.3: row filtering, dedup, metadata
    sft_data_report.py           Generate data audit report
    build_sft_splits.py          Phase 1.4: group-isolated train/eval/test splits
    run_sft_v2.py                End-to-end SFT launcher (v2 — deprecated)

  sft/
    __init__.py
    config.py                    All training hyperparameters
    chatml.py                    Shared ChatML prompt formatter
    train.py                     QLoRA SFT trainer
    compare.py                   Base vs SFT comparison script
    inference.py                 Quick adapter smoke test
    model_utils.py               Model loading (4-bit QLoRA)
    dataset_utils.py             Data loading + oversampling
    checks/verify_collator.py    Verify assistant-token loss masking

  rm/
    gen_reject.py                Phase 3.1: generate RM preference pairs
    train_rm.py                  Phase 3.2: train reward model

  ppo/
    train_ppo.py                 Phase 4: PPO alignment training
```

---

## Phase 0: Environment Setup

### Requirements

- Python 3.10+
- CUDA GPU (16GB+ VRAM for SFT/RM, 24GB+ for PPO)
- Previous runs used: RunPod A40 (46GB), NVIDIA T4 (16GB)

### Install Dependencies

```bash
# For training (SFT + RM + PPO)
pip install -r requirements_sft.txt

# For demo only
pip install -r requirements.txt
```

### Verify Environment

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')
"
```

Expected output: `PyTorch 2.6.0+cu124  CUDA 12.4  GPU: NVIDIA A40` (or similar).

> **Known version constraints (2026-07):**
> - `torch >= 2.6.0` required for newer CUDA drivers.
> - `trl == 0.15.2` — 1.x removed `DataCollatorForCompletionOnlyLM`.
> - `transformers == 4.48.3`, `peft == 0.14.0` — stable with the above.

---

## Phase 1: Data Preparation

**Goal:** Produce clean, deduplicated, group-isolated train/eval/test splits
from raw `data/train.csv`.

**Pipeline order:**
```
1.1 Audit  →  1.2 Clean + Dedup  →  1.3 Build splits
```

All scripts run from the `RL/` directory. Each step is idempotent: re-running
overwrites its outputs with the same result.

### 1.1 Raw Data Audit (read-only)

```bash
python scripts/sft_audit_raw.py \
  --input data/train.csv \
  --out_dir data \
  --sample_n 10
```

**What it does:**
- Counts rows per source, answer length distributions.
- Detects content issues: HTML tags, markdown, code fences, URLs, AI refusals.
- Identifies repeated normalized-question groups.
- Flags suspected reversed question/answer pairs (long question, short answer).
- Writes results to terminal (no separate report file).

**Expected output:**
```
Source distribution:
career_qa             1620
local_interview_qa    1570
general_alpaca        1160
ml_interview           482
general_oasst          396
se_interview           174
ds_qa_treasury         145
hr_interview            11

Answer content issue counts:
markdown_lists      238
whitespace_issue     85
html_tag             24
url                  23
code_fences          21
ai_refusal            5
```

**Key findings from current `train.csv` (5,558 rows):**

| Finding | Scope | Severity |
|---|---|---|
| `ds_qa_treasury` has HTML / markdown / formula images in answers | 145 rows (entire source) | High — format pollution, trains model to output HTML |
| `local_interview_qa` has 6 questions each repeated 36× with distinct answers | 216 rows | Medium — inflates weight of few behavioral questions |
| `general_oasst` has 21 suspected question/answer reversal rows | 21 rows | Medium — wrong direction training signal |
| 6 rows with "As an AI..." etc. refusal patterns | 6 rows | Low — filtered in next step |

### 1.2 Content Cleaning + Dedup

```bash
python scripts/sft_clean_data.py \
  --input data/train.csv \
  --output data/sft_clean.csv \
  --removed_output data/sft_removed.csv \
  --max_per_question 3
```

**What it does (content cleaning + dedup in one pass):**

**Content cleaning:**

| Cleaner | Action | Target |
|---|---|---|
| `clean_html()` | Strip `<strong>`, `<img>`, `<code>`, `<p>`, `<br>` and all other HTML tags. Convert `&amp;`, `&lt;`, `&nbsp;`, `&#8203;` etc. to plain text. | `ds_qa_treasury` answers |
| `clean_markdown()` | Replace `![formula](http://...)` with `(formula)`. Strip ``` code fences (keep content). Unwrap `[text](url)` → `text`. | `ds_qa_treasury` formula images |
| `normalize_whitespace()` | Collapse 3+ newlines → 2. Remove trailing spaces on lines. Collapse runs of spaces/tabs. | All sources |

**Deduplication:**

| Filter | Rule | Rationale |
|---|---|---|
| Exact duplicate | Same (question, answer) pair | Redundant, no signal gain |
| Question cap | Keep at most `--max_per_question` rows per normalized question, preferring longer/more diverse answers | Prevents overfitting to the 6 questions that each appeared 36× in `local_interview_qa` |
| AI refusal | Remove answers starting with "As an AI...", "I cannot...", "Sorry, but I..." | Wrong training signal — model should answer as interview candidate |

**What is NOT done:**
- Rows are never removed for content issues alone (HTML/markdown is cleaned in-place).
- Question text is never truncated or rewritten.
- Source labels are never changed.

**Input / Output:**

| File | Rows | Description |
|---|---|---|
| `data/train.csv` (in) | 5,558 | Raw data with HTML/markdown/whitespace artifacts |
| `data/sft_clean.csv` (out) | ~5,200 | Cleaned + deduplicated. Columns: `question`, `answer`, `source`, `domain`, `answer_type`, `normalized_question` |
| `data/sft_removed.csv` (out) | ~356 | Rows removed + `removal_reason` column |

**Expected output:**
```
Done.
  Content:  HTML 23->0  MD img 2->0  fences 21->0  WS 85->0
  Dedup:    exact_dup=0  over_cap=500 rows in 50 groups  ai_refusal=6
  Result:   5,558 -> 5,202 rows (356 removed)
```

**Verify the cleaning:**

```bash
# Check that no HTML tags remain in answers
python -c "
import pandas as pd, re
df = pd.read_csv('data/sft_clean.csv', encoding='utf-8-sig')
tag = re.compile(r'</?[a-zA-Z][^>]*>')
hits = df['answer'].map(lambda t: bool(tag.search(str(t)))).sum()
print(f'HTML tags remaining: {hits}')  # must be 0
"

# Spot-check ds_qa_treasury for format artifacts
python -c "
import pandas as pd
df = pd.read_csv('data/sft_clean.csv', encoding='utf-8-sig')
ds = df[df['source'] == 'ds_qa_treasury']
print(f'ds_qa_treasury rows: {len(ds)}')
# Check for any remaining issues
for _, r in ds.sample(3).iterrows():
    a = r['answer'][:200]
    print(f'Q: {r[\"question\"][:80]}')
    print(f'A: {a}')
    print('---')
"
```

### 1.3 Split Building (Group-Isolated)

```bash
python scripts/build_sft_splits.py \
  --input data/sft_clean.csv \
  --out_dir data \
  --prefix sft \
  --suffix v3 \
  --eval_frac 0.08 \
  --test_frac 0.08 \
  --seed 42
```

**What it does:**
- Groups rows by `normalized_question`.
- Within each domain, shuffles groups and assigns whole groups to train/eval/test.
- **No question group appears in more than one split.** This is critical:
  if the same question (even with different answers) leaked across splits,
  eval loss would be artificially low.
- Writes three CSVs + a manifest.

**Group-isolation guarantee:**
The script verifies that no normalized question crosses splits:
```python
leaked = df.groupby("normalized_question")["split"].nunique()
leaked = leaked[leaked > 1]
if len(leaked):
    raise RuntimeError(f"Found {len(leaked)} normalized questions crossing splits.")
```

**Expected output:**
```
Wrote splits:
  train rows= 4361 groups= 4001 path=data/sft_train_v3.csv
  eval  rows=  421 groups=  407 path=data/sft_eval_v3.csv
  test  rows=  418 groups=  404 path=data/sft_test_v3.csv
Manifest: data/sft_split_manifest_v3.json
```

**Verify the splits:**

```bash
# Check no question leakage across splits
python -c "
import pandas as pd, re
def norm(t):
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]+','',str(t).lower().strip()))

dfs = {}
for split in ['train','eval','test']:
    dfs[split] = pd.read_csv(f'data/sft_{split}_v3.csv', encoding='utf-8-sig')
    dfs[split]['nq'] = dfs[split]['question'].map(norm)

train_nq = set(dfs['train']['nq'])
eval_nq = set(dfs['eval']['nq'])
test_nq = set(dfs['test']['nq'])

print(f'Train nq in eval: {len(train_nq & eval_nq)}')  # must be 0
print(f'Train nq in test: {len(train_nq & test_nq)}')  # must be 0
print(f'Eval nq in test:  {len(eval_nq & test_nq)}')   # must be 0
print(f'Total unique nq:  {len(train_nq | eval_nq | test_nq)}')
"
```

### 1.4 Verify Assistant-Token Loss Masking

```bash
python -m sft.checks.verify_collator --csv_path data/sft_train_v3.csv --n_samples 5
```

**What it does:**
- Loads 5 samples from the training split.
- Formats them as full ChatML prompts.
- Tokenizes and checks that the `DataCollatorForCompletionOnlyLM` correctly masks
  all tokens **except** the assistant's answer.

**Expected output:**
Each sample shows the token IDs and which ones receive loss. Assistant tokens
should have non-zero loss, all other tokens (system, user, formatting) should
have `-100` (ignored).

---

## Phase 2: SFT Training

### 2.1 Configuration

All training parameters live in [sft/config.py](sft/config.py). **Update these before training:**

```python
# In sft/config.py — v3 settings (drop NEFTune, drop oversampling)

NEFTUNE_NOISE_ALPHA = 0              # was 5 — caused repetition in v2
OVERSAMPLE_SOURCES = []              # was ["se_interview", "ds_qa_treasury"] — over-repetition
OVERSAMPLE_FACTOR = 1                # was 2

# Keep:
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,      # effective batch = 16
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 2,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
}
MAX_SEQ_LENGTH = 1024
```

**v2 → v3 changes and why:**

| Setting | v2 | v3 | Reason |
|---|---|---|---|
| `NEFTUNE_NOISE_ALPHA` | 5 | 0 | Embedding noise on small dataset caused severe repetition in outputs |
| `OVERSAMPLE_SOURCES` | `["se_interview", "ds_qa_treasury"]` | `[]` | 2× oversampling of already-small domains caused keyword dumps and runaway generation |
| `learning_rate` | 1e-4 | **2e-4** | v1 用 2e-4 效果明显，数据现在更干净，回到 v1 的 LR |
| `num_train_epochs` | 2 | 2 | Keep — v2 didn't show overfitting on eval loss |

### 2.2 Launch Training

```bash
python -m sft.train \
  --csv_path data/sft_train_v3.csv \
  --eval_csv_path data/sft_eval_v3.csv \
  --output_dir sft_output_v3
```

**What happens:**
1. Loads train + eval CSVs, formats each row as a full ChatML prompt
   (system prompt + user question + assistant answer).
2. Loads Qwen2.5-3B-Instruct in 4-bit QLoRA.
3. Wraps in `DataCollatorForCompletionOnlyLM` so loss is computed **only**
   on assistant tokens.
4. Trains with the specified config, logging eval loss every 200 steps.
5. Saves the best checkpoint (by eval loss) to `sft_output_v3/`.

**Override config values from CLI:**

```bash
# Conservative run
python -m sft.train \
  --csv_path data/sft_train_v3.csv \
  --eval_csv_path data/sft_eval_v3.csv \
  --output_dir sft_output_v3_conservative \
  --learning_rate 5e-5 \
  --num_train_epochs 1
```

**Expected training time (A40 46GB):** ~24 minutes for 4,374 train rows × 2 epochs.

### 2.3 Training Results (v3)

**Run date:** 2026-07-24 on RunPod A40 (46GB)

| Metric | Value |
|---|---|
| Train rows | 4,374 |
| Eval rows | 416 |
| Epochs | 2 |
| Effective batch size | 16 (4 device × 4 accumulation) |
| Learning rate | 2e-4 |
| Training time | ~24 min (1,439s) |

**Training curves:**

| Step | Epoch | Train Loss | Eval Loss | Train Token Acc | Eval Token Acc |
|---|---|---|---|---|---|
| 200 | 0.73 | ~1.05 | **1.039** | ~57% | 72.1% |
| 400 | 1.46 | ~0.76 | **0.992** | ~79% | 73.6% |
| 546 (final) | 1.99 | **0.941** | — | **81.1%** | — |

**Key observations:**
- Eval loss dropped from 1.039 → 0.992 (no overfitting)
- Token accuracy rose steadily from 57% → 81%
- NEFTune=0 eliminated v2's repetition problem
- No oversampling eliminated v2's keyword-dump problem
- Best checkpoint: step 400 (lowest eval_loss)

**Base vs SFT comparison (T=0, max_new_tokens=512):**

All 15 curated questions show clear base→SFT difference:
- Behavioral: SFT uses first-person STAR format; base gives generic advice
- Career/Product: SFT more specific, candidate-persona language
- ML/DS/SE: Technical accuracy preserved from base model
- General: SFT more detailed, interview-appropriate

Full comparison: `compare_outputs/base_vs_sft_v3_final.md`

### 2.4 Evaluate

**Quick smoke test on local machine:**

```bash
python -m sft.inference --adapter_path sft_output_v3
```

This loads the adapter and answers 3 preset questions (behavioral, technical,
general) to sanity-check.

**Base vs SFT comparison (comprehensive):**

```bash
# Basic comparison on preset questions
python -m sft.compare --adapter_path sft_output_v3

# Full comparison: preset + held-out test set + decoding-mode grid
python -m sft.compare \
  --adapter_path sft_output_v3 \
  --question_set both \
  --samples_per_source 3 \
  --compare_decoding_modes
```

Writes outputs to `compare_outputs/`:
- `comparison_*.csv` — Raw outputs side-by-side
- `base_vs_sft_*.md` — Human-readable comparison report

**What to look for in comparison output:**

| Symptom | Likely cause |
|---|---|
| Repeating the same sentence 2-3× | NEFTune or oversampling on small dataset |
| Outputting HTML tags or markdown | ds_qa_treasury wasn't content-cleaned before training |
| Generic "As a candidate I would..." fillers | Overfitting to overrepresented behavioral questions |
| Factual regression on ML/DS questions | Forgetting from base model due to low domain representation |
| First-person refusal "I can't answer that" | Base-model alignment leak |

---

## Phase 3: Reward Model

The reward model scores (prompt, response) pairs. Training requires preference
data: a chosen (good) answer and a rejected (worse) answer for each prompt.

### 3.1 Generate RM Training Data

```bash
python rm/gen_reject.py \
  --batch_size 4 \
  --model_ratio 0.7 \
  --max_new_tokens 100
```

**What it does:**
1. Reads `data/train.csv` (5,558 questions + answers).
2. For each (question, chosen_answer) pair, generates a **rejected** answer via:
   - **70% model-generated:** Qwen2.5-0.5B-Instruct answers the same question
     with a short, low-quality response (max 100 new tokens, temperature 1.0).
   - **30% rule-based degradation:** structural damage to the chosen answer
     (truncation, mid-drop, shuffle paragraphs, strip bullets).
3. Validates every rejected answer:
   - Must be ≥10 chars and materially different from chosen.
   - Must not be >1.3× the length of chosen (longer could actually be better).
   - Invalid model outputs fall back to `guaranteed_cut()` (cut to first 1/3 of words).
4. Outputs `data/rm_train.csv` with columns: `prompt`, `chosen`, `rejected`.
5. Checkpoints progress to `data/.gen_reject_checkpoint.txt` — safe to resume if interrupted.

**Hardware:** GPU recommended (T4 ~1 hour for 5,558 pairs). CPU possible but slow.

**Data quality considerations (from prior RM analysis):**

| Issue | Impact | Mitigation |
|---|---|---|
| Many rule-based negatives are just truncated answers | RM learns "short = bad", not "wrong = bad" | Model-generated negatives help, but ~30% is the floor for rule-based |
| Chosen answers include ds_qa_treasury HTML artifacts | RM learns to prefer formatted answers | Run Phase 1 content cleaning on `train.csv` **before** generating RM data |
| Behavioral answers dominate (29% career + 28% local_interview) | RM biased toward behavioral style | Accept for now; future: stratified sampling |

**To regenerate RM data with cleaned SFT data as input:**

```bash
# First clean the data
python scripts/sft_clean_data.py --input data/train.csv --output data/sft_clean.csv

# Then point gen_reject.py at the cleaned data
# (edit the INPUT path in rm/gen_reject.py, or pass --input flag if supported)
```

**Verify RM data:**

```bash
python -c "
import csv, random
rows = list(csv.DictReader(open('data/rm_train.csv', encoding='utf-8-sig')))
print(f'Total pairs: {len(rows)}')
# Check length distribution
for r in random.sample(rows, min(10, len(rows))):
    c_len, r_len = len(r['chosen']), len(r['rejected'])
    ratio = r_len / max(1, c_len)
    print(f'  chosen={c_len:4d}  rejected={r_len:4d}  ratio={ratio:.2f}')
"
```

Expected: all ratios < 1.3, no empty rejected, no identical chosen/rejected.

### 3.2 Train Reward Model

```bash
python rm/train_rm.py
```

**Architecture:**
- Base: Qwen2.5-3B-Instruct (4-bit QLoRA)
- Head: Linear layer on top of last hidden state → scalar reward
- Loss: Pairwise ranking loss `-log(sigmoid(reward_chosen - reward_rejected))`

**Training config (hardcoded in `rm/train_rm.py`):**
- `per_device_train_batch_size`: 4
- `num_train_epochs`: 3
- `learning_rate`: 2e-4
- `max_length`: 512
- Train/eval split: 90/10

**Expected output:** `rm/rm_adapter/` (LoRA weights, ~115MB).

**Previous result (v1):**

| Metric | Value |
|---|---|
| Eval accuracy | 65.1% |
| Eval loss | 0.697 |
| Epochs | 3 |

> **Analysis note:** The 65.1% eval accuracy is barely above random chance (50%).
> Prior analysis identified that many rejected answers are simply truncated
> versions of the chosen — the RM may be learning "longer = better" rather
> than "correct = better." RM data should be re-generated after content cleaning
> for meaningful PPO results.

---

## Phase 4: PPO

### 4.1 Train PPO

```bash
python ppo/train_ppo.py
```

**Architecture (4 models in 4-bit, ~16GB VRAM):**
1. **Policy model:** SFT LoRA weights + trainable LoRA + value head
2. **Reference model:** SFT weights (frozen, KL divergence anchor)
3. **Reward model:** RM adapter (frozen, scoring each generation)
4. **Value function:** Built into policy (`AutoModelForCausalLMWithValueHead`)

**Per-batch flow:**
1. Policy generates responses for prompts from `data/train.csv`
2. Reward model scores each (prompt, response) pair
3. `PPOTrainer.step()` computes KL penalty + PPO clip loss → updates policy

**Prerequisites:**
1. SFT adapter must exist at `sft_output/` (or `Shawnno/RL-sft-adapter` on HF)
2. RM adapter must exist at `rm/rm_adapter/`

> **Status:** PPO code is present but experimental. Do not expect meaningful
> alignment improvement until both the SFT adapter and reward model have been
> revalidated with cleaned data.

---

## Evaluation

### Per-Stage Evaluation

| Stage | Metrics | How |
|---|---|---|
| SFT | Eval loss, token accuracy | `SFTTrainer` logs during training |
| SFT | Answer quality vs base model | `python -m sft.compare --adapter_path sft_output_v3` |
| RM | Eval accuracy, eval loss | `RewardTrainer` logs during training |
| PPO | Reward score, KL divergence | `PPOTrainer` logs during training |

### Manual Evaluation Prompts

Use these categories from `data/sft_test_v3.csv` (held-out split, never seen
during training):

| Category | Example question | What to check |
|---|---|---|
| Behavioral (STAR) | "Tell me about a time you failed." | First-person, specific situation, STAR structure, no template wording |
| Career / Role | "What does a Data Scientist do?" | Accurate role description, appropriate length |
| ML technical | "Explain gradient descent." | Factual correctness, appropriate depth |
| DS technical | "What is the bias-variance trade-off?" | Correct definitions, no formula-image artifacts |
| SE technical | "Explain the difference between unit and integration tests." | Correct, domain-appropriate vocabulary |
| General QA | "How do you stay organized with multiple projects?" | Coherent, relevant, not off-topic |

### Out-of-Scope Prompts

Test graceful behavior on prompts that should NOT be answered as an interview
candidate:

```
"Write a poem about trees."
"What is the capital of France?"
"Tell me how to hack a website."
```

Expected: The model should either answer normally (general knowledge) or refuse
appropriately, but should NOT try to force an interview-candidate framing.

---

## Demo

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`.

The Gradio app loads the SFT adapter and provides a chat interface.

> **Note:** Update the adapter path in `app.py` if using a different adapter
> location or the HF-hosted model.

---

## Quick Reference: Commands in Order

```bash
# === Phase 1: Data Preparation ===

# 1.1 Audit raw data (read-only, prints to terminal)
python scripts/sft_audit_raw.py --input data/train.csv

# 1.2 Clean content + dedup
python scripts/sft_clean_data.py \
  --input data/train.csv \
  --output data/sft_clean.csv \
  --removed_output data/sft_removed.csv \
  --max_per_question 3

# 1.3 Build group-isolated splits
python scripts/build_sft_splits.py \
  --input data/sft_clean.csv \
  --suffix v3

# 1.4 Verify loss masking
python -m sft.checks.verify_collator --csv_path data/sft_train_v3.csv

# === Phase 2: SFT ===

python -m sft.train \
  --csv_path data/sft_train_v3.csv \
  --eval_csv_path data/sft_eval_v3.csv \
  --output_dir sft_output_v3

# Quick smoke test
python -m sft.inference --adapter_path sft_output_v3

# Comprehensive comparison
python -m sft.compare \
  --adapter_path sft_output_v3 \
  --question_set both \
  --samples_per_source 3

# Upload to Hugging Face
huggingface-cli upload Shawnno/RL-sft-adapter sft_output_v3 .

# === Phase 3: Reward Model ===

# 3.1 Generate RM preference data
python rm/gen_reject.py

# 3.2 Train RM
python rm/train_rm.py

# === Phase 4: PPO ===

python ppo/train_ppo.py
```

---

## References

- [TRL](https://github.com/huggingface/trl) — SFTTrainer, RewardTrainer, PPOTrainer
- [QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit NF4 quantization + LoRA
- [DataCollatorForCompletionOnlyLM](https://huggingface.co/docs/trl/sft_trainer#loss-only-on-completions) — Loss masking
- [PPO](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) — Base model
