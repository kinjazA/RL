# RLHF Interview Feedback Assistant

End-to-end RLHF pipeline to align a language model for **constructive interview feedback** — training the model to point out issues and suggest improvements, rather than giving generic answers or doing the candidate's work for them.

Built on **Qwen2.5-3B-Instruct** with **QLoRA** (4-bit quantization), runnable on a single T4 GPU.

---

## Project Motivation

Most interview prep tools just generate model answers. This project explores a different direction: **feedback that teaches, not tells**.

| Behavior | Undesirable | Desirable |
|---|---|---|
| Candidate gives a vague answer | "Here's a better answer: ..." | "Your answer lacks a specific example. Try the STAR format: Situation → Task → Action → Result." |
| Candidate says "I'm too perfectionist" | "Everyone says that, say this instead..." | "This is a textbook 'weakness disguised as strength'. Pick a real but non-fatal weakness and mention what you're doing to improve it." |
| Candidate rambles for 2 minutes | "Keep it short." | "Your answer took ~2 minutes. For behavioral questions, aim for 60–90 seconds. The key detail was buried at the end — lead with it next time." |

---

## Pipeline Overview

```
                  ┌──────────────────────────────────────┐
                  │            Phase 1: SFT               │
                  │                                      │
                  │  Qwen2.5-3B-Instruct                  │
                  │       + QLoRA (4bit)                  │
                  │       + Mixed data (~8K samples)      │
                  │         · Alpaca general instructions │
                  │         · OpenAssistant conversations │
                  │         · Interview feedback pairs    │
                  │                                      │
                  │  Output: Model learns to give          │
                  │  interview feedback in dialogue format│
                  └──────────────────┬───────────────────┘
                                     ↓
                  ┌──────────────────────────────────────┐
                  │         Phase 2: Reward Model         │
                  │                                      │
                  │  Same base model (Qwen2.5-3B)         │
                  │       + QLoRA                         │
                  │       + Reward head (Linear: d→1)     │
                  │       + Preference pairs (~300)       │
                  │         · Good feedback: specific,    │
                  │           actionable, asks questions  │
                  │         · Bad feedback: generic,       │
                  │           gives away answer, vague    │
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
                  │  1. Policy generates feedback         │
                  │  2. Reward model scores it            │
                  │  3. Reference model computes KL       │
                  │  4. Critic estimates expected value   │
                  │  5. PPO clip loss updates policy      │
                  └──────────────────┬───────────────────┘
                                     ↓
                           Aligned Model
                    (gives helpful, specific,
                     non-answer-spoiling feedback)
```

---

## Technical Stack

| Component | Choice | Rationale |
|---|---|---|
| **Base Model** | Qwen2.5-3B-Instruct | Good instruction-following, English-capable, fits T4 with QLoRA |
| **PEFT** | QLoRA (4-bit NF4) | 4 models in PPO fit within ~16GB VRAM |
| **SFT Data** | Alpaca + OpenAssistant + custom | ~8K samples: 5K interview / 2K general / 1K GPT-4 synthesized |
| **RM Data** | Self-annotated preference pairs | ~300 pairs: good vs bad feedback for same Q&A |
| **PPO Framework** | TRL (HuggingFace) | Production-ready RLHF library |
| **Hardware** | Single T4 (16GB) | Free Colab tier or any cloud T4 instance |

---

## Data

### SFT Data (~8K samples, 67:33 domain-to-general ratio)

| Source | Count | Purpose |
|---|---|---|
| Custom interview feedback | ~5,000 | Core domain skill: giving interview advice |
| Alpaca + OpenAssistant (OASST1) | ~2,000 | General instruction following & conversation |
| LLM-synthesized (GPT-4) | ~1,000 | Diversity expansion: var iations on interview scenarios |

**Custom data format:**
```json
{
  "instruction": "Give constructive feedback on this interview answer.\n\nQuestion: What's your biggest weakness?\nAnswer: I work too hard and sometimes forget to take breaks.",
  "output": "This is a classic 'humble-brag' weakness. Interviewers hear this constantly and it signals lack of self-awareness. Try a real but non-fatal weakness, structured as:\n1. What the weakness is (be specific)\n2. A concrete example of when it caused a problem\n3. What you're actively doing to improve\n\nFor example: 'I struggle with delegating. On a recent project, I took on too much myself and missed a deadline. Now I use a RACI matrix at project kickoff to clarify ownership.'"
}
```

### RM Data (~300 preference pairs)

| Source | Count |
|---|---|
| Self-annotated | ~300 pairs |

**Preference pair format:**
```json
{
  "prompt": "Question: Why did you leave your last job?\nCandidate's answer: The culture was toxic and my boss was terrible. I needed to get out.",
  "chosen": "Your answer focuses entirely on the negative. Even if true, this raises red flags for interviewers. Try reframing around what you're looking FOR rather than what you're running FROM. For example: 'I'm looking for a role with more ownership and a collaborative culture — my last position had become very siloed.' This says the same thing without badmouthing anyone.",
  "rejected": "Never badmouth your previous employer. Just say you wanted new challenges or growth opportunities. Also, your tone is too emotional — stay professional."
}
```

---

## PPO Training Details

### Why 4 models?

| Model | Trainable | Role |
|---|---|---|
| **Policy** | Yes | The model being aligned — generates feedback |
| **Reference** | No (frozen) | SFT checkpoint — computes KL divergence to prevent catastrophic forgetting |
| **Reward** | No (frozen) | Scores generated feedback |
| **Critic** | Yes | Learns to predict expected reward; enables advantage calculation |

### PPO Clip Loss

The core objective that prevents the policy from changing too much in one update:

```
L_clip = max(-advantage × ratio, -advantage × clip(ratio, 0.8, 1.2))

where:
  ratio = π_new(token) / π_old(token)
  advantage = actual_reward - critic_estimate
  clip ensures ratio stays within [0.8, 1.2]
```

### KL Regularization

Without a KL penalty (computed against the frozen reference model), the policy may "reward hack" — generating nonsense that scores highly under the reward model but has lost all language coherence.

```
effective_reward = reward_model_score - β × KL(policy || reference)
```

---

## Evaluation & Comparison

After training, we compare SFT-only vs RLHF-aligned outputs on held-out prompts:

| Prompt | SFT Output | RLHF Output |
|---|---|---|
| "Tell me about a time you failed." + generic answer | "Try using STAR format next time." | "Your answer has the right structure but lacks specifics. What was the actual failure? How did you recover? Add those details — interviewers want resilience and learning, not a perfect track record." |
| "Why do you want this job?" + cliché answer | "Be more specific about the company." | "You mentioned the company's reputation but nothing about the role itself. What about this specific position excites you? Connect it to your skills: 'I want to use my X experience to solve Y problem at your company.'" |

**Metrics tracked:**
- **Reward score** (RM evaluation): expected to rise from ~0.5 → 2.0+
- **KL divergence**: expected to stay below 0.05 (no reward hacking)
- **Qualitative human review**: feedback specificity, actionability, non-answer-spoiling rate

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/kinjazA/RL.git
cd RL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
python data/prepare_sft_data.py    # downloads Alpaca/OASST1 + merges custom data
python data/prepare_rm_data.py     # formats preference pairs

# 4. Train SFT
python sft/train_sft.py            # QLoRA, ~2 hours on T4

# 5. Train Reward Model
python rm/train_rm.py              # QLoRA + reward head, ~1 hour on T4

# 6. PPO Alignment
python ppo/train_ppo.py            # TRL PPOTrainer, ~3 hours on T4

# 7. Compare
python eval/evaluate.py            # Side-by-side SFT vs RLHF outputs
```

---

## Roadmap

### Phase 1 — Data Preparation

#### 1.1 Interview Seed Examples
- [ ] Define 6–8 interview question categories:
  - Behavioral ("Tell me about a time you failed")
  - Technical ("Explain a difficult technical decision")
  - Situational ("What would you do if you missed a deadline")
  - Background ("Walk me through your resume")
  - Weakness/Strength ("What's your biggest weakness")
  - Career motivation ("Why do you want this job / leave current role")
  - Teamwork/Conflict ("Deal with a difficult coworker")
  - Closing questions ("Do you have any questions for us")
- [ ] For each category, write 3–5 candidate answers at different quality levels:
  - `poor` — vague, no example, too short or too long
  - `mediocre` — right structure but lacks specifics
  - `solid` — good content but delivery could improve
- [ ] Write "good feedback" for each answer
  - Max 80 words
  - Structures: "Your answer is [weak/ok], not the format, then demonstrates how to improve
  - Form at: point out issue → explain why it matters → give concrete rewrite example
- [ ] Write "bad feedback" for each answer (used as negative example in RM training)
  - Generic advice: "Be more specific" / "Use STAR format" without demonstrating how
  - Answer-spoiling: rewrites the entire answer for them instead of teaching
  - Vague praise: "Good job!" without pointing out what can still improve
- [ ] Total seed dataset: **250–300 examples**
  - ~50 per category (covering all 3 quality levels)

#### 1.2 Expand via GPT-4
- [ ] Prompt design: give GPT-4 seed examples as few-shot, instruct it to generate new (Q, A, feedback) triples
  - Prompt includes: 3 seed examples → then "Generate 10 new triples in the same format"
  - Specify constraints: question stays within the same category, answer quality varies, feedback matches the style
- [ ] Run generation for each of the 8 categories
- [ ] Total generated: **~5,000 examples**
  - ~600–700 per category
- [ ] Manual spot-check: review 50 random generated samples, check feedback quality
  - Reject criteria: feedback gives away the answer, feedback is longer than 100 words, feedback is generic praise
  - Accept when: feedback is specific, actionable, includes rewrite hints
- [ ] Final domain dataset: **~5,000 clean samples**

#### 1.3 General Instruction Data
- [ ] Download Alpaca dataset (`tatsu-lab/alpaca`, 52K samples)
- [ ] Download OpenAssistant OASST1 (`OpenAssistant/oasst1`, filter English only)
- [ ] Random sample ~1,500 from Alpaca + ~500 from OpenAssistant
- [ ] Remove any samples where `output` is empty or < 10 characters
- [ ] General subset: **~2,000 clean samples**

#### 1.4 LLM-Synthesized Diversity Data
- [ ] Use GPT-4 to generate ~1,000 edge-case interview scenarios
  - Cross-category hybrids ("Weakness question but candidate gave a teamwork example")
  - Unusual answer tones (overly technical, too emotional, too casual)
  - Follow-up question simulation ("The interviewer pushes back — what now?")
- [ ] Spot-check 30 random samples
- [ ] Synthesis subset: **~1,000 clean samples**

#### 1.5 Final Dataset Assembly
- [ ] Merge 5K interview + 2K general + 1K synthesis = **8,000 total**
- [ ] Shuffle with fixed random seed (seed=42)
- [ ] Split: 7,600 train / 400 held-out validation
- [ ] Verify no duplicate questions across train/val
- [ ] Save as `data/sft_train.jsonl` and `data/sft_val.jsonl`
- [ ] Format per sample:
  ```json
  {"instruction": "<question>\n<candidate answer>\nGive constructive feedback:", "output": "<feedback>"}
  ```

#### 1.6 Reward Model Preference Pairs
- [ ] Select 300 prompts (interview Q&A pairs) from the SFT training set
- [ ] For each prompt, write two feedbacks:
  - `chosen` — specific, actionable, teaches without giving answer
  - `rejected` — one of: generic advice / answer-spoiling / vague praise
- [ ] Verify that `chosen` is objectively better by reviewing 30 random pairs
  - Flip test: if you needed interview prep help, would you prefer this feedback?
- [ ] Format per pair:
  ```json
  {"prompt": "<question>\n<answer>", "chosen": "<good feedback>", "rejected": "<bad feedback>"}
  ```
- [ ] Save as `data/rm_preferences.jsonl`

---

### Phase 2 — SFT (Supervised Fine-Tuning)

#### 2.1 Environment & Model Setup
- [ ] Install dependencies: `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `trl`
- [ ] Pin versions in `requirements.txt`
- [ ] Verify GPU: `torch.cuda.get_device_name()` → T4 16GB
- [ ] Load `Qwen/Qwen2.5-3B-Instruct` with 4-bit NF4 quantization via `bitsandbytes`
  - `load_in_4bit=True`
  - `bnb_4bit_compute_dtype=torch.bfloat16`
  - `bnb_4bit_quant_type="nf4"`
- [ ] Verify: model loaded, peak memory < 6 GB

#### 2.2 QLoRA Configuration
- [ ] LoRA config:
  - `r=16` (rank)
  - `lora_alpha=32`
  - `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - `lora_dropout=0.05`
- [ ] Verify: trainable params ~0.5% of total (≈ 15M / 3B)
- [ ] Apply LoRA adapter → `peft.get_peft_model()`

#### 2.3 Training
- [ ] Training arguments:
  - `per_device_train_batch_size=4`
  - `gradient_accumulation_steps=4` (effective batch size = 16)
  - `num_train_epochs=3`
  - `learning_rate=2e-4`
  - `lr_scheduler_type="cosine"`
  - `warmup_ratio=0.03`
  - `bf16=True`
  - `logging_steps=10`
  - `save_steps=200`
  - `max_seq_length=1024`
- [ ] Load `data/sft_train.jsonl` via `datasets.load_dataset()`
- [ ] Train with `SFTTrainer` (from TRL)
- [ ] Expected: ~2 hours on T4
- [ ] Monitor: training loss drops from ~2.5 to ~1.0–1.5

#### 2.4 Validation
- [ ] Run inference on 10 prompts from `data/sft_val.jsonl`
- [ ] Check outputs:
  - [ ] All responses are in feedback format (not just answering the question)
  - [ ] No responses are empty or truncated
  - [ ] Feedback is relevant to the specific answer given
- [ ] Save SFT adapter weights: `checkpoints/sft/`
- [ ] Merge adapter into base model for RM/PPO use: `model.merge_and_unload()`

---

### Phase 3 — Reward Model

#### 3.1 Model Architecture
- [ ] Load SFT-merged model (Qwen2 .5-3B + SFT adapter merged)
- [ ] Replace LM head with reward head:
  ```python
  model.reward_head = nn.Linear(hidden_size, 1)
  # Only train reward_head + final 2 transformer layers
  ```
- [ ] Freeze all other parameters
- [ ] Apply QLoRA on the trainable layers
- [ ] Verify: trainable params < 2% of total

#### 3.2 Training
- [ ] Load `data/rm_preferences.jsonl`
- [ ] Training arguments:
  - `per_device_train_batch_size=2` (each batch = 1 prompt × 2 responses)
  - `gradient_accumulation_steps=8`
  - `num_train_epochs=3`
  - `learning_rate=1e-4`
  - `max_seq_length=1024`
- [ ] Loss function: Pairwise Ranking Loss
  ```python
  probs = torch.sigmoid(score_chosen - score_rejected)
  loss = -torch.log(probs + 1e-5).mean()
  ```
- [ ] Expected: ~1 hour on T4
- [ ] Monitor: accuracy (score_chosen > score_rejected) → target > 75%

#### 3.3 Validation
- [ ] Create 50 held-out preference pairs (not in training)
- [ ] Evaluate: compute ranking accuracy
  - Target: accuracy > 70%
- [ ] Spot-check: for 10 pairs, manually verify RM ordering makes sense
- [ ] If accuracy < 70%: add more preference data, retrain
- [ ] Save RM checkpoint: `checkpoints/rm/`
- [ ] Merge RM adapter for PPO use

---

### Phase 4 — PPO Alignment

#### 4.1 Model Loading (4 models, all QLoRA 4-bit)
- [ ] **Policy model**: load SFT-merged checkpoint, apply fresh QLoRA adapter (trainable)
- [ ] **Reference model**: load SFT-merged checkpoint (frozen, no gradient)
- [ ] **Reward model**: load RM checkpoint (frozen, no gradient)
- [ ] **Critic model**: load SFT-merged checkpoint, add value head, apply QLoRA (trainable)
- [ ] Verify total VRAM usage < 14 GB (peaks inside T4's 16GB)

#### 4.2 Prompt Dataset
- [ ] Use 500 interview Q&A pairs from SFT training set as PPO prompts
- [ ] Each prompt format:
  ```
  Question: <interview question>
  Candidate's answer: <their answer>
  Give constructive feedback:
  ```
- [ ] No ground-truth feedback needed for PPO — RM scores model output

#### 4.3 PPO Configuration
- [ ] TRL `PPOConfig`:
  - `model_name="policy_qwen_3b"`
  - `learning_rate=1e-5`
  - `batch_size=8` (experience batch)
  - `mini_batch_size=4` (PPO update batch)
  - `ppo_epochs=4`
  - `cliprange=0.2` (ratio clip: [0.8, 1.2])
  - `kl_penalty="kl"`
  - `init_kl_coef=0.05`
  - `target_kl=0.05`
  - `steps=5000`
  - `whiten_rewards=True` (normalize rewards)
  - `score_clip=5.0` (clip extreme rewards)

#### 4.4 Training Loop
- [ ] Initialize `PPOTrainer` from TRL
- [ ] Run for 5000 steps (~3 hours on T4)
- [ ] Each step:
  1. Policy generates responses for batch of prompts
  2. Reward model scores each response
  3. Reference model computes per-token KL
  4. Critic estimates values
  5. Compute advantage = effective_reward − critic_estimate
  6. PPO clip loss → backprop → update policy + critic
  7. Log: reward mean, KL mean, policy loss, value loss

#### 4.5 Monitoring Dashboard
- [ ] Log every 50 steps:
  - `reward/mean` — average RM score (target: increasing trend, 0.5 → 2.0+)
  - `reward/std` — score variance (target: not collapsing to single value)
  - `kl/mean` — average KL divergence (target: < 0.05, no upward drift)
  - `ppo/policy_loss` — PPO clip loss (target: stable, not NaN)
  - `ppo/value_loss` — critic MSE (target: decreasing)
  - `objective/kl_coef` — adaptive KL coefficient
- [ ] Save checkpoint every 500 steps
- [ ] Alert if: KL > 0.1 (reward hacking risk) or reward suddenly spikes > 5×

#### 4.6 Post-Training
- [ ] Merge PPO adapter into policy model → final aligned model
- [ ] Save final checkpoint: `checkpoints/ppo/aligned_model/`
- [ ] Save training metrics: `logs/ppo_metrics.csv`

---

### Phase 5 — Evaluation

#### 5.1 Test Set
- [ ] Select 20 held-out interview Q&A pairs (not in SFT training, RM preferences, or PPO prompts)
- [ ] Cover all 6–8 question categories
- [ ] Include 3 quality levels: poor / mediocre / solid answers
- [ ] Save as `data/test_prompts.jsonl`

#### 5.2 Output Generation
- [ ] Load SFT-only checkpoint → generate feedback for all 20 prompts
- [ ] Load RLHF-aligned checkpoint → generate feedback for all 20 prompts
- [ ] Same generation params for both: `temperature=0.7`, `max_new_tokens=150`, `do_sample=True`
- [ ] Save as `eval/sft_outputs.jsonl` and `eval/rlhf_outputs.jsonl`

#### 5.3 Quantitative Comparison (by RM)
- [ ] Run Reward Model on both sets of outputs
- [ ] Compute mean reward score for SFT vs RLHF
- [ ] Expected: RLHF mean > SFT mean by ≥ 0.3

#### 5.4 Qualitative Comparison (human review)
- [ ] For each of the 20 prompts, compare SFT vs RLHF output side-by-side
- [ ] Rate on 3 dimensions (1–5 scale):
  - **Specificity**: does the feedback name a concrete issue?
  - **Actionability**: can the candidate act on it immediately?
  - **Non-answer-spoil**: does it teach without doing the work for them?
- [ ] Expected: RLHF higher on all 3 dimensions

#### 5.5 Manual Showcase
- [ ] Pick 5 representative examples where RLHF clearly improves over SFT
- [ ] Format as table in `docs/evaluation.md`
- [ ] Screen shot or GIF of side-by-side comparison

---

### Phase 6 — Documentation & Polish

#### 6.1 Repository
- [ ] Finalize `README.md` with results and examples
- [ ] Add `docs/methodology.md` — technical deep-dive
- [ ] Add `docs/evaluation.md` — evaluation results with side-by-side table
- [ ] Add inline code comments to all training scripts
- [ ] `.gitignore`: exclude checkpoints, wandb logs, `.ipynb_checkpoints`

#### 6.2 Model Release
- [ ] Create HuggingFace model card:
  - Model description and intended use
  - Training data summary
  - Limitations (English-only, interview prep only, not for real hiring decisions)
  - Evaluation results
- [ ] Push merged weights to HuggingFace Hub
- [ ] Add HF Hub link to README

#### 6.3 Demo
- [ ] Record short demo video or GIF
  - Show one prompt → SFT output → RLHF output
  - Highlight qualitative improvement
- [ ] Add to README

#### 6.4 Interview Prep
- [ ] Prepare 2-minute project overview for interview
- [ ] Prepare STAR answer for "Tell me about this RL project"
- [ ] Prepare answers for likely follow-ups:
  - "Why RLHF instead of DPO?"
  - "What was the hardest part?"
  - "How would you scale this to 7B?"
  - "What would you do differently next time?"

- **MOSS-RLHF**: [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- **TRL**: [HuggingFace Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (alternative to RLHF, simpler pipeline)
- **ChatDoctor**: [A Medical Chat Model Fine-Tuned on LLaMA Using Medical Domain Knowledge](https://www.cureus.com/articles/152858) — two-stage SFT approach referenced for data mixing strategy
