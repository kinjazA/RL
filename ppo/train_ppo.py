"""
train_ppo.py — Phase 3: PPO alignment training.

Architecture (4 models in 4-bit):
  - Policy model:   SFT weights + trainable LoRA + value head
  - Reference model: SFT weights (frozen, KL anchor)
  - Reward model:    RM adapter (frozen, scoring)
  - Value function:  built into policy (AutoModelForCausalLMWithValueHead)

Flow per batch:
  1. Policy generates responses for prompts
  2. Reward model scores each (prompt, response) pair
  3. PPOTrainer.step() computes KL penalty + PPO clip loss → updates policy

Input:  data/train.csv  (questions column as prompts)
Output: ppo/ppo_adapter/ (LoRA weights, ~115MB)

Usage:
  python ppo/train_ppo.py

Hardware: A40 (46GB) or any GPU with 24GB+ VRAM
"""

import os
import random

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "..")
DATA_PATH = os.path.join(ROOT, "data", "train.csv")
SFT_ADAPTER = os.path.join(ROOT, "sft_output")
RM_ADAPTER = os.path.join(ROOT, "rm", "rm_adapter")
OUTPUT_DIR = os.path.join(BASE_DIR, "ppo_adapter")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# ChatML
# ---------------------------------------------------------------------------
U = "<|im_start|>user"
A = "<|im_start|>assistant"
E = "<|im_end|>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chatml_prompt(question: str) -> str:
    return f"{U}\n{question}{E}\n{A}\n"


def chatml_full(question: str, answer: str) -> str:
    return f"{U}\n{question}{E}\n{A}\n{answer}{E}"


# ---------------------------------------------------------------------------
# 1. Load prompts
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
prompts = df["question"].drop_duplicates().tolist()
print(f"Loaded {len(prompts)} unique prompts")

# ---------------------------------------------------------------------------
# 2. QLoRA base config
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# 3. Tokenizers
# ---------------------------------------------------------------------------
# Policy tokenizer (left padding for generation)
policy_tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER, trust_remote_code=True)
policy_tokenizer.pad_token = policy_tokenizer.eos_token
policy_tokenizer.padding_side = "left"

# RM tokenizer
rm_tokenizer = AutoTokenizer.from_pretrained(RM_ADAPTER, trust_remote_code=True)

# ---------------------------------------------------------------------------
# 4. Policy model — SFT weights → merge → ValueHead → fresh LoRA
# ---------------------------------------------------------------------------
print("Loading policy model ...")
policy_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True,
)
policy_base = prepare_model_for_kbit_training(policy_base)

# Load SFT adapter and merge into 4-bit weights
policy_base = PeftModel.from_pretrained(policy_base, SFT_ADAPTER)
policy_base = policy_base.merge_and_unload()
print("  SFT adapter merged into base")

# Wrap with value head for PPO
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_base)

# Apply fresh LoRA — only the new deltas are trained
policy_model = get_peft_model(policy_model, lora_config)
policy_model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 5. Reference model — same SFT weights, frozen, no LoRA
# ---------------------------------------------------------------------------
print("Loading reference model ...")
ref_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True,
)
ref_base = PeftModel.from_pretrained(ref_base, SFT_ADAPTER)
ref_base = ref_base.merge_and_unload()
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_base)

for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()
print("  Reference model frozen")

# ---------------------------------------------------------------------------
# 6. Reward model — RM adapter, frozen
# ---------------------------------------------------------------------------
print("Loading reward model ...")
rm_base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, num_labels=1,
)
rm_base.config.pad_token_id = rm_tokenizer.pad_token_id
reward_model = PeftModel.from_pretrained(rm_base, RM_ADAPTER)
reward_model = reward_model.merge_and_unload()
for p in reward_model.parameters():
    p.requires_grad = False
reward_model.eval()
print("  Reward model frozen")

# ---------------------------------------------------------------------------
# 7. Build dataset (just prompt strings)
# ---------------------------------------------------------------------------
dataset = Dataset.from_dict({"question": prompts})
print(f"Dataset: {len(dataset)} prompts")

# ---------------------------------------------------------------------------
# 8. PPO Config
# ---------------------------------------------------------------------------
ppo_config = PPOConfig(
    output_dir=OUTPUT_DIR,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    bf16=True,
    logging_steps=5,
    report_to="none",
    remove_unused_columns=False,
    optimize_cuda_cache=True,
)

# ---------------------------------------------------------------------------
# 9. PPOTrainer
# ---------------------------------------------------------------------------
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=policy_tokenizer,
    dataset=dataset,
)

# ---------------------------------------------------------------------------
# 10. Generation config
# ---------------------------------------------------------------------------
gen_kwargs = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "pad_token_id": policy_tokenizer.eos_token_id,
    "eos_token_id": policy_tokenizer.eos_token_id,
}

# ---------------------------------------------------------------------------
# 11. PPO training loop
# ---------------------------------------------------------------------------
NUM_EPOCHS = 2
STEPS_PER_EPOCH = 200  # 200 * 8 = 1600 responses per epoch

print(f"\nStarting PPO training — {STEPS_PER_EPOCH} steps/epoch, {NUM_EPOCHS} epochs\n")

global_step = 0

for epoch in range(NUM_EPOCHS):
    random.shuffle(prompts)
    epoch_rewards = []

    for i in range(STEPS_PER_EPOCH):
        # --- pick batch ---
        batch_idx = (i * ppo_config.batch_size) % len(prompts)
        batch_prompts = [
            prompts[(batch_idx + j) % len(prompts)]
            for j in range(ppo_config.batch_size)
        ]

        # --- tokenize queries ---
        query_texts = [chatml_prompt(p) for p in batch_prompts]
        query_tensors = [
            policy_tokenizer.encode(t, return_tensors="pt", truncation=True,
                                    max_length=512).squeeze(0)
            for t in query_texts
        ]

        # --- generate responses ---
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **gen_kwargs,
        )

        # --- decode answers ---
        answers = [
            policy_tokenizer.decode(r.squeeze(0) if r.dim() > 1 else r,
                                    skip_special_tokens=True).strip()
            for r in response_tensors
        ]

        # --- reward scoring ---
        rewards = []
        for q, a in zip(batch_prompts, answers):
            rm_text = chatml_full(q, a)
            rm_inputs = rm_tokenizer(
                rm_text, return_tensors="pt", truncation=True, max_length=1024,
            ).to(reward_model.device)
            with torch.no_grad():
                score = reward_model(**rm_inputs).logits[0, 0].item()
            rewards.append(score)

        reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

        # --- PPO step ---
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        epoch_rewards.extend(rewards)
        global_step += 1

        if global_step % 10 == 0:
            avg_r = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            batch_r = sum(rewards) / len(rewards)
            kl_val = stats.get("objective/kl", 0.0)
            print(f"  step {global_step:4d}  "
                  f"batch_r={batch_r:+.3f}  avg_r={avg_r:+.3f}  "
                  f"kl={kl_val:.4f}  "
                  f"ans_len={sum(len(a) for a in answers)//len(answers)}")

    avg_r = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} done — avg reward: {avg_r:+.3f}\n")

# ---------------------------------------------------------------------------
# 12. Save
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
policy_model.save_pretrained(OUTPUT_DIR)
policy_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved PPO adapter → {OUTPUT_DIR}")
