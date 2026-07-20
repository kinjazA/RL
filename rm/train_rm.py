"""
train_rm.py — Train a Reward Model on preference pairs.

Architecture:
  - Base: Qwen2.5-3B-Instruct (4-bit QLoRA)
  - Head:  linear layer on top of last hidden state → scalar reward
  - Loss:  pairwise ranking (-log sigmoid(chosen - rejected))

Input:  data/rm_train.csv  (prompt, chosen, rejected)
Output: rm/rm_adapter/     (LoRA weights)

Usage:
  python rm/train_rm.py

Hardware: 16GB GPU (T4 / RTX 4060+)
Time:     ~2 hours for 3 epochs on ~5K pairs
"""

import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import RewardConfig, RewardTrainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "rm_train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "rm_adapter")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# 1. Load Preference Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Loaded {len(df)} RM pairs")

# ChatML formatting for reward model: prompt + response as one sequence
def format_rm_row(row):
    """Return chosen and rejected as full ChatML sequences."""
    prompt = str(row["prompt"])
    chosen = str(row["chosen"])
    rejected = str(row["rejected"])

    chosen_text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{chosen}<|im_end|>"
    )
    rejected_text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{rejected}<|im_end|>"
    )
    return {"chosen": chosen_text, "rejected": rejected_text}


# ---------------------------------------------------------------------------
# 2. Load Model with QLoRA
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Set num_labels=1 for scalar reward output
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    num_labels=1,
)
model = prepare_model_for_kbit_training(model)

# Patch: ensure pad_token_id is set before LoRA injects modules
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

# Build tokenized dataset (needs tokenizer already loaded)
records = [format_rm_row(row) for _, row in df.iterrows()]
raw = Dataset.from_list(records).train_test_split(test_size=0.05, seed=42)

def _tok(batch):
    out = {"input_ids_chosen": [], "attention_mask_chosen": [],
           "input_ids_rejected": [], "attention_mask_rejected": []}
    for c, r in zip(batch["chosen"], batch["rejected"]):
        ce = tokenizer(c, truncation=True, max_length=1024)
        re = tokenizer(r, truncation=True, max_length=1024)
        out["input_ids_chosen"].append(ce["input_ids"])
        out["attention_mask_chosen"].append(ce["attention_mask"])
        out["input_ids_rejected"].append(re["input_ids"])
        out["attention_mask_rejected"].append(re["attention_mask"])
    return out

dataset = raw.map(_tok, batched=True, remove_columns=["chosen", "rejected"])
print(f"Train: {len(dataset['train']):,}  |  Val: {len(dataset['test']):,}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
training_args = RewardConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False,
    max_length=1024,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved RM adapter → {OUTPUT_DIR}")
