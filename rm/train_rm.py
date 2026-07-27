"""
train_rm.py — Train a Reward Model with group-isolated split + eval summary.

Architecture:
  - Base: Qwen2.5-3B-Instruct (4-bit QLoRA)
  - Head:  linear layer on last hidden state → scalar reward
  - Loss:  pairwise ranking (-log sigmoid(chosen - rejected))

Split:  Grouped by normalized prompt — no prompt appears in both train & eval.
        This prevents inflated accuracy from memorized question patterns.

Eval:   Writes eval summary JSON with overall + per-domain + per-reject-type accuracy.

Input:  data/rm_train_v2.csv  (prompt, chosen, rejected, source, domain, rejected_type)
Output: rm/rm_adapter_v2/     (LoRA weights)
        rm/rm_eval_summary.json (evaluation breakdown)

Usage:  python rm/train_rm.py [--data data/rm_train_v2.csv] [--output_dir rm/rm_adapter_v2]
"""

import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
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
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "rm_train_v2.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "rm_adapter_v2")
EVAL_SUMMARY_PATH = os.path.join(BASE_DIR, "rm_eval_summary.json")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# 1. Load Preference Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Loaded {len(df)} RM pairs")

# Carry metadata through for per-domain eval
for col in ["source", "domain", "rejected_type"]:
    if col not in df.columns:
        df[col] = "unknown"
    df[col] = df[col].fillna("unknown").astype(str)

# ---------------------------------------------------------------------------
# 2. Group-Isolated Train/Eval Split
# ---------------------------------------------------------------------------

def _normalize_for_group(text: str) -> str:
    """Normalize prompt text so nearly-identical questions share the same group."""
    t = str(text).lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


# Assign each row a group key from the normalized prompt
df["_group"] = df["prompt"].map(_normalize_for_group)

# Shuffle groups, assign 90% to train, 10% to eval
groups = list(set(df["_group"]))
import random
random.seed(42)
random.shuffle(groups)
n_train_groups = max(1, int(len(groups) * 0.9))
train_groups = set(groups[:n_train_groups])
eval_groups = set(groups[n_train_groups:])

train_mask = df["_group"].isin(train_groups)
eval_mask = df["_group"].isin(eval_groups)

train_df = df[train_mask].copy()
eval_df = df[eval_mask].copy()

# Verify no group leakage
leaked = len(train_groups & eval_groups)
assert leaked == 0, f"BUG: {leaked} groups leaked across train/eval split!"

print(f"Groups: {len(groups)} total, {len(train_groups)} train, {len(eval_groups)} eval")
print(f"Rows:   {len(train_df):,} train, {len(eval_df):,} eval")
print(f"Group leakage check: OK (0 leaked)")

# ---------------------------------------------------------------------------
# 3. Format as ChatML sequences
# ---------------------------------------------------------------------------
def format_rm_row(prompt: str, chosen: str, rejected: str) -> dict:
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
# 4. Load Model with QLoRA
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    num_labels=1,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.pad_token_id

# Build datasets from DataFrames (preserving metadata for eval)
def _build_dataset(sub_df: pd.DataFrame) -> Dataset:
    records = [format_rm_row(row["prompt"], row["chosen"], row["rejected"])
               for _, row in sub_df.iterrows()]
    ds = Dataset.from_list(records)
    return ds

train_ds = _build_dataset(train_df)
eval_ds = _build_dataset(eval_df)

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

train_ds = train_ds.map(_tok, batched=True, remove_columns=["chosen", "rejected"])
eval_ds = eval_ds.map(_tok, batched=True, remove_columns=["chosen", "rejected"])

# ---------------------------------------------------------------------------
# 5. LoRA config
# ---------------------------------------------------------------------------
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
# 6. Train
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
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

trainer.train()

# ---------------------------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved RM adapter -> {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 8. Eval Summary (per-domain, per-reject-type accuracy)
# ---------------------------------------------------------------------------
print("\n--- Eval Summary ---")

model.eval()
eval_preds = []
eval_labels = []

# Evaluate in small batches
with torch.no_grad():
    for i in range(0, len(eval_ds), 4):
        batch = eval_ds[i:i+4]
        chosen_inputs = {
            "input_ids": torch.tensor(batch["input_ids_chosen"]).to(model.device),
            "attention_mask": torch.tensor(batch["attention_mask_chosen"]).to(model.device),
        }
        rejected_inputs = {
            "input_ids": torch.tensor(batch["input_ids_rejected"]).to(model.device),
            "attention_mask": torch.tensor(batch["attention_mask_rejected"]).to(model.device),
        }
        c_score = model(**chosen_inputs).logits.squeeze(-1)
        r_score = model(**rejected_inputs).logits.squeeze(-1)
        eval_preds.extend((c_score > r_score).cpu().tolist())
        eval_labels.extend([True] * len(batch["input_ids_chosen"]))

overall_acc = accuracy_score(eval_labels, eval_preds)
print(f"Overall eval accuracy: {overall_acc:.4f} ({sum(eval_preds)}/{len(eval_preds)})")

# Per-domain accuracy
eval_df_eval = eval_df.reset_index(drop=True)
assert len(eval_df_eval) == len(eval_preds), f"Length mismatch: {len(eval_df_eval)} vs {len(eval_preds)}"
per_domain = defaultdict(lambda: {"correct": 0, "total": 0})
per_reject_type = defaultdict(lambda: {"correct": 0, "total": 0})

for i, (pred, label) in enumerate(zip(eval_preds, eval_labels)):
    dom = eval_df_eval.iloc[i]["domain"]
    rtype = eval_df_eval.iloc[i]["rejected_type"]
    per_domain[dom]["total"] += 1
    per_domain[dom]["correct"] += int(pred)
    per_reject_type[rtype]["total"] += 1
    per_reject_type[rtype]["correct"] += int(pred)

print("\nPer-domain accuracy:")
for dom in sorted(per_domain):
    d = per_domain[dom]
    print(f"  {dom:<18}: {d['correct']}/{d['total']} = {d['correct']/max(1,d['total']):.3f}")

print("\nPer reject-type accuracy:")
for rt in sorted(per_reject_type):
    r = per_reject_type[rt]
    print(f"  {rt:<22}: {r['correct']}/{r['total']} = {r['correct']/max(1,r['total']):.3f}")

# Save summary
summary = {
    "overall_accuracy": float(overall_acc),
    "total_eval_pairs": len(eval_preds),
    "per_domain": {dom: {"accuracy": float(d["correct"] / max(1, d["total"])),
                         "correct": d["correct"], "total": d["total"]}
                   for dom, d in per_domain.items()},
    "per_reject_type": {rt: {"accuracy": float(r["correct"] / max(1, r["total"])),
                             "correct": r["correct"], "total": r["total"]}
                        for rt, r in per_reject_type.items()},
    "split_info": {
        "n_train_rows": len(train_df),
        "n_eval_rows": len(eval_df),
        "n_train_groups": len(train_groups),
        "n_eval_groups": len(eval_groups),
        "group_leakage": 0,
    },
}

os.makedirs(os.path.dirname(EVAL_SUMMARY_PATH), exist_ok=True)
with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nEval summary saved -> {EVAL_SUMMARY_PATH}")
