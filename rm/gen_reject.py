"""
gen_reject.py — Generate rejected answers for RM training.

Mix of two strategies:
  - 70% Qwen2.5-0.5B-Instruct:  naturally lower-quality answers
  - 30% rule-based degradation:  truncate / mid-drop / shuffle / strip bullets

The mix gives RM both subtle and obvious quality differences to learn from.

Input:  data/train.csv      (question, answer, source)
Output: data/rm_train.csv   (prompt, chosen, rejected)

Hardware: GPU recommended (T4 ~1 hour)
Usage:   python rm/gen_reject.py [--batch_size 4] [--model_ratio 0.7]
"""

import csv
import os
import sys
import re
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INPUT = os.path.join(DATA_DIR, "train.csv")
OUTPUT = os.path.join(DATA_DIR, "rm_train.csv")
CHECKPOINT = os.path.join(DATA_DIR, ".gen_reject_checkpoint.txt")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MODEL_RATIO = 0.7

SYSTEM_PROMPT = "Answer the following question briefly and concisely."

random.seed(42)

# ---------------------------------------------------------------------------
# Rule-based degradation
# ---------------------------------------------------------------------------
def truncate(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    keep = random.randint(1, min(2, len(sents)))
    return " ".join(sents[:keep])


def mid_drop(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 4:
        return truncate(text)
    n = len(sents)
    cut_start = n // 4
    cut_end = 3 * n // 4
    return " ".join(sents[:cut_start] + sents[cut_end:])


def shuffle_paragraphs(text: str) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return truncate(text)
    random.shuffle(paras)
    return "\n\n".join(paras)


def strip_bullets(text: str) -> str:
    lines = text.split("\n")
    kept = [l for l in lines if not re.match(r"^\s*(\d+[\.\)]|[-*•])\s", l)]
    if len(kept) < len(lines) * 0.3:
        return truncate(text)
    return "\n".join(kept)


STRATEGIES = [truncate, mid_drop, shuffle_paragraphs, strip_bullets]

# ---- Negative-sample quality checks ----
_MIN_NEG_LEN = 10
_MAX_LEN_RATIO = 1.3  # rejected must not be >1.3x chosen (longer → may be better)


def is_valid_negative(chosen: str, rejected: str) -> bool:
    """Return False if rejected is not clearly worse than chosen."""
    if not rejected or len(rejected.strip()) < _MIN_NEG_LEN:
        return False
    if rejected.strip() == chosen.strip():
        return False
    if len(rejected) > len(chosen) * _MAX_LEN_RATIO:
        return False
    return True


def rule_reject(chosen: str) -> str:
    """Apply a random degradation strategy, ensuring output is shorter."""
    strategy = random.choice(STRATEGIES)
    rejected = strategy(chosen)
    if len(rejected) >= len(chosen) * 0.9:
        rejected = truncate(chosen)
    return rejected


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------
def load_model(device: str = "auto"):
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Device: {model.device}")
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    messages = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": p}]
        for p in prompts
    ]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        gen_ids = out[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return set(int(l.strip()) for l in f if l.strip().isdigit())
    return set()


def save_checkpoint(idx: int):
    with open(CHECKPOINT, "a") as f:
        f.write(f"{idx}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--model_ratio", type=float, default=DEFAULT_MODEL_RATIO,
                        help="Fraction of samples to use 0.5B model for (rest use rule-based)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load questions
    questions = [(r["question"], r["answer"]) for r in csv.DictReader(open(INPUT, encoding="utf-8-sig"))]
    done = load_checkpoint()
    print(f"Loaded {len(questions)} questions" + (f" ({len(done)} already done)" if done else ""))

    # Assign strategies: True = model, False = rule
    assignments = {}
    for i in range(len(questions)):
        assignments[i] = random.random() < args.model_ratio

    n_model = sum(1 for i in range(len(questions)) if assignments[i] and i not in done)
    n_rule = sum(1 for i in range(len(questions)) if not assignments[i] and i not in done)
    print(f"Strategy: {n_model} model-generated + {n_rule} rule-based")

    # Load model (only if we have model-assigned samples remaining)
    model = tokenizer = None
    if n_model > 0:
        model, tokenizer = load_model(args.device)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    write_header = not done
    out_f = open(OUTPUT, "w" if write_header else "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["prompt", "chosen", "rejected"])

    processed = 0

    model_indices = []
    model_prompts = []
    model_chosens = []

    for i in range(len(questions)):
        if i in done:
            continue

        prompt, chosen = questions[i]

        if assignments[i]:  # model-generated
            model_indices.append(i)
            model_prompts.append(prompt)
            model_chosens.append(chosen)
        else:  # rule-based — write immediately
            rejected = rule_reject(chosen)
            writer.writerow([prompt, chosen, rejected])
            save_checkpoint(i)
            processed += 1

    print(f"Rule-based: {processed} done")

    # --- Pass 2: Model-generated samples (batched GPU) ---
    if model_indices:
        print(f"Model-generated: {len(model_indices)} remaining")
        batch_size = args.batch_size
        pbar = tqdm(range(0, len(model_indices), batch_size), desc="Generating (0.5B)")
        total_fallback = 0

        for start in pbar:
            end = min(start + batch_size, len(model_indices))
            batch_prompts = model_prompts[start:end]
            batch_chosens = model_chosens[start:end]
            batch_idxs = model_indices[start:end]

            try:
                rejected_list = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM at batch size {batch_size}. Try --batch_size {max(1, batch_size // 2)}")
                out_f.close()
                sys.exit(1)

            n_fallback = 0
            for idx, prompt, chosen, rejected in zip(batch_idxs, batch_prompts, batch_chosens, rejected_list):
                if not is_valid_negative(chosen, rejected):
                    rejected = rule_reject(chosen)
                    n_fallback += 1
                writer.writerow([prompt, chosen, rejected])
                save_checkpoint(idx)
                processed += 1
            total_fallback += n_fallback

            pbar.set_postfix({"done": processed, "fallback": f"{n_fallback}/{len(batch_idxs)}"})

        if total_fallback:
            print(f"  → {total_fallback} model outputs ({100*total_fallback/len(model_indices):.0f}%) fell back to rule-based")
        print(f"Model-generated: {len(model_indices)} done")

    out_f.close()

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\nTotal: {processed} RM pairs → {OUTPUT}")

    # Show samples
    print("\n--- Samples ---")
    with open(OUTPUT, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
        for row in random.sample(rows, min(4, len(rows))):
            is_model = len(row["rejected"]) > len(row["chosen"]) * 0.5  # rough heuristic
            tag = "[0.5B]" if is_model else "[rule]"
            print(f"\n{tag} Prompt:   {row['prompt'][:100]}")
            print(f"    Chosen:   {row['chosen'][:150]}...")
            print(f"    Rejected: {row['rejected'][:150]}...")


if __name__ == "__main__":
    main()
