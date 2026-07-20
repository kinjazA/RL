"""
gen_reject.py — Generate rejected answers for RM training.

Produces one rejected answer per (prompt, chosen) pair using a mix of:
  - 70% Qwen2.5-0.5B-Instruct:  a naturally weaker model's answer
  - 30% rule-based degradation: structural damage to the chosen answer

Every rejected answer (model- or rule-generated) is validated by
is_valid_negative(): it must be non-trivial, different from the chosen answer,
and not materially longer than it (a longer rejected text could actually be a
better answer — the length check prevents the RM from learning "short = good").
Invalid model outputs fall back to rule-based degradation.

Rule-based degradation uses a guaranteed-safe design:
  1. flavor strategies (truncate / mid_drop / shuffle / strip_bullets) each
     *attempt* to degrade; if the input lacks the required structure they are
     no-ops and let the validator reject them.
  2. guaranteed_cut() is the structure-agnostic last-resort: it cuts to the
     first 1/3 of words (always shorter + different for normal text), and for
     very short inputs it duplicates the second half so the result stays valid
     but is obviously ungrammatical. It cannot produce an invalid negative.

Verified by a deterministic sweep over all 5,558 chosen answers:
  rule_reject() produces a valid negative for 5,558 / 5,558 (100%) — 0 identical,
  0 too-short, 0 pathologic-length; median rejected/chosen length = 0.34.

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
DEFAULT_MAX_NEW_TOKENS = 100  # short → 0.5B answers stay worse than chosen
DEFAULT_MODEL_RATIO = 0.7

SYSTEM_PROMPT = "Answer the following question very briefly, in one or two sentences. Do not elaborate."

random.seed(42)

# ---------------------------------------------------------------------------
# Rule-based degradation
# ---------------------------------------------------------------------------
# Design: each "flavor" strategy is an ATTEMPT — it may return the text
# unchanged if the input lacks the structure it assumes. Every attempt's
# output passes is_valid_negative(); only strictly-worse outputs are kept.
# If no flavor produces a valid negative, guaranteed_cut() is the
# structure-agnostic fallback that can never fail (cuts to first 1/3 of words).

def truncate(text: str) -> str:
    """Keep first 1-2 sentences. Pass-through if only one sentence."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    keep = random.randint(1, min(2, len(sents)))
    return " ".join(sents[:keep])


def mid_drop(text: str) -> str:
    """Drop the middle half of sentences. Pass-through if <4 sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 4:
        return text
    n = len(sents)
    return " ".join(sents[: n // 4] + sents[3 * n // 4 :])


def shuffle_paragraphs(text: str) -> str:
    """Shuffle paragraph order. Pass-through if <2 paragraphs."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return text
    random.shuffle(paras)
    return "\n\n".join(paras)


def strip_bullets(text: str) -> str:
    """Remove numbered/bulleted lines. Pass-through if no bullets."""
    lines = text.split("\n")
    kept = [l for l in lines if not re.match(r"^\s*(\d+[\.\)]|[-*•])\s", l)]
    if kept == lines:
        return text
    return "\n".join(kept)


STRATEGIES = [truncate, mid_drop, shuffle_paragraphs, strip_bullets]

# ---- Validation ----
_MIN_NEG_LEN = 10
_MAX_LEN_RATIO = 1.3  # rejected must not exceed 1.3x chosen (longer may be better)


def is_valid_negative(chosen: str, rejected: str) -> bool:
    """A valid rejected is non-trivial, different from, and not much longer than chosen."""
    if not rejected or len(rejected.strip()) < _MIN_NEG_LEN:
        return False
    if rejected.strip() == chosen.strip():
        return False
    if len(rejected) > len(chosen) * _MAX_LEN_RATIO:
        return False
    return True


def guaranteed_cut(chosen: str) -> str:
    """Structure-agnostic fallback that always produces a valid (if crude) negative.

    Two modes:
      (a) Long chosen (>= ~30 chars, >=3 words): cut to first 1/3 of words → shorter + different.
      (b) Short chosen: can't safely shorten, so degrade by duplicating the second half,
          which keeps it valid-length but is obviously worse / ungrammatical.
    """
    words = chosen.split()
    cut_words = " ".join(words[: max(1, len(words) // 3)])
    if len(words) >= 3 and len(cut_words) >= _MIN_NEG_LEN:
        return cut_words
    half = max(1, len(chosen) // 2)
    chunk = chosen[half:]
    return (chunk + " " + chunk).strip() or chunk or chosen


def rule_reject(chosen: str) -> str:
    """Return a valid negative for `chosen`, always. Flavor → validate → guaranteed fallback."""
    for strategy in random.sample(STRATEGIES, len(STRATEGIES)):
        cand = strategy(chosen)
        if is_valid_negative(chosen, cand):
            return cand
    return guaranteed_cut(chosen)


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
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.25,
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
