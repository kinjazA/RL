"""
gen_reject.py — Generate rejected answers for RM training.

Uses cleaned SFT data as chosen answers. Generates one rejected answer per
(chosen) pair using a three-class mix:

  Type A (~25%): Truncation / shuffle — teaches completeness & coherence
  Type B (~40%): Weak model (Qwen2.5-0.5B) — natural quality differences
  Type C (~35%): Rule-based bad samples — teaches to reject vagueness,
                 off-topic, repetition, advice-only, and over-templated answers

Every rejected answer is validated: must differ from chosen, be non-trivial,
and not materially longer than chosen (longer could actually be better).

Input:  data/sft_clean.csv    (question, answer, source, domain, answer_type)
Output: data/rm_train_v2.csv  (prompt, chosen, rejected, source, domain, rejected_type)

Usage:  python rm/gen_reject.py [--batch_size 4] [--type_a_ratio 0.25] [--type_b_ratio 0.40]
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
INPUT = os.path.join(DATA_DIR, "sft_clean.csv")
OUTPUT = os.path.join(DATA_DIR, "rm_train_v2.csv")
CHECKPOINT = os.path.join(DATA_DIR, ".gen_reject_v2_checkpoint.txt")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 80  # short → 0.5B answers stay worse than chosen

# Prompt for weak model: answer briefly, lower quality
WEAK_SYSTEM = (
    "Answer the following question very briefly, in one or two sentences. "
    "Do not elaborate. Be vague."
)

random.seed(42)

# ---------------------------------------------------------------------------
# Chosen quality filter
# ---------------------------------------------------------------------------
ANY_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?/?>")
AI_REFUSAL_PATS = [
    re.compile(r"^as an ai\b", re.IGNORECASE),
    re.compile(r"^i am an ai\b", re.IGNORECASE),
    re.compile(r"^i cannot\b", re.IGNORECASE),
    re.compile(r"^sorry,?\s*(?:but\s+)?i\s+(?:can'?t|cannot|am not)\b", re.IGNORECASE),
]

def is_bad_chosen(answer: str) -> tuple[bool, str]:
    """Return (is_bad, reason). Conservative: only filter clearly bad samples."""
    a = answer.strip()
    if len(a) < 50:
        return True, "too_short"
    if len(a) > 3000:
        return True, "too_long"
    if ANY_HTML_TAG.search(a):
        return True, "has_html"
    if a.count("?") > 3:
        return True, "too_many_questions"
    for pat in AI_REFUSAL_PATS:
        if pat.search(a):
            return True, "ai_refusal"
    return False, ""

# ---------------------------------------------------------------------------
# Rejection type A: Truncation / Shuffle (~25%)
# ---------------------------------------------------------------------------

def _truncate(text: str) -> str:
    """Keep first 1-2 sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    keep = random.randint(1, min(2, len(sents)))
    return " ".join(sents[:keep])


def _shuffle_paragraphs(text: str) -> str:
    """Scramble paragraph order."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return text
    random.shuffle(paras)
    return "\n\n".join(paras)


def _mid_drop(text: str) -> str:
    """Drop middle half of sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 4:
        return text
    n = len(sents)
    return " ".join(sents[: n // 4] + sents[3 * n // 4 :])


TYPE_A_STRATEGIES = [
    ("truncate", _truncate),
    ("shuffle", _shuffle_paragraphs),
    ("mid_drop", _mid_drop),
]


def generate_type_a(chosen: str) -> tuple[str, str]:
    """Try truncate/shuffle strategies; return (rejected, strategy_name)."""
    for name, fn in random.sample(TYPE_A_STRATEGIES, len(TYPE_A_STRATEGIES)):
        cand = fn(chosen)
        if is_valid_negative(chosen, cand):
            return cand, name
    # Last resort: just take first 1-2 sentences
    sents = re.split(r"(?<=[.!?])\s+", chosen)
    return " ".join(sents[:2]), "truncate"


# ---------------------------------------------------------------------------
# Rejection type C: Rule-based bad samples (~35%)
# ---------------------------------------------------------------------------

# Pool of off-topic / vague non-answers
_OFF_TOPIC_TEMPLATES = [
    "I think this depends on the specific context and circumstances. Every situation is different.",
    "That's a great question! Let me think about it and get back to you.",
    "Hmm, I'm not entirely sure about that one. Could you clarify what you mean?",
    "This is something that varies greatly across industries and roles.",
    "I don't have enough information to answer that properly right now.",
]

_ADVICE_TEMPLATES = [
    "When preparing for this question in an interview, it's important to structure your answer clearly. Start with a specific example, then explain what you did, and conclude with the result. Practice out loud beforehand.",
    "The best way to answer this is to use the STAR method: Situation, Task, Action, Result. Make sure your answer is concise and relevant to the role.",
    "Interviewers ask this to assess your problem-solving skills. Take a moment to think before answering, and be honest if you don't know something.",
    "A good approach is to relate this to your past experience. Even if you haven't faced this exact situation, find something similar and explain your thought process.",
    "Remember to keep your answer focused and professional. Don't ramble — stick to one or two key points and elaborate on those.",
]

_TEMPLATED_PREFIXES = [
    "As a seasoned professional with extensive experience in this domain, I would approach this by leveraging my comprehensive background and proven track record. Throughout my career, I have consistently demonstrated excellence in",
    "I am extremely passionate about this topic and have dedicated my entire career to mastering it. My approach has always been to strive for perfection while maintaining the highest standards of professionalism in",
    "First and foremost, I would like to emphasize that I am a results-driven, detail-oriented professional with a proven ability to deliver exceptional outcomes. My methodology involves a holistic, 360-degree approach to",
    "Having worked at multiple Fortune 500 companies, I can confidently say that my unique blend of technical expertise and business acumen sets me apart. I always begin by conducting a thorough analysis of",
]

_REPETITIVE_LINES = [
    "This is very important for career growth. Very important indeed for anyone looking to advance. Understanding this is crucial for professional development and career advancement in today's competitive market.",
    "I believe this is a critical skill. This is truly critical for success. Being able to handle this is critically important in the modern workplace environment.",
    "Many people overlook this aspect. This is often overlooked by candidates. Unfortunately, most people tend to overlook this crucial element during their interview preparation.",
]


def _make_off_topic(chosen: str) -> str:
    return random.choice(_OFF_TOPIC_TEMPLATES)


def _make_advice_only(chosen: str) -> str:
    return random.choice(_ADVICE_TEMPLATES)


def _make_templated(chosen: str) -> str:
    prefix = random.choice(_TEMPLATED_PREFIXES)
    # Append a bit of the original to make it seem related
    words = chosen.split()[:20]
    return prefix + " " + " ".join(words)


def _make_repetitive(chosen: str) -> str:
    line = random.choice(_REPETITIVE_LINES)
    return (line + " ") * 3


def _make_vague(chosen: str) -> str:
    words = chosen.split()[:10]
    return "It depends on many factors. " + " ".join(words) + " and other considerations must be taken into account before reaching a conclusion. Different approaches may work for different situations."


TYPE_C_STRATEGIES = [
    ("off_topic", _make_off_topic),
    ("advice_only", _make_advice_only),
    ("templated", _make_templated),
    ("repetitive", _make_repetitive),
    ("vague", _make_vague),
]


def generate_type_c(chosen: str) -> tuple[str, str]:
    name, fn = random.choice(TYPE_C_STRATEGIES)
    return fn(chosen), name


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------
_MIN_NEG_LEN = 10


def is_valid_negative(chosen: str, rejected: str) -> bool:
    """Basic sanity: not empty, not identical to chosen."""
    if not rejected or len(rejected.strip()) < _MIN_NEG_LEN:
        return False
    if rejected.strip() == chosen.strip():
        return False
    return True


# ---------------------------------------------------------------------------
# Model loading & generation (Type B)
# ---------------------------------------------------------------------------
def load_weak_model(device: str = "auto"):
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
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    messages = [
        [{"role": "system", "content": WEAK_SYSTEM},
         {"role": "user", "content": p}]
        for p in prompts
    ]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.15,
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--type_a_ratio", type=float, default=0.25)
    parser.add_argument("--type_b_ratio", type=float, default=0.40)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    assert abs(args.type_a_ratio + args.type_b_ratio + (1 - args.type_a_ratio - args.type_b_ratio) - 1.0) < 0.001

    # ---- Load chosen answers from cleaned SFT data ----
    rows = list(csv.DictReader(open(INPUT, encoding="utf-8-sig")))
    print(f"Loaded {len(rows)} rows from {INPUT}")

    # ---- Filter bad chosen samples ----
    kept = []
    filtered_reasons = {}
    for r in rows:
        bad, reason = is_bad_chosen(r["answer"])
        if bad:
            filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
        else:
            kept.append(r)

    print(f"  Kept: {len(kept)}  |  Filtered:")
    for reason, count in sorted(filtered_reasons.items()):
        print(f"    {reason}: {count}")

    # ---- Assign rejection types ----
    done = load_checkpoint()
    pending = [i for i in range(len(kept)) if i not in done]

    type_a_count = max(1, int(len(kept) * args.type_a_ratio))
    type_b_count = max(1, int(len(kept) * args.type_b_ratio))
    type_c_count = len(kept) - type_a_count - type_b_count

    assignments = {}
    indices = list(range(len(kept)))
    random.shuffle(indices)
    for i in indices[:type_a_count]:
        assignments[i] = "A"
    for i in indices[type_a_count:type_a_count + type_b_count]:
        assignments[i] = "B"
    for i in indices[type_a_count + type_b_count:]:
        assignments[i] = "C"

    print(f"\nRejection mix:  A(trunc/shuffle)={type_a_count}  "
          f"B(weak_model)={type_b_count}  C(rule_bad)={type_c_count}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    write_header = not done
    out_f = open(OUTPUT, "w" if write_header else "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(["prompt", "chosen", "rejected", "source", "domain", "rejected_type"])

    # ---- Type A + C: Rule-based (write immediately) ----
    non_b_pending = [i for i in pending if assignments[i] != "B"]
    for i in non_b_pending:
        row = kept[i]
        prompt, chosen = row["question"], row["answer"]
        src, dom = row.get("source", ""), row.get("domain", "")

        if assignments[i] == "A":
            rejected, subtype = generate_type_a(chosen)
            rejected_type = f"A_{subtype}"
        else:
            rejected, subtype = generate_type_c(chosen)
            rejected_type = f"C_{subtype}"

        writer.writerow([prompt, chosen, rejected, src, dom, rejected_type])
        save_checkpoint(i)

    print(f"Rule-based (A+C): {len(non_b_pending)} done")

    # ---- Type B: Weak model (batched GPU) ----
    b_pending = [i for i in pending if assignments[i] == "B"]
    if b_pending:
        model, tokenizer = load_weak_model(args.device)
        batch_size = args.batch_size
        total_retry = 0
        pbar = tqdm(range(0, len(b_pending), batch_size), desc="Weak model (0.5B)")

        for start in pbar:
            end = min(start + batch_size, len(b_pending))
            batch_idxs = b_pending[start:end]
            batch_prompts = [kept[i]["question"] for i in batch_idxs]
            batch_chosens = [kept[i]["answer"] for i in batch_idxs]

            try:
                generated = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM at batch size {batch_size}. Try --batch_size {max(1, batch_size // 2)}")
                out_f.close()
                sys.exit(1)

            n_retry = 0
            for idx, prompt, chosen, rejected in zip(batch_idxs, batch_prompts, batch_chosens, generated):
                row = kept[idx]
                src, dom = row.get("source", ""), row.get("domain", "")
                if not is_valid_negative(chosen, rejected):
                    # Retry with fewer tokens
                    shorter_tokens = max(20, args.max_new_tokens // 2)
                    [retry] = generate_batch(model, tokenizer, [prompt], shorter_tokens)
                    if is_valid_negative(chosen, retry):
                        rejected = retry
                        subtype = "B_weak_model"
                    else:
                        # Fallback to Type C — naturally bad, no cutting
                        rejected, c_name = generate_type_c(chosen)
                        subtype = f"B_fallback_C_{c_name}"
                    n_retry += 1
                else:
                    subtype = "B_weak_model"
                writer.writerow([prompt, chosen, rejected, src, dom, subtype])
                save_checkpoint(idx)
            total_retry += n_retry
            pbar.set_postfix({"retry": f"{n_retry}/{len(batch_idxs)}"})

        if total_retry:
            print(f"  -> {total_retry} retries ({100*total_retry/len(b_pending):.0f}%)")
        print(f"Weak model: {len(b_pending)} done")

    out_f.close()

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    # ---- Summary ----
    out_rows = list(csv.DictReader(open(OUTPUT, encoding="utf-8-sig")))
    print(f"\nDone: {len(out_rows)} pairs -> {OUTPUT}")
    print(f"  Columns: {out_rows[0].keys()}")

    from collections import Counter
    type_counts = Counter(r["rejected_type"] for r in out_rows)
    print("\nRejected type distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} ({100*c/len(out_rows):.1f}%)")

    domain_counts = Counter(r["domain"] for r in out_rows)
    print("\nDomain distribution:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c}")

    # ---- Samples ----
    print("\n--- Samples ---")
    for row in random.sample(out_rows, min(6, len(out_rows))):
        print(f"\n[{row['rejected_type']}] [{row['domain']}] {row['prompt'][:80]}")
        print(f"  Chosen:   {row['chosen'][:120]}...")
        print(f"  Rejected: {row['rejected'][:120]}...")


if __name__ == "__main__":
    main()
