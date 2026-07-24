"""
Pre-training sanity checks for SFT v2.

Verifies the two "silent failure" risks before spending ~50 min training:
  1. DataCollatorForCompletionOnlyLM imports cleanly on this TRL version
  2. The response_template tokens actually match in a tokenized sample,
     so loss masking is genuinely applied (and -100 lands on the right tokens)

Run order:  verify_collator.py  →  (if both pass)  train.py
                      ↓
           labels_mask_test.py  (optional, deeper proof)

Usage (from project root):
    python -m sft.checks.verify_collator
    python -m sft.checks.verify_collator --csv_path path/to/your/train.csv
"""

import argparse
import os
import sys

# Ensure we can run from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def check_import() -> bool:
    """Check 1: trl version compatibility for the completion-only collator."""
    print("\n" + "=" * 64)
    print("CHECK 1 — DataCollatorForCompletionOnlyLM imports cleanly")
    print("=" * 64)

    import trl
    print(f"  trl version: {trl.__version__}")

    collator_cls = None
    tried = []

    for path in ["trl.DataCollatorForCompletionOnlyLM",
                 "trl.trainer.utils.DataCollatorForCompletionOnlyLM"]:
        mod_name, attr = path.rsplit(".", 1)
        tried.append(path)
        try:
            mod = __import__(mod_name, fromlist=[attr])
            collator_cls = getattr(mod, attr)
            print(f"  ✓ imported via: {path}")
            break
        except (ImportError, AttributeError) as e:
            print(f"  ✗ failed via:   {path}  ({e.__class__.__name__})")

    if collator_cls is None:
        print("\n  ❌ RESULT: no working import path found. Tried:")
        for p in tried:
            print(f"       - {p}")
        print("  👉 Fix: either pin trl, or switch to SFTConfig "
              "(completion_only_loss / assistant_only_loss).")
        return False

    # deprecation warning?
    import inspect
    src_file = inspect.getfile(collator_cls)
    print(f"  collator defined in: {src_file}")
    print("  ✅ RESULT: collator usable.")
    return True


def check_masking(csv_path: str) -> bool:
    """Check 2: response_template tokens actually match → masking applies."""
    print("\n" + "=" * 64)
    print("CHECK 2 — response_template token match (loss masking real?)")
    print("=" * 64)

    # Locate the exact objects train.py uses
    from sft.chatml import CHATML_ASSISTANT, NL
    from sft.dataset_utils import _format_one
    from sft.config import MODEL_NAME
    from transformers import AutoTokenizer

    # train.py's response template
    response_template = f"{CHATML_ASSISTANT}{NL}"
    print(f"  response_template (str): {response_template!r}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the template ALONE — this is what the collator matches against
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"  template token ids: {template_ids}")
    print(f"  template tokens:    {tokenizer.convert_ids_to_tokens(template_ids)}")

    # Build ONE real training example with the v2 formatter
    import pandas as pd
    if not os.path.isfile(csv_path):
        print(f"  ⚠️  CSV not found: {csv_path} — using a synthetic sample.")
        row = {"question": "Why do you want to leave your current job?",
               "answer": "I'm looking for more growth in ML.",
               "source": "hr_interview"}
    else:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        row = df.iloc[0].to_dict()
    text = _format_one(row)
    print(f"\n  sample text (first 300 chars):\n  {text[:300]}...")

    full_ids = tokenizer.encode(text, add_special_tokens=False)

    # Does the template token subsequence appear as a contiguous slice in
    # the full sample? This is exactly what the collator looks for.
    def subseq(hay, needle):
        n = len(needle)
        for i in range(len(hay) - n + 1):
            if hay[i:i + n] == needle:
                return i
        return -1

    pos = subseq(full_ids, template_ids)
    print(f"  template found in sample at position: {pos}")
    if pos == -1:
        print("\n  ❌ RESULT: response_template tokens did NOT match as a "
              "contiguous slice.")
        print("     → loss masking sees no anchor → ALL tokens get loss.")
        print("     → You'd be silently running v1 behaviour + NEFTune.")
        print("  👉 Likely cause: tokenizer merges <|im_start|> and 'assistant' "
              "into one token. Use a token-id-level template instead.")
        return False

    mask_start = pos + len(template_ids)
    n_assistant_tokens = len(full_ids) - mask_start
    print(f"  mask boundary (first non-masked token idx): {mask_start}")
    print(f"  # of tokens that KEEP loss (should be assistant answer + EOS): "
          f"{n_assistant_tokens}")
    print(f"  # tokens MASKED to -100 (system + user + template): "
          f"{mask_start}")

    # Sanity: assistant tokens in the tail should decode to the answer
    answer_tail = tokenizer.decode(full_ids[mask_start:], skip_special_tokens=True)
    expected = str(row.get("answer", ""))
    print(f"  tail decodes to:  {answer_tail[:120]!r}")
    print(f"  expected answer:  {expected[:120]!r}")
    if expected and expected.strip() in answer_tail:
        print("\n  ✅ RESULT: template matched + tail is the assistant answer. "
              "Loss masking will apply correctly.")
        return True

    print("\n  ⚠️  template matched, but tail does not equal the answer — "
          "inspect manually before trusting.")
    return False


def main():
    default_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "train.csv"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_path", default=default_csv,
                        help=f"path to train.csv (default: {default_csv})")
    args = parser.parse_args()

    ok1 = check_import()
    ok2 = check_masking(args.csv_path) if ok1 else (
        print("\n  (skipping CHECK 2 because CHECK 1 failed)"), False)[1]

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  CHECK 1 (collator import): {'PASS' if ok1 else 'FAIL'}")
    print(f"  CHECK 2 (mask anchor):     {'PASS' if ok2 else 'FAIL'}")
    if ok1 and ok2:
        print("\n  🟢 Both checks passed — safe to run SFT v2 training.")
        sys.exit(0)
    else:
        print("\n  🔴 At least one check failed — fix before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
