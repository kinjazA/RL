"""
Load CSV dataset and format into Qwen2.5 ChatML for SFT.
Input:  train.csv  (question, answer, source)
Output: DatasetDict with 'train' and 'test' splits, each sample has a 'text' field.

v2 improvements:
  - System prompt in every sample (role-locks the model)
  - Oversampling of underrepresented domains (SE, DS)
  - check_seq_lengths() utility to inspect truncation ratio
"""

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

from .config import (
    TRAIN_VAL_SPLIT,
    RANDOM_SEED,
    MAX_SEQ_LENGTH,
    OVERAMPLE_SOURCES,
    OVERAMPLE_FACTOR,
)

CHATML_SYSTEM = "<|im_start|>system"
CHATML_USER = "<|im_start|>user"
CHATML_ASSISTANT = "<|im_start|>assistant"
CHATML_END = "<|im_end|>"
NL = "\n"

SYSTEM_TEXT = (
    "You are a job candidate in an interview. "
    "Answer in first person with specific examples. "
    "Be concise, factual, and professional."
)


def _format_one(row: dict) -> str:
    """Build a single ChatML-formatted training example with system prompt."""
    question = str(row["question"])
    answer = str(row["answer"])
    return (
        f"{CHATML_SYSTEM}{NL}{SYSTEM_TEXT}{CHATML_END}{NL}"
        f"{CHATML_USER}{NL}{question}{CHATML_END}{NL}"
        f"{CHATML_ASSISTANT}{NL}{answer}{CHATML_END}{NL}"
    )


def load_and_format(csv_path: str) -> DatasetDict:
    """Read merged CSV, oversample underrepresented domains, return DatasetDict."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    texts = [_format_one(row) for _, row in df.iterrows()]
    sources = df["source"].tolist()

    ds = Dataset.from_dict({"text": texts, "source": sources})
    ds = ds.train_test_split(test_size=TRAIN_VAL_SPLIT, seed=RANDOM_SEED)

    # ---- Oversample underrepresented domains (train set only, no leakage) ----
    train = ds["train"]
    n_before = len(train)
    for src in OVERAMPLE_SOURCES:
        subset = train.filter(lambda x, src=src: x["source"] == src)
        if len(subset) > 0:
            for _ in range(OVERAMPLE_FACTOR - 1):
                train = concatenate_datasets([train, subset])
    if len(train) > n_before:
        print(f"  Oversampled {', '.join(OVERAMPLE_SOURCES)}: {n_before:,} → {len(train):,}")

    # Strip source column — only text goes to the trainer
    train = train.remove_columns(["source"])
    test = ds["test"].remove_columns(["source"])

    result = DatasetDict({"train": train, "test": test})
    print(f"Train: {len(train):,}  |  Val: {len(test):,}")
    return result


def check_seq_lengths(csv_path: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Print token-length statistics to check if MAX_SEQ_LENGTH is adequate.

    Usage (from project root):
        python -c "from sft.dataset_utils import check_seq_lengths; check_seq_lengths('data/train.csv')"
    """
    from transformers import AutoTokenizer

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts = [_format_one(row) for _, row in df.iterrows()]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lengths = [len(tokenizer.encode(t)) for t in texts]

    over = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
    print(f"Total samples:      {len(lengths):,}")
    print(f"Max length:         {max(lengths)}")
    print(f"P95 length:         {sorted(lengths)[int(len(lengths) * 0.95)]}")
    print(f"P50 length:         {sorted(lengths)[len(lengths) // 2]}")
    print(f"P05 length:         {sorted(lengths)[int(len(lengths) * 0.05)]}")
    print(f"> {MAX_SEQ_LENGTH} tokens:  {over} ({100 * over / len(lengths):.1f}%)")
    if over / len(lengths) > 0.05:
        print(f"\n  WARNING: {over/len(lengths)*100:.1f}% samples exceed {MAX_SEQ_LENGTH} — consider raising it.")
    else:
        print(f"\n  OK: <5% truncated at {MAX_SEQ_LENGTH}.")
