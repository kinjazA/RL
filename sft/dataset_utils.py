"""Load CSV datasets and format them into Qwen2.5 ChatML for SFT."""

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

from .chatml import (
    CHATML_ASSISTANT,
    CHATML_END,
    CHATML_SYSTEM,
    CHATML_USER,
    NL,
    SYSTEM_TEXT,
    build_full,
)
from .config import (
    MAX_SEQ_LENGTH,
    OVERSAMPLE_FACTOR,
    OVERSAMPLE_SOURCES,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
)


def _format_one(row: dict) -> str:
    """Build a single ChatML-formatted training example with system prompt."""
    return build_full(row["question"], row["answer"])


def _read_formatted(csv_path: str) -> Dataset:
    """Read one CSV file and return text/source columns for training."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"question", "answer", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    texts = [_format_one(row) for _, row in df.iterrows()]
    sources = df["source"].tolist()
    return Dataset.from_dict({"text": texts, "source": sources})


def _oversample_train(train: Dataset) -> Dataset:
    """Oversample configured sources in the train split only."""
    n_before = len(train)
    for src in OVERSAMPLE_SOURCES:
        subset = train.filter(lambda x, src=src: x["source"] == src)
        if len(subset) > 0:
            for _ in range(OVERSAMPLE_FACTOR - 1):
                train = concatenate_datasets([train, subset])
    if len(train) > n_before:
        print(f"  Oversampled {', '.join(OVERSAMPLE_SOURCES)}: {n_before:,} -> {len(train):,}")
    return train


def load_and_format(csv_path: str, eval_csv_path: str | None = None) -> DatasetDict:
    """Return train/test datasets.

    Prefer passing eval_csv_path from scripts/build_sft_splits.py. If omitted,
    this preserves the old random row split behavior for quick experiments.
    """
    train_source = _read_formatted(csv_path)
    if eval_csv_path:
        train = train_source
        test = _read_formatted(eval_csv_path)
    else:
        ds = train_source.train_test_split(test_size=TRAIN_VAL_SPLIT, seed=RANDOM_SEED)
        train = ds["train"]
        test = ds["test"]

    train = _oversample_train(train)

    train = train.remove_columns(["source"])
    test = test.remove_columns(["source"])

    result = DatasetDict({"train": train, "test": test})
    print(f"Train: {len(train):,}  |  Val: {len(test):,}")
    return result


def check_seq_lengths(csv_path: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Print token-length statistics to check if MAX_SEQ_LENGTH is adequate."""
    from transformers import AutoTokenizer

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    texts = [_format_one(row) for _, row in df.iterrows()]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lengths = [len(tokenizer.encode(t)) for t in texts]

    over = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
    sorted_lengths = sorted(lengths)
    print(f"Total samples:      {len(lengths):,}")
    print(f"Max length:         {max(lengths)}")
    print(f"P95 length:         {sorted_lengths[int(len(lengths) * 0.95)]}")
    print(f"P50 length:         {sorted_lengths[len(lengths) // 2]}")
    print(f"P05 length:         {sorted_lengths[int(len(lengths) * 0.05)]}")
    print(f"> {MAX_SEQ_LENGTH} tokens:  {over} ({100 * over / len(lengths):.1f}%)")
    if over / len(lengths) > 0.05:
        print(f"\nWARNING: {over / len(lengths) * 100:.1f}% samples exceed {MAX_SEQ_LENGTH}.")
    else:
        print(f"\nOK: <5% truncated at {MAX_SEQ_LENGTH}.")
