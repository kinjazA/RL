"""
Load CSV dataset and format into Qwen2.5 ChatML for SFT.
Input:  train.csv  (question, answer, source)
Output: DatasetDict with 'train' and 'test' splits, each sample has a 'text' field.
"""

import pandas as pd
from datasets import Dataset, DatasetDict

from .config import TRAIN_VAL_SPLIT, RANDOM_SEED

CHATML_USER = "<|im_start|>user"
CHATML_ASSISTANT = "<|im_start|>assistant"
CHATML_END = "<|im_end|>"
NL = "\n"


def _format_one(row: dict) -> str:
    """Build a single ChatML-formatted training example."""
    question = str(row["question"])
    answer = str(row["answer"])
    return (
        f"{CHATML_USER}{NL}{question}{CHATML_END}{NL}"
        f"{CHATML_ASSISTANT}{NL}{answer}{CHATML_END}{NL}"
    )


def load_and_format(csv_path: str) -> DatasetDict:
    """Read merged CSV and return train/val DatasetDict with 'text' column."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    texts = [_format_one(row) for _, row in df.iterrows()]

    ds = Dataset.from_dict({"text": texts})
    ds = ds.train_test_split(test_size=TRAIN_VAL_SPLIT, seed=RANDOM_SEED)

    print(f"Train: {len(ds['train']):,}  |  Val: {len(ds['test']):,}")
    return ds
