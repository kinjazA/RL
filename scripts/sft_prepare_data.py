"""Prepare cleaned SFT data for the interview-answer SFT stage.

This script is intentionally conservative: it removes obvious bad rows,
adds source-derived metadata, and caps repeated normalized questions. All
thresholds are CLI flags so data decisions stay reviewable in a team workflow.
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd


SOURCE_DOMAIN = {
    "career_qa": "career",
    "local_interview_qa": "behavioral",
    "hr_interview": "behavioral",
    "ml_interview": "ml",
    "ds_qa_treasury": "ds",
    "se_interview": "se",
    "general_alpaca": "general",
    "general_oasst": "general",
}

SOURCE_ANSWER_TYPE = {
    "career_qa": "role_overview",
    "local_interview_qa": "behavioral_star",
    "hr_interview": "behavioral_star",
    "ml_interview": "technical_explanation",
    "ds_qa_treasury": "technical_explanation",
    "se_interview": "technical_explanation",
    "general_alpaca": "general_qa",
    "general_oasst": "general_qa",
}


def normalize_question(text: str) -> str:
    """Normalize a question for duplicate grouping."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"question", "answer", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["domain"] = df["source"].map(SOURCE_DOMAIN).fillna("unknown")
    df["answer_type"] = df["source"].map(SOURCE_ANSWER_TYPE).fillna("unknown")
    df["normalized_question"] = df["question"].map(normalize_question)
    df["q_len"] = df["question"].str.len()
    df["a_len"] = df["answer"].str.len()
    return df


def flag_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    reasons: dict[int, list[str]] = defaultdict(list)

    empty = (df["question"] == "") | (df["answer"] == "")
    for idx in df.index[empty]:
        reasons[idx].append("empty_question_or_answer")

    short_answer = df["a_len"] < args.min_answer_chars
    for idx in df.index[short_answer]:
        reasons[idx].append("answer_too_short")

    long_answer = df["a_len"] > args.max_answer_chars
    for idx in df.index[long_answer]:
        reasons[idx].append("answer_too_long")

    exact_dup = df.duplicated(["question", "answer"], keep="first")
    for idx in df.index[exact_dup]:
        reasons[idx].append("exact_duplicate_question_answer")

    reversed_oasst = (
        (df["source"] == "general_oasst")
        & (df["q_len"] > args.reversed_question_chars)
        & (df["a_len"] < args.reversed_answer_chars)
    )
    for idx in df.index[reversed_oasst]:
        reasons[idx].append("suspected_oasst_reversed_or_truncated")

    ratio = df["q_len"] / df["a_len"].clip(lower=1)
    high_ratio_oasst = (df["source"] == "general_oasst") & (ratio > args.max_question_answer_ratio)
    for idx in df.index[high_ratio_oasst]:
        reasons[idx].append("oasst_question_answer_ratio_high")

    duplicate_rank = df.groupby("normalized_question").cumcount()
    over_cap = duplicate_rank >= args.max_per_question
    for idx in df.index[over_cap]:
        reasons[idx].append("duplicate_question_over_cap")

    return pd.Series({idx: ";".join(vals) for idx, vals in reasons.items()}, dtype="object")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join("data", "train.csv"))
    parser.add_argument("--output", default=os.path.join("data", "sft_clean_v2.csv"))
    parser.add_argument("--removed_output", default=os.path.join("data", "sft_removed_v2.csv"))
    parser.add_argument("--max_per_question", type=int, default=3)
    parser.add_argument("--min_answer_chars", type=int, default=50)
    parser.add_argument("--max_answer_chars", type=int, default=3000)
    parser.add_argument("--reversed_question_chars", type=int, default=500)
    parser.add_argument("--reversed_answer_chars", type=int, default=200)
    parser.add_argument("--max_question_answer_ratio", type=float, default=5.0)
    args = parser.parse_args()

    df = add_metadata(read_csv(args.input))
    removal_reasons = flag_rows(df, args)
    df["removal_reason"] = df.index.map(removal_reasons).fillna("")

    removed = df[df["removal_reason"] != ""].copy()
    kept = df[df["removal_reason"] == ""].copy()

    output_cols = ["question", "answer", "source", "domain", "answer_type", "normalized_question"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept[output_cols].to_csv(args.output, index=False, encoding="utf-8-sig")
    removed.to_csv(args.removed_output, index=False, encoding="utf-8-sig")

    print(f"Input rows:   {len(df):,}")
    print(f"Kept rows:    {len(kept):,} -> {args.output}")
    print(f"Removed rows: {len(removed):,} -> {args.removed_output}")
    if len(removed):
        print("\nRemoved by reason:")
        exploded = removed["removal_reason"].str.split(";").explode()
        print(exploded.value_counts().to_string())
    print("\nKept by source:")
    print(kept["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
