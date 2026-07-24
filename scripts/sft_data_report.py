"""Write a compact markdown report for an SFT CSV."""

import argparse
import os

import pandas as pd

from sft_prepare_data import normalize_question


def describe_lengths(df: pd.DataFrame, col: str) -> str:
    s = df[col].astype(str).str.len()
    return (
        f"mean={s.mean():.1f}, median={s.median():.1f}, "
        f"p90={s.quantile(0.90):.1f}, p95={s.quantile(0.95):.1f}, "
        f"min={s.min()}, max={s.max()}"
    )


def build_report(df: pd.DataFrame, input_path: str) -> str:
    df = df.copy()
    if "normalized_question" not in df.columns:
        df["normalized_question"] = df["question"].map(normalize_question)
    df["q_len"] = df["question"].astype(str).str.len()
    df["a_len"] = df["answer"].astype(str).str.len()

    lines = [
        "# SFT Data Report",
        "",
        f"Input: `{input_path}`",
        f"Rows: {len(df):,}",
        "",
        "## Columns",
        "",
        ", ".join(f"`{c}`" for c in df.columns),
        "",
        "## Source Distribution",
        "",
        "```text",
        df["source"].value_counts().to_string(),
        "```",
    ]

    if "domain" in df.columns:
        lines += [
            "",
            "## Domain Distribution",
            "",
            "```text",
            df["domain"].value_counts().to_string(),
            "```",
        ]

    lines += [
        "",
        "## Lengths",
        "",
        f"Question chars: {describe_lengths(df, 'question')}",
        f"Answer chars: {describe_lengths(df, 'answer')}",
        "",
        "## Duplicate Questions",
        "",
    ]

    q_counts = df["normalized_question"].value_counts()
    repeated = q_counts[q_counts > 1]
    lines.append(f"Repeated normalized questions: {len(repeated):,}")
    lines.append(f"Rows inside repeated questions: {int(q_counts[q_counts > 1].sum()):,}")
    lines.append("")
    lines.append("Top repeated normalized questions:")
    lines.append("")
    lines.append("```text")
    for q, count in repeated.head(20).items():
        sample = df.loc[df["normalized_question"] == q, "question"].iloc[0]
        lines.append(f"{count:>3}x  {sample[:140]}")
    lines.append("```")

    suspicious = df[
        ((df["source"] == "general_oasst") & (df["q_len"] > 500) & (df["a_len"] < 200))
        | ((df["source"] == "general_oasst") & ((df["q_len"] / df["a_len"].clip(lower=1)) > 5))
    ]
    lines += [
        "",
        "## Suspicious General OASST Rows",
        "",
        f"Rows flagged by length heuristics: {len(suspicious):,}",
    ]
    if len(suspicious):
        lines += ["", "```text"]
        for _, row in suspicious.head(15).iterrows():
            q = str(row["question"]).replace("\n", " ")[:120]
            a = str(row["answer"]).replace("\n", " ")[:120]
            lines.append(f"q_len={row['q_len']:<4} a_len={row['a_len']:<4} Q={q} | A={a}")
        lines.append("```")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join("data", "sft_clean_v2.csv"))
    parser.add_argument("--output", default=os.path.join("data", "sft_data_report_v2.md"))
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    report = build_report(df, args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
