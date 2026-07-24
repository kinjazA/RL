"""Raw-data audit for train.csv. Identifies actual quality issues before cleaning.

Run from RL/ directory:
    python scripts/sft_audit_raw.py --input data/train.csv
"""

import argparse
import os
import re
from collections import Counter

import pandas as pd


def read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["source"] = df["source"].astype(str)
    return df


HTML_TAGS = re.compile(r"<[a-z/][^>]*>", re.IGNORECASE)
HTML_ENTITIES = re.compile(r"&(?:#\d+|[a-z]+);", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+")
WHITESPACE_RUN = re.compile(r"[ \t]{2,}|\n{3,}")
TRAILING_SPACES = re.compile(r" +\n")
MARKDOWN = re.compile(r"\n#{1,6}\s|\n\*\s|\n-\s|\n\d\.\s")


def detect_issues(text: str) -> list[str]:
    issues = []
    if HTML_TAGS.search(text):
        issues.append("html_tag")
    if HTML_ENTITIES.search(text):
        issues.append("html_entity")
    if URL_PATTERN.search(text):
        issues.append("url")
    if WHITESPACE_RUN.search(text) or TRAILING_SPACES.search(text):
        issues.append("whitespace_issue")
    if text.lower().startswith(("as an ai", "i am an ai", "as a language model")):
        issues.append("ai_refusal")
    if "```" in text:
        issues.append("code_fences")
    if MARKDOWN.search(text):
        issues.append("markdown_lists")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join("data", "train.csv"))
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--sample_n", type=int, default=5)
    args = parser.parse_args()

    df = read(args.input)
    print(f"=== Raw audit: {args.input} ===")
    print(f"Total rows: {len(df):,}\n")

    # Source distribution
    print("Source distribution:")
    print(df["source"].value_counts().to_string())
    print()

    # Length stats
    df["q_len"] = df["question"].str.len()
    df["a_len"] = df["answer"].str.len()
    print("Answer length (chars) by source:")
    print(df.groupby("source")["a_len"].describe()[["mean", "50%", "min", "max"]].round(0).to_string())
    print()

    # Empty / near-empty
    empty_q = (df["question"].str.strip() == "") | (df["question"].str.lower().isin(["nan", "none"]))
    empty_a = (df["answer"].str.strip() == "") | (df["answer"].str.lower().isin(["nan", "none"]))
    print(f"Empty/'nan' question rows: {empty_q.sum()}")
    print(f"Empty/'nan' answer rows:   {empty_a.sum()}")

    # Answer very short (<30 chars) — suspicious
    very_short = df["a_len"] < 30
    print(f"Answers < 30 chars:        {very_short.sum()}")
    # Answer extremely long (>3000)
    very_long = df["a_len"] > 3000
    print(f"Answers > 3000 chars:      {very_long.sum()}\n")

    # Reversed: long question / short answer bizarrely
    reversed_rows = (df["q_len"] > 500) & (df["a_len"] < 200)
    print(f"Suspected reversed (q>500, a<200): {reversed_rows.sum()}")

    # Exact duplicates
    qd = df.duplicated(["question", "answer"], keep="first").sum()
    print(f"Exact duplicate (question,answer) rows: {qd}\n")

    # Per-source issue tally
    print("Detecting content issues per answer (html/fence/url/refusal/etc) ...")
    issue_rows = df["answer"].map(detect_issues)
    df["issue_tags"] = issue_rows.map(lambda L: ";".join(L)).fillna("")
    flat = []
    for tags in issue_rows:
        flat.extend(tags)
    if flat:
        print("Answer content issue counts:")
        print(pd.Series(flat).value_counts().to_string())
    else:
        print("No content issues flagged in answers.")

    issue_rows_q = df["question"].map(detect_issues)
    qflat = []
    for tags in issue_rows_q:
        qflat.extend(tags)
    if qflat:
        print("\nQuestion content issue counts:")
        print(pd.Series(qflat).value_counts().to_string())
    print()

    # Question duplicates (normalized)
    def norm(t):
        t = re.sub(r"[^a-z0-9 ]+", "", str(t).lower().strip())
        return re.sub(r"\s+", " ", t)
    df["norm_q"] = df["question"].map(norm)
    dup_groups = df["norm_q"].value_counts()
    heavy = dup_groups[dup_groups >= 5]
    print(f"Normalized-question groups appearing >=5 times: {len(heavy)}")
    if len(heavy):
        print("Top repeated question groups:")
        print(heavy.head(15).to_string())
    print()

    # Show sample rows that have issues
    flagged = df[df["issue_tags"] != ""]
    if len(flagged) and args.sample_n:
        print(f"\n=== Sample flagged answers (n={min(args.sample_n, len(flagged))}) ===")
        for _, r in flagged.head(args.sample_n).iterrows():
            q = r["question"][:100].encode("ascii", "replace").decode("ascii")
            a = r["answer"][:300].encode("ascii", "replace").decode("ascii")
            print(f"[{r['source']}] issues={r['issue_tags']}")
            print(f"Q: {q}")
            print(f"A: {a}")
            print("-" * 60)

    # Write the flagged rows to disk for review
    if len(flagged):
        out = os.path.join(args.out_dir, "sft_raw_audit_flagged.csv")
        cols = ["question", "answer", "source", "a_len", "issue_tags", "norm_q"]
        flagged[cols].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nWrote flagged rows -> {out}")


if __name__ == "__main__":
    main()