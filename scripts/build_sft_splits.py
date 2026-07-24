"""Build train/eval/test SFT splits with normalized-question group isolation."""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import pandas as pd

from sft_prepare_data import normalize_question


def group_domain(rows: pd.DataFrame) -> str:
    if "domain" not in rows.columns:
        return "unknown"
    return Counter(rows["domain"]).most_common(1)[0][0]


def assign_groups(df: pd.DataFrame, eval_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    groups_by_domain: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, rows in df.groupby("normalized_question", sort=False):
        groups_by_domain[group_domain(rows)].append((key, len(rows)))

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for domain, groups in groups_by_domain.items():
        rng.shuffle(groups)
        n_groups = len(groups)
        n_test = round(n_groups * test_frac)
        n_eval = round(n_groups * eval_frac)

        if n_groups >= 10 and test_frac > 0:
            n_test = max(1, n_test)
        if n_groups >= 10 and eval_frac > 0:
            n_eval = max(1, n_eval)

        for idx, (key, _) in enumerate(groups):
            if idx < n_test:
                split = "test"
            elif idx < n_test + n_eval:
                split = "eval"
            else:
                split = "train"
            assignments[key] = split
    return assignments


def write_split(df: pd.DataFrame, split: str, out_dir: str, prefix: str, suffix: str) -> str:
    out = df[df["split"] == split].drop(columns=["split"])
    path = os.path.join(out_dir, f"{prefix}_{split}_{suffix}.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join("data", "sft_clean_v2.csv"))
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--prefix", default="sft")
    parser.add_argument("--suffix", default="v2")
    parser.add_argument("--eval_frac", type=float, default=0.08)
    parser.add_argument("--test_frac", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if "normalized_question" not in df.columns:
        df["normalized_question"] = df["question"].map(normalize_question)

    assignments = assign_groups(df, args.eval_frac, args.test_frac, args.seed)
    df["split"] = df["normalized_question"].map(assignments)

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        "train": write_split(df, "train", args.out_dir, args.prefix, args.suffix),
        "eval": write_split(df, "eval", args.out_dir, args.prefix, args.suffix),
        "test": write_split(df, "test", args.out_dir, args.prefix, args.suffix),
    }

    manifest = {
        "input": args.input,
        "seed": args.seed,
        "eval_frac": args.eval_frac,
        "test_frac": args.test_frac,
        "paths": paths,
        "rows": {split: int((df["split"] == split).sum()) for split in ["train", "eval", "test"]},
        "groups": {
            split: int(df.loc[df["split"] == split, "normalized_question"].nunique())
            for split in ["train", "eval", "test"]
        },
    }
    manifest_path = os.path.join(args.out_dir, f"{args.prefix}_split_manifest_{args.suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote splits:")
    for split, path in paths.items():
        rows = manifest["rows"][split]
        groups = manifest["groups"][split]
        print(f"  {split:5s} rows={rows:5d} groups={groups:5d} path={path}")
    print(f"Manifest: {manifest_path}")

    leaked = df.groupby("normalized_question")["split"].nunique()
    leaked = leaked[leaked > 1]
    if len(leaked):
        raise RuntimeError(f"Found {len(leaked)} normalized questions crossing splits.")


if __name__ == "__main__":
    main()
