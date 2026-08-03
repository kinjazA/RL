"""Audit lexical overlap between the frozen evaluation questions and SFT prompts."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from heapq import nlargest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "rlhf_answers_filled.csv"
TEST_PATH = Path(__file__).with_name("sft_test_v1.json")
REPORT_PATH = Path(__file__).with_name("overlap_report_v1.csv")
TOP_K = 3
CANDIDATE_POOL = 20


def normalize(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).lower()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def ranked_matches(question: str, train_rows: Iterable[dict[str, object]]) -> list[dict[str, str | float]]:
    question_normalized = normalize(question)
    question_ngrams = char_ngrams(question_normalized)
    ngram_candidates = []
    for row in train_rows:
        ngram_candidates.append((jaccard(question_ngrams, row["question_ngrams"]), row))

    matches = []
    for ngram, row in nlargest(CANDIDATE_POOL, ngram_candidates, key=lambda item: item[0]):
        sequence = SequenceMatcher(None, question_normalized, row["normalized_question"]).ratio()
        matches.append(
            {
                "train_id": row["id"],
                "train_role": row["role"],
                "train_question": row["question"],
                "sequence_similarity": sequence,
                "ngram_jaccard": ngram,
                "max_similarity": max(sequence, ngram),
            }
        )
    return sorted(matches, key=lambda item: item["max_similarity"], reverse=True)[:TOP_K]


def main() -> None:
    with TRAIN_PATH.open(encoding="utf-8-sig", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    for row in train_rows:
        normalized = normalize(row["question"])
        row["normalized_question"] = normalized
        row["question_ngrams"] = char_ngrams(normalized)
    with TEST_PATH.open(encoding="utf-8") as handle:
        test_rows = json.load(handle)

    report_rows = []
    for test in test_rows:
        for rank, match in enumerate(ranked_matches(test["question"], train_rows), start=1):
            report_rows.append(
                {
                    "eval_id": test["id"],
                    "eval_role": test["role"],
                    "rank": rank,
                    "eval_question": test["question"],
                    **match,
                    "manual_review": "pending",
                }
            )

    fieldnames = list(report_rows[0])
    with REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    flagged = sum(row["max_similarity"] >= 0.45 for row in report_rows if row["rank"] == 1)
    print(f"Wrote {len(report_rows)} candidate matches to {REPORT_PATH}")
    print(f"Top matches at or above 0.45: {flagged}/{len(test_rows)}. Review all top matches manually.")


if __name__ == "__main__":
    main()
