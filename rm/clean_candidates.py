"""Remove duplicate and visibly degenerate SFT candidates before judging."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEGENERATE = re.compile(
    r"^(?:creampie|tacos|pancake|utrecht|筀|鲯骨|螬+|(?:-+\s*){3,})$",
    re.IGNORECASE,
)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def invalid_sample(answer: str, min_chars: int, max_chars: int) -> str | None:
    text = normalize(answer)
    if not text:
        return "empty"
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if DEGENERATE.fullmatch(text):
        return "degenerate_token"
    words = text.split()
    if len(words) >= 3 and words[-1] == words[-2] == words[-3]:
        return "repeated_span"
    meaningful = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", text))
    if meaningful < max(20, min_chars // 2):
        return "not_meaningful"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_chars", type=int, default=40)
    parser.add_argument("--max_chars", type=int, default=1200)
    args = parser.parse_args()

    seen_ids: set[str] = set()
    seen_answers: set[str] = set()
    kept: list[dict] = []
    dropped = Counter()
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            key = f"{record['prompt_id']}::{record['candidate_id']}"
            if key in seen_ids:
                dropped["duplicate_id"] += 1
                continue
            seen_ids.add(key)

            if record.get("candidate_source") == "sft_reference":
                kept.append(record)
                continue

            answer_key = normalize(record.get("answer", ""))
            reason = invalid_sample(answer_key, args.min_chars, args.max_chars)
            if reason:
                dropped[reason] += 1
                continue
            if answer_key in seen_answers:
                dropped["duplicate_answer"] += 1
                continue
            seen_answers.add(answer_key)
            kept.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for record in kept:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"input": sum(1 for _ in args.input.open(encoding='utf-8')), "kept": len(kept), "dropped": dict(dropped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
