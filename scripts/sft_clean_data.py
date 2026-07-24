"""
Content cleaning + deduplication for interview SFT data.

One-step data preparation from raw train.csv → clean, deduplicated output.
Handles both problems found in the raw data audit:

  1. Format pollution (ds_qa_treasury): strip HTML tags, markdown images,
     code fences, HTML entities; normalize whitespace.
  2. Question repetition (local_interview_qa): cap identical questions
     at N rows, preferring diverse/longer answers within each group.

Also flags AI refusal rows for removal and adds metadata columns
(domain, answer_type, normalized_question).

Input:  data/train.csv       (question, answer, source)
Output: data/sft_clean.csv   (cleaned, deduplicated)
        data/sft_removed.csv (rows removed + removal reasons)

Usage from RL/ directory:
    python scripts/sft_clean_data.py [--input data/train.csv] [--output data/sft_clean.csv]
"""

import argparse
import os
import re
from collections import Counter

import pandas as pd


# ===========================================================================
# Content cleaners
# ===========================================================================

BLOCK_HTML = re.compile(
    r"</?(?:p|div|br|hr|li|ul|ol|h[1-6]|table|tr|td|th|thead|tbody|section|article|header|footer|nav|main|aside|blockquote|pre|figure|figcaption|details|summary|fieldset|form|dl|dt|dd)[^>]*/?>",
    re.IGNORECASE,
)
INLINE_HTML = re.compile(
    r"</?(?:strong|b|em|i|u|s|span|code|a|sub|sup|small|mark|ins|del|abbr|cite|q|font|tt|kbd|var|samp|wbr|nobr|ruby|rt|rp|bdi|bdo|dfn|time|data|output|label|button)[^>]*/?>",
    re.IGNORECASE,
)
SELF_CLOSING_HTML = re.compile(
    r"<(?:img|input|meta|link|area|base|col|embed|source|track|param|hr|br|wbr)[^>]*/?>",
    re.IGNORECASE,
)
ANY_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_-]*(?:\s[^>]*)?/?>")

ENTITY_MAP = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'",
    "&#8203;": "", "&#x200B;": "", "&ZeroWidthSpace;": "",
    "&nbsp;": " ", "&#160;": " ", "&ensp;": " ", "&emsp;": " ", "&thinsp;": " ",
    "&mdash;": "—", "&ndash;": "–",
    "&lsquo;": "‘", "&rsquo;": "’",
    "&ldquo;": "“", "&rdquo;": "”",
    "&hellip;": "…", "&laquo;": "«", "&raquo;": "»",
}
_ENTITY_PAT = re.compile("|".join(re.escape(k) for k in ENTITY_MAP))
_NUMERIC_ENTITY = re.compile(r"&#(\d+);")
_HEX_ENTITY = re.compile(r"&#x([0-9A-Fa-f]+);")

MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
CODE_FENCE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")

MULTI_NEWLINE = re.compile(r"\n{3,}")
TRAILING_SPACES = re.compile(r"[ \t]+\n")
MULTI_SPACE = re.compile(r"[ \t]{2,}")
LEADING_WHITESPACE = re.compile(r"^[ \t]+", re.MULTILINE)

AI_REFUSAL_PATTERNS = [
    re.compile(r"^as an ai\b", re.IGNORECASE),
    re.compile(r"^i am an ai\b", re.IGNORECASE),
    re.compile(r"^as a language model\b", re.IGNORECASE),
    re.compile(r"^i am a language model\b", re.IGNORECASE),
    re.compile(r"^as an artificial intelligence\b", re.IGNORECASE),
    re.compile(r"^i cannot\b", re.IGNORECASE),
    re.compile(r"^i can'?t\b", re.IGNORECASE),
    re.compile(r"^i'm not able to\b", re.IGNORECASE),
    re.compile(r"^sorry,?\s*(?:but\s+)?i\s+(?:can'?t|cannot|am not)\b", re.IGNORECASE),
]

# ===========================================================================
# Source → metadata mapping
# ===========================================================================

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

# ===========================================================================
# Normalization (for dedup grouping)
# ===========================================================================

def normalize_question(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ===========================================================================
# Content cleaning functions
# ===========================================================================

def clean_html(text: str) -> tuple[str, list[str]]:
    """Strip HTML tags and convert entities."""
    actions = []
    original = text
    if BLOCK_HTML.search(text):
        text = BLOCK_HTML.sub("\n", text)
        actions.append("strip_block_html")
    if INLINE_HTML.search(text):
        text = INLINE_HTML.sub("", text)
        actions.append("strip_inline_html")
    if SELF_CLOSING_HTML.search(text):
        text = SELF_CLOSING_HTML.sub("", text)
        actions.append("strip_self_closing_html")
    if ANY_HTML_TAG.search(text):
        text = ANY_HTML_TAG.sub("", text)
    if _ENTITY_PAT.search(text):
        text = _ENTITY_PAT.sub(lambda m: ENTITY_MAP.get(m.group(0), m.group(0)), text)
        actions.append("convert_entities")
    if _NUMERIC_ENTITY.search(text):
        text = _NUMERIC_ENTITY.sub(lambda m: chr(int(m.group(1))), text)
        actions.append("convert_numeric_entities")
    if _HEX_ENTITY.search(text):
        text = _HEX_ENTITY.sub(lambda m: chr(int(m.group(1), 16)), text)
        actions.append("convert_hex_entities")
    if text != original:
        actions.append("html_cleaned")
    return text, actions


def clean_markdown(text: str) -> tuple[str, list[str]]:
    actions = []
    if MD_IMAGE.search(text):
        text = MD_IMAGE.sub(lambda m: m.group(1).strip() or "(formula)", text)
        actions.append("strip_md_images")
    if MD_LINK.search(text):
        text = MD_LINK.sub(r"\1", text)
        actions.append("strip_md_links")
    if CODE_FENCE.search(text):
        text = CODE_FENCE.sub(lambda m: m.group(0).strip("`~").strip(), text)
        actions.append("strip_code_fences")
    return text, actions


def normalize_whitespace(text: str) -> tuple[str, list[str]]:
    actions = []
    original = text
    text = TRAILING_SPACES.sub("\n", text)
    text = MULTI_SPACE.sub(" ", text)
    text = LEADING_WHITESPACE.sub("", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()
    if text != original:
        actions.append("whitespace_normalized")
    return text, actions


def detect_ai_refusal(text: str) -> bool:
    stripped = text.strip()
    for pat in AI_REFUSAL_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def clean_text(text: str) -> tuple[str, list[str]]:
    all_actions: list[str] = []
    text, actions = clean_html(text)
    all_actions.extend(actions)
    text, actions = clean_markdown(text)
    all_actions.extend(actions)
    text, actions = normalize_whitespace(text)
    all_actions.extend(actions)
    return text, all_actions


# ===========================================================================
# Deduplication
# ===========================================================================

def _answer_diversity_score(answer: str) -> float:
    """Prefer answers with more structural variety (paragraphs, sentences)."""
    paragraphs = max(1, len([p for p in answer.split("\n\n") if p.strip()]))
    sentences = max(1, len(re.split(r"[.!?]+", answer)))
    return paragraphs + sentences * 0.5


def deduplicate(df: pd.DataFrame, max_per_question: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicates and cap repeated normalized questions.

    Returns (kept, removed).
    """
    removal_reasons: dict[int, list[str]] = {}

    # --- Exact duplicate (same question + same answer) ---
    dup_mask = df.duplicated(["question", "answer"], keep="first")
    for idx in df.index[dup_mask]:
        removal_reasons.setdefault(idx, []).append("exact_duplicate")

    # --- Cap per normalized question ---
    # Within the same norm_q, keep rows with the most diverse answers
    df_temp = df.copy()
    df_temp["_score"] = df_temp["answer"].map(_answer_diversity_score)
    # Add small random jitter to break ties deterministically
    df_temp["_score"] += df_temp.index.map(lambda i: (hash(str(i)) % 100) * 0.001)

    kept_mask = pd.Series(True, index=df_temp.index)
    for nq, grp in df_temp.groupby("normalized_question", sort=False):
        if len(grp) <= max_per_question:
            continue
        # Sort by diversity score descending, keep top N
        sorted_idx = grp.sort_values("_score", ascending=False).index
        drop_idx = sorted_idx[max_per_question:]
        for idx in drop_idx:
            removal_reasons.setdefault(idx, []).append(
                f"duplicate_question_over_cap_{max_per_question}"
            )

    # --- AI refusal ---
    refusal_mask = df["answer"].map(detect_ai_refusal)
    for idx in df.index[refusal_mask]:
        removal_reasons.setdefault(idx, []).append("ai_refusal")

    # Split
    removed_indices = set(removal_reasons.keys())
    kept_rows = []
    removed_rows = []
    for idx in df.index:
        row = df.loc[idx].to_dict()
        if idx in removed_indices:
            row["removal_reason"] = ";".join(removal_reasons[idx])
            removed_rows.append(row)
        else:
            row["removal_reason"] = ""
            kept_rows.append(row)

    return pd.DataFrame(kept_rows), pd.DataFrame(removed_rows)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=os.path.join("data", "train.csv"))
    parser.add_argument("--output", default=os.path.join("data", "sft_clean.csv"))
    parser.add_argument("--removed_output", default=os.path.join("data", "sft_removed.csv"))
    parser.add_argument("--max_per_question", type=int, default=3)
    args = parser.parse_args()

    # ---- Load ----
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"question", "answer", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    n_total = len(df)
    print(f"Loaded {n_total:,} rows from {args.input}")

    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    # ---- Pre-cleaning stats ----
    pre_html = int(df["answer"].map(lambda t: bool(ANY_HTML_TAG.search(t))).sum())
    pre_md_img = int(df["answer"].map(lambda t: bool(MD_IMAGE.search(t))).sum())
    pre_fences = int(df["answer"].map(lambda t: bool(CODE_FENCE.search(t))).sum())
    pre_ws = int(df["answer"].map(
        lambda t: bool(TRAILING_SPACES.search(t) or MULTI_SPACE.search(t) or MULTI_NEWLINE.search(t))
    ).sum())
    pre_refusal = int(df["answer"].map(detect_ai_refusal).sum())

    # ---- Content cleaning ----
    print("Cleaning content ...")
    for idx, row in df.iterrows():
        q, a = row["question"], row["answer"]
        clean_a, _ = clean_text(a)
        clean_q, _ = clean_text(q)
        df.at[idx, "question"] = clean_q
        df.at[idx, "answer"] = clean_a

    # ---- Add metadata ----
    df["domain"] = df["source"].map(SOURCE_DOMAIN).fillna("unknown")
    df["answer_type"] = df["source"].map(SOURCE_ANSWER_TYPE).fillna("unknown")
    df["normalized_question"] = df["question"].map(normalize_question)

    # ---- Dedup stats (before) ----
    n_exact_dup = int(df.duplicated(["question", "answer"], keep="first").sum())
    nq_counts = df["normalized_question"].value_counts()
    over_cap_groups = int((nq_counts > args.max_per_question).sum())
    over_cap_rows = int(nq_counts[nq_counts > args.max_per_question].sum())

    # ---- Deduplicate ----
    print("Deduplicating ...")
    kept, removed = deduplicate(df, args.max_per_question)
    n_removed = len(removed)

    # ---- Save ----
    out_cols = ["question", "answer", "source", "domain", "answer_type", "normalized_question"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept[out_cols].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  Kept    -> {args.output}  ({len(kept):,} rows)")

    if n_removed > 0:
        removed_cols = out_cols + ["removal_reason"]
        removed[removed_cols].to_csv(args.removed_output, index=False, encoding="utf-8-sig")
        print(f"  Removed -> {args.removed_output}  ({n_removed:,} rows)")
    else:
        print("  No rows removed.")

    # ---- Post-cleaning stats ----
    post_html = int(kept["answer"].map(lambda t: bool(ANY_HTML_TAG.search(t))).sum())
    post_md_img = int(kept["answer"].map(lambda t: bool(MD_IMAGE.search(t))).sum())
    post_fences = int(kept["answer"].map(lambda t: bool(CODE_FENCE.search(t))).sum())
    post_ws = int(kept["answer"].map(
        lambda t: bool(TRAILING_SPACES.search(t) or MULTI_SPACE.search(t) or MULTI_NEWLINE.search(t))
    ).sum())

    # ---- Per-source summary ----
    print("\nPer-source changes:")
    print(f"  {'Source':<22} {'Before':>7} {'After':>7} {'Removed':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}")
    for src in sorted(df["source"].unique()):
        pre_n = len(df[df["source"] == src])
        post_n = len(kept[kept["source"] == src])
        rm_n = pre_n - post_n
        print(f"  {src:<22} {pre_n:>7,} {post_n:>7,} {rm_n:>7,}")

    # ---- Removal reason breakdown ----
    if n_removed > 0:
        print("\nRemoval reasons:")
        exploded = removed["removal_reason"].str.split(";").explode().dropna()
        for reason, count in exploded.value_counts().items():
            print(f"  {reason}: {count:,}")

    # ---- Terminal summary ----
    print(f"""
Done.
  Content:  HTML {pre_html}->{post_html}  MD img {pre_md_img}->{post_md_img}  fences {pre_fences}->{post_fences}  WS {pre_ws}->{post_ws}
  Dedup:    exact_dup={n_exact_dup}  over_cap={over_cap_rows} rows in {over_cap_groups} groups  ai_refusal={pre_refusal}
  Result:   {n_total:,} -> {len(kept):,} rows ({n_removed} removed)
""")


if __name__ == "__main__":
    main()
