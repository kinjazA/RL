"""Build auditable synthetic preference pairs for reward-model training.

The script deliberately separates policy generation, judge scoring, and pair
selection.  Each stage is resumable and writes its intermediate data to disk:

    selected_prompts.jsonl -> candidates.jsonl -> scored_candidates.jsonl
                            -> preference_pairs.jsonl

Start with a stratified pilot.  For example:

    python rm/build_preference_data.py --output_dir rm/artifacts/pilot_v1 --limit 200

The frozen ``eval/sft_test_v1.json`` is never read by this script and must not
be added to the source input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "rlhf_answers_filled.csv"
DEFAULT_SFT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_SFT_ADAPTER = "Shawnno/qwen2.5-3b-interview-sft-lora"
DEFAULT_JUDGE = "Skywork/Skywork-Reward-V2-Qwen3-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "score", "build"),
        default="all",
        help="Run one resumable stage, or all stages in sequence.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Stratified prompt count. 0 uses the full source dataset. Default is a pilot.",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--sft_model", default=DEFAULT_SFT_MODEL)
    parser.add_argument("--sft_adapter", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--temperatures",
        default="0.3,0.5,0.7,0.9,1.1",
        help="Comma-separated temperatures; one policy sample is generated per value.",
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--judge_max_length", type=int, default=2048)
    parser.add_argument("--judge_batch_size", type=int, default=8)
    parser.add_argument("--min_answer_chars", type=int, default=80)
    parser.add_argument("--max_answer_chars", type=int, default=700)
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.5,
        help="Minimum chosen-minus-rejected teacher reward within a prompt.",
    )
    parser.add_argument(
        "--max_margin",
        type=float,
        default=8.0,
        help="Maximum teacher reward gap accepted for a hard negative.",
    )
    parser.add_argument(
        "--no_sample_chosen_fallback",
        action="store_true",
        help="Only retain pairs whose original SFT response can be the chosen answer.",
    )
    parser.add_argument(
        "--overwrite_selection",
        action="store_true",
        help="Replace an existing selected_prompts.jsonl. It does not delete other artifacts.",
    )
    return parser.parse_args()


def parse_temperatures(value: str) -> list[float]:
    try:
        temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--temperatures must be comma-separated numbers.") from error
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("--temperatures must contain at least one value greater than zero.")
    return temperatures


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Source data not found: {args.input}")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1].")
    if args.min_answer_chars < 1 or args.max_answer_chars <= args.min_answer_chars:
        raise ValueError("Answer-length bounds are invalid.")
    if args.min_margin < 0 or args.max_margin <= args.min_margin:
        raise ValueError("Reward-margin bounds are invalid.")
    parse_temperatures(args.temperatures)
    if args.stage in {"all", "generate", "score"} and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. In Colab select Runtime > Change runtime type > T4 GPU.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    materialized = list(records)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("--input must be CSV or JSON.")

    if not rows:
        raise ValueError("Source dataset is empty.")
    columns = set(rows[0])
    if not ({"question", "instruction"} & columns) or not ({"answer", "output"} & columns):
        raise ValueError("Source must contain question/answer or instruction/output columns.")

    normalized = []
    seen_ids = set()
    for index, row in enumerate(rows):
        question = str(row.get("question") or row.get("instruction") or "").strip()
        answer = str(row.get("answer") or row.get("output") or "").strip()
        if not question or not answer:
            continue
        prompt_id = str(row.get("id") or stable_id("prompt", question))
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id in source: {prompt_id}")
        seen_ids.add(prompt_id)
        normalized.append(
            {
                "prompt_id": prompt_id,
                "question": question,
                "reference_answer": answer,
                "role": str(row.get("role") or "unknown"),
                "category": str(row.get("category") or ""),
                "skill": str(row.get("skill") or ""),
                "question_type": str(row.get("question_type") or ""),
                "difficulty": str(row.get("difficulty") or ""),
                "seniority": str(row.get("seniority") or ""),
                "source_type": str(row.get("source_type") or "unknown"),
                "source_row": index,
            }
        )
    if not normalized:
        raise ValueError("No complete question/answer records found in source.")
    return normalized


def stratified_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit == 0 or limit >= len(rows):
        return sorted(rows, key=lambda row: row["prompt_id"])

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["role"]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    role_names = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for role in role_names:
            if groups[role] and len(selected) < limit:
                selected.append(groups[role].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda row: row["prompt_id"])


def selected_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.output_dir / "selected_prompts.jsonl"
    if path.exists() and not args.overwrite_selection:
        records = load_jsonl(path)
        if not records:
            raise ValueError(f"Existing selection is empty: {path}")
        return records
    if path.exists() and args.overwrite_selection:
        path.unlink()
    rows = stratified_sample(load_source_rows(args.input), args.limit, args.seed)
    append_jsonl(path, rows)
    return rows


def quantization_config() -> BitsAndBytesConfig:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_policy(model_name: str, adapter_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_name)
    model.eval()
    return model, tokenizer


def policy_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def sample_answer(
    model,
    tokenizer,
    question: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, str]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prompt = policy_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip(), prompt


def candidate_key(record: dict[str, Any]) -> str:
    return f"{record['prompt_id']}::{record['candidate_id']}"


def run_generate(args: argparse.Namespace, prompts: list[dict[str, Any]], temperatures: list[float]) -> None:
    candidates_path = args.output_dir / "candidates.jsonl"
    existing = {candidate_key(row) for row in load_jsonl(candidates_path)}
    references = []
    for row in prompts:
        record = {
            **row,
            "candidate_id": "reference",
            "candidate_source": "sft_reference",
            "answer": row["reference_answer"],
            "temperature": None,
            "generation_seed": None,
            "generation_prompt": None,
            "created_at_utc": utc_now(),
        }
        if candidate_key(record) not in existing:
            references.append(record)
    append_jsonl(candidates_path, references)

    pending = []
    for row in prompts:
        for candidate_index, temperature in enumerate(temperatures):
            candidate_id = f"sample_t{temperature:g}_{candidate_index}"
            record = {"prompt_id": row["prompt_id"], "candidate_id": candidate_id}
            if candidate_key(record) not in existing:
                pending.append((row, candidate_id, candidate_index, temperature))
    if not pending:
        print(f"Generation already complete: {candidates_path}")
        return

    print(f"Loading SFT policy: {args.sft_model} + {args.sft_adapter}")
    model, tokenizer = load_policy(args.sft_model, args.sft_adapter)
    try:
        for index, (row, candidate_id, candidate_index, temperature) in enumerate(pending, start=1):
            generation_seed = args.seed + row["source_row"] * 100 + candidate_index
            answer, prompt = sample_answer(
                model=model,
                tokenizer=tokenizer,
                question=row["question"],
                temperature=temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                seed=generation_seed,
            )
            append_jsonl(
                candidates_path,
                [
                    {
                        **row,
                        "candidate_id": candidate_id,
                        "candidate_source": "sft_sample",
                        "answer": answer,
                        "temperature": temperature,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "generation_seed": generation_seed,
                        "generation_prompt": prompt,
                        "created_at_utc": utc_now(),
                    }
                ],
            )
            print(f"Generated {index}/{len(pending)}: {row['prompt_id']} {candidate_id}")
    finally:
        del model
        torch.cuda.empty_cache()


def load_judge(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map="auto",
        num_labels=1,
    )
    model.eval()
    return model, tokenizer


def format_judge_conversation(tokenizer, question: str, answer: str) -> str:
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Skywork's official example removes the chat template's duplicate BOS token.
    if tokenizer.bos_token is not None and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token) :]
    return formatted


@torch.inference_mode()
def score_batch(model, tokenizer, records: list[dict[str, Any]], max_length: int) -> list[float]:
    texts = [format_judge_conversation(tokenizer, row["question"], row["answer"]) for row in records]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model_device(model))
    logits = model(**inputs).logits.squeeze(-1)
    return [float(value) for value in logits.detach().float().cpu().tolist()]


def run_score(args: argparse.Namespace) -> None:
    candidates_path = args.output_dir / "candidates.jsonl"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Run --stage generate first: {candidates_path}")
    candidates = load_jsonl(candidates_path)
    if not candidates:
        raise ValueError("Candidate file is empty.")
    scored_path = args.output_dir / "scored_candidates.jsonl"
    existing = {candidate_key(row) for row in load_jsonl(scored_path)}
    pending = [row for row in candidates if candidate_key(row) not in existing]
    if not pending:
        print(f"Scoring already complete: {scored_path}")
        return

    print(f"Loading teacher judge: {args.judge_model}")
    model, tokenizer = load_judge(args.judge_model)
    try:
        for start in range(0, len(pending), args.judge_batch_size):
            batch = pending[start : start + args.judge_batch_size]
            scores = score_batch(model, tokenizer, batch, args.judge_max_length)
            append_jsonl(
                scored_path,
                [
                    {
                        **record,
                        "judge_model": args.judge_model,
                        "judge_reward": score,
                        "scored_at_utc": utc_now(),
                    }
                    for record, score in zip(batch, scores, strict=True)
                ],
            )
            print(f"Scored {min(start + len(batch), len(pending))}/{len(pending)}")
    finally:
        del model
        torch.cuda.empty_cache()


def quality_flags(answer: str, min_chars: int, max_chars: int) -> list[str]:
    compact = normalize_text(answer)
    flags = []
    if not compact:
        return ["empty"]
    if len(compact) < min_chars:
        flags.append("too_short")
    if len(compact) > max_chars:
        flags.append("too_long")
    ngrams = Counter(compact[index : index + 5] for index in range(max(0, len(compact) - 4)))
    if ngrams and max(ngrams.values()) >= 4:
        flags.append("repeated_5gram")
    if re.search(r"(.{3,20})\1{2,}", compact):
        flags.append("repeated_span")
    return flags


def deduplicate_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one exact-text candidate, preferring the reference then higher reward."""
    by_answer: dict[str, dict[str, Any]] = {}
    for record in records:
        key = normalize_text(record["answer"])
        current = by_answer.get(key)
        if current is None:
            by_answer[key] = record
            continue
        current_is_reference = current["candidate_source"] == "sft_reference"
        record_is_reference = record["candidate_source"] == "sft_reference"
        if record_is_reference and not current_is_reference:
            by_answer[key] = record
        elif current_is_reference == record_is_reference and record["judge_reward"] > current["judge_reward"]:
            by_answer[key] = record
    return list(by_answer.values())


def select_pair(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, Any] | None, str | None]:
    viable = [
        row
        for row in deduplicate_candidates(records)
        if not quality_flags(row["answer"], args.min_answer_chars, args.max_answer_chars)
    ]
    if len(viable) < 2:
        return None, "fewer_than_two_viable_answers"

    reference = next((row for row in viable if row["candidate_source"] == "sft_reference"), None)
    possible_chosen = [reference] if reference else []
    if not args.no_sample_chosen_fallback:
        possible_chosen.extend(
            sorted(viable, key=lambda row: row["judge_reward"], reverse=True)
        )

    visited = set()
    for chosen in possible_chosen:
        if chosen is None or candidate_key(chosen) in visited:
            continue
        visited.add(candidate_key(chosen))
        negatives = []
        for rejected in viable:
            if candidate_key(rejected) == candidate_key(chosen):
                continue
            margin = chosen["judge_reward"] - rejected["judge_reward"]
            if args.min_margin <= margin <= args.max_margin:
                negatives.append((margin, rejected))
        if negatives:
            # The smallest valid score gap is a useful hard negative. Quality filters
            # above have already removed empty, looping, and wildly off-length text.
            margin, rejected = min(negatives, key=lambda item: item[0])
            return {
                "chosen": chosen,
                "rejected": rejected,
                "margin": margin,
                "viable_count": len(viable),
                "chosen_source": chosen["candidate_source"],
            }, None
    return None, "no_hard_negative_in_margin_band"


def write_pair_outputs(args: argparse.Namespace, pairs: list[dict[str, Any]]) -> None:
    canonical_path = args.output_dir / "preference_pairs.jsonl"
    if canonical_path.exists():
        canonical_path.unlink()
    append_jsonl(canonical_path, pairs)

    llama_factory_rows = [
        {
            "instruction": row["question"],
            "input": "",
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in pairs
    ]
    write_json(args.output_dir / "preference_pairs_llamafactory.json", llama_factory_rows)
    write_json(
        args.output_dir / "dataset_info.json",
        {
            "interview_preference_pairs": {
                "file_name": "preference_pairs_llamafactory.json",
                "ranking": True,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            }
        },
    )


def write_audit_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = [
        "prompt_id", "role", "difficulty", "question_type", "source_type", "question",
        "chosen_source", "chosen_temperature", "chosen_reward", "chosen",
        "rejected_temperature", "rejected_reward", "rejected", "judge_margin",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)


def summary_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
    }


def run_build(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    scored_path = args.output_dir / "scored_candidates.jsonl"
    records = load_jsonl(scored_path)
    if not records:
        raise FileNotFoundError(f"Run --stage score first: {scored_path}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["prompt_id"]].append(record)

    pairs = []
    skipped = Counter()
    for prompt in prompts:
        result, reason = select_pair(grouped.get(prompt["prompt_id"], []), args)
        if result is None:
            skipped[reason or "unknown"] += 1
            continue
        chosen = result["chosen"]
        rejected = result["rejected"]
        pairs.append(
            {
                "prompt_id": prompt["prompt_id"],
                "role": prompt["role"],
                "category": prompt["category"],
                "skill": prompt["skill"],
                "difficulty": prompt["difficulty"],
                "question_type": prompt["question_type"],
                "source_type": prompt["source_type"],
                "question": prompt["question"],
                "prompt": [{"role": "user", "content": prompt["question"]}],
                "chosen": chosen["answer"],
                "rejected": rejected["answer"],
                "chosen_source": result["chosen_source"],
                "chosen_temperature": chosen["temperature"],
                "chosen_reward": chosen["judge_reward"],
                "rejected_temperature": rejected["temperature"],
                "rejected_reward": rejected["judge_reward"],
                "judge_margin": result["margin"],
                "viable_candidate_count": result["viable_count"],
                "judge_model": args.judge_model,
                "selection_policy": "reference_first_then_top_sample_hard_negative",
            }
        )

    write_pair_outputs(args, pairs)
    write_audit_csv(args.output_dir / "manual_audit.csv", pairs)
    role_counts = Counter(pair["role"] for pair in pairs)
    chosen_sources = Counter(pair["chosen_source"] for pair in pairs)
    summary = {
        "created_at_utc": utc_now(),
        "source_input": str(args.input),
        "selected_prompts": len(prompts),
        "scored_candidates": len(records),
        "preference_pairs": len(pairs),
        "retention_rate": round(len(pairs) / len(prompts), 4) if prompts else 0.0,
        "judge_model": args.judge_model,
        "selection": {
            "min_answer_chars": args.min_answer_chars,
            "max_answer_chars": args.max_answer_chars,
            "min_margin": args.min_margin,
            "max_margin": args.max_margin,
            "sample_chosen_fallback": not args.no_sample_chosen_fallback,
        },
        "role_counts": dict(sorted(role_counts.items())),
        "chosen_source_counts": dict(sorted(chosen_sources.items())),
        "skipped": dict(sorted(skipped.items())),
        "judge_margin": summary_statistics([pair["judge_margin"] for pair in pairs]),
    }
    write_json(args.output_dir / "data_quality_report.json", summary)
    markdown = [
        "# Preference Data Quality Report",
        "",
        f"- Selected prompts: {summary['selected_prompts']}",
        f"- Scored candidates: {summary['scored_candidates']}",
        f"- Retained preference pairs: {summary['preference_pairs']} ({summary['retention_rate']:.1%})",
        f"- Teacher judge: `{args.judge_model}`",
        f"- Margin band: [{args.min_margin}, {args.max_margin}]",
        f"- Chosen sources: {summary['chosen_source_counts']}",
        f"- Skipped: {summary['skipped']}",
        f"- Reward-margin distribution: {summary['judge_margin']}",
        "",
        "`manual_audit.csv` contains every selected pair for review. Do not train a reward model before auditing a stratified sample from that file.",
    ]
    (args.output_dir / "data_quality_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Built {len(pairs)}/{len(prompts)} preference pairs: {args.output_dir}")


def write_manifest(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    manifest = {
        "created_at_utc": utc_now(),
        "input": str(args.input),
        "selected_prompt_count": len(prompts),
        "seed": args.seed,
        "sft_model": args.sft_model,
        "sft_adapter": args.sft_adapter,
        "judge_model": args.judge_model,
        "temperatures": parse_temperatures(args.temperatures),
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = selected_prompts(args)
    write_manifest(args, prompts)
    temperatures = parse_temperatures(args.temperatures)

    if args.stage in {"all", "generate"}:
        run_generate(args, prompts, temperatures)
    if args.stage in {"all", "score"}:
        run_score(args)
    if args.stage in {"all", "build"}:
        run_build(args, prompts)


if __name__ == "__main__":
    main()
