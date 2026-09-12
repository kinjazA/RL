"""Generate deterministic Base-vs-SFT-vs-DPO comparisons on the frozen eval set.

The evaluation questions and their scoring points are never passed to the model.
Reward-model scoring is optional and kept separate from generation; the RM prompt
must match the format used during RM training.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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
DEFAULT_EVAL_FILE = ROOT / "eval" / "sft_test_v1.json"

SENTENCE_END = set("。！？….!?")


def ensure_natural_ending(text: str) -> str:
    """Cut a truncated answer back to its last sentence-ending punctuation."""
    text = text.rstrip()
    for index in range(len(text) - 1, -1, -1):
        if text[index] in SENTENCE_END:
            return text[: index + 1]
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--sft_adapter", default="Shawnno/qwen2.5-3b-interview-sft-lora"
    )
    parser.add_argument(
        "--dpo_adapter",
        default=None,
        help="Optional SFT+DPO LoRA adapter. Omit to run Base-vs-SFT only.",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "eval" / "results" / "sft_acceptance_v1")
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only. 0 evaluates all questions.")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rm_model", help="Reward-model base or full model path. Omit to skip RM scoring.")
    parser.add_argument("--rm_adapter", help="Optional LoRA adapter trained on --rm_model.")
    parser.add_argument(
        "--rm_input_format",
        choices=("question_answer", "qwen_chat"),
        help="Required with --rm_model. It must match the RM training format exactly.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab select Runtime > Change runtime type > T4 GPU.")
    if bool(args.rm_model) != bool(args.rm_input_format):
        raise ValueError("--rm_model and --rm_input_format must be supplied together.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite or a new path.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_policy(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_policy_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> tuple[str, str]:
    prompt = format_policy_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if generated_ids.shape[0] >= max_new_tokens:
        answer = ensure_natural_ending(answer)
    return answer, prompt


def load_eval_rows(path: Path, limit: int) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "role", "difficulty", "question_type", "question", "expected_points"}
    if not all(required.issubset(row) for row in rows):
        raise ValueError(f"Evaluation file is missing required fields: {path}")
    return rows[:limit] if limit else rows


def format_rm_input(tokenizer, question: str, answer: str, input_format: str) -> str:
    if input_format == "question_answer":
        return f"Question: {question}\nAnswer: {answer}"
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def load_reward_model(model_name: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map="auto",
        num_labels=1,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def reward_score(model, tokenizer, question: str, answer: str, input_format: str) -> float:
    text = format_rm_input(tokenizer, question, answer, input_format)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model_device(model))
    logits = model(**inputs).logits.squeeze()
    if logits.numel() != 1:
        raise ValueError(f"Reward model must output one scalar, got logits shape {tuple(model(**inputs).logits.shape)}")
    return float(logits.item())


def safe_median(values: list[int]) -> float:
    return float(np.median(values)) if values else 0.0


def length_summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean_chars": 0.0, "median_chars": 0.0, "p90_chars": 0.0, "max_chars": 0}
    return {
        "mean_chars": round(float(np.mean(values)), 2),
        "median_chars": round(safe_median(values), 2),
        "p90_chars": round(float(np.percentile(values, 90)), 2),
        "max_chars": int(max(values)),
    }


def bucket_rates(values: list[int]) -> dict[str, float]:
    n = len(values)
    if not n:
        return {"pct_150_300": 0.0, "pct_150_450": 0.0, "pct_le_550": 0.0}
    return {
        "pct_150_300": round(sum(150 <= v <= 300 for v in values) / n * 100, 1),
        "pct_150_450": round(sum(150 <= v <= 450 for v in values) / n * 100, 1),
        "pct_le_550": round(sum(v <= 550 for v in values) / n * 100, 1),
    }


def natural_ending(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] in SENTENCE_END


def natural_ending_rate(values: list[str]) -> float:
    n = len(values)
    if not n:
        return 0.0
    return round(sum(natural_ending(v) for v in values) / n * 100, 1)


def repetition_rates(values: list[str]) -> dict[str, float]:
    n = len(values)
    if not n:
        return {"strong_pct": 0.0, "extreme_pct": 0.0}
    strong = extreme = 0
    for text in values:
        if len(text) < 5:
            continue
        counts = Counter(text[i : i + 5] for i in range(len(text) - 4))
        max_repeat = max(counts.values())
        if max_repeat >= 4:
            strong += 1
        if max_repeat >= 8:
            extreme += 1
    return {
        "strong_pct": round(strong / n * 100, 1),
        "extreme_pct": round(extreme / n * 100, 1),
    }


def model_stats(values: list[str]) -> dict:
    lengths = [len(v) for v in values]
    return {
        "length": length_summary(lengths),
        "buckets": bucket_rates(lengths),
        "natural_ending_pct": natural_ending_rate(values),
        "repetition": repetition_rates(values),
    }


def reward_summary(rows: list[dict], seed: int, models: list[str]) -> dict[str, float]:
    arrays = {m: np.array([row[f"rm_{m}"] for row in rows], dtype=float) for m in models}
    out: dict[str, float] = {f"{m}_mean": round(float(v.mean()), 4) for m, v in arrays.items()}

    pairs = [(a, b) for a in models for b in models if a != b]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(5000, len(rows)))
    for a, b in pairs:
        delta = arrays[a] - arrays[b]
        boot_delta = delta[indices].mean(axis=1)
        boot_win = (delta[indices] > 0).mean(axis=1)
        out[f"{a}_vs_{b}_mean_delta"] = round(float(delta.mean()), 4)
        out[f"{a}_vs_{b}_delta_ci95_low"] = round(float(np.percentile(boot_delta, 2.5)), 4)
        out[f"{a}_vs_{b}_delta_ci95_high"] = round(float(np.percentile(boot_delta, 97.5)), 4)
        out[f"{a}_vs_{b}_win_rate"] = round(float((delta > 0).mean()), 4)
        out[f"{a}_vs_{b}_win_ci95_low"] = round(float(np.percentile(boot_win, 2.5)), 4)
        out[f"{a}_vs_{b}_win_ci95_high"] = round(float(np.percentile(boot_win, 97.5)), 4)
    return out


def _format_num(value: float) -> str:
    return f"{value:.1f}"


def render_markdown(summary: dict, models: list[str]) -> str:
    lines = [
        "# Acceptance Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Base: `{summary['base_model']}`",
        f"- SFT adapter: `{summary['sft_adapter']}`",
    ]
    if summary.get("dpo_adapter"):
        lines.append(f"- DPO adapter: `{summary['dpo_adapter']}`")
    lines += [
        f"- Decoding: greedy (`do_sample=false`), max_new_tokens={summary['max_new_tokens']}",
        "",
        "## Length",
        "",
        "| Metric | " + " | ".join(m.upper() for m in models) + " |",
        "|---|---|" + "---|" * (len(models) - 1),
    ]
    for metric in ("mean_chars", "median_chars", "p90_chars", "max_chars"):
        row = [summary[m]["length"][metric] for m in models]
        lines.append(f"| {metric} | " + " | ".join(_format_num(v) if isinstance(v, float) else str(v) for v in row) + " |")

    lines += [
        "",
        "## Length buckets & style diagnostics",
        "",
        "| Metric | " + " | ".join(m.upper() for m in models) + " |",
        "|---|---|" + "---|" * (len(models) - 1),
    ]
    for metric in ("pct_150_300", "pct_150_450", "pct_le_550"):
        row = [summary[m]["buckets"][metric] for m in models]
        lines.append(f"| {metric} (%) | " + " | ".join(_format_num(v) for v in row) + " |")
    row = [summary[m]["natural_ending_pct"] for m in models]
    lines.append("| natural_ending (%) | " + " | ".join(_format_num(v) for v in row) + " |")
    for metric, label in (("strong_pct", "strong 5-gram repeat (%)"), ("extreme_pct", "extreme 5-gram repeat (%)")):
        row = [summary[m]["repetition"][metric] for m in models]
        lines.append(f"| {label} | " + " | ".join(_format_num(v) for v in row) + " |")

    if "reward" in summary:
        r = summary["reward"]
        lines += [
            "",
            "## Reward Model (paired, not an absolute quality score)",
            "",
            f"- RM: `{summary['rm_model']}`",
            f"- Input format: `{summary['rm_input_format']}`",
            "",
            "| Pair | Mean delta | Win rate |",
            "|---|---|---|",
        ]
        for a in models:
            for b in models:
                if a == b:
                    continue
                lines.append(
                    f"| {a.upper()} vs {b.upper()} | {r[f'{a}_vs_{b}_mean_delta']} "
                    f"({r[f'{a}_vs_{b}_delta_ci95_low']}, {r[f'{a}_vs_{b}_delta_ci95_high']}) | "
                    f"{r[f'{a}_vs_{b}_win_rate']} "
                    f"({r[f'{a}_vs_{b}_win_ci95_low']}, {r[f'{a}_vs_{b}_win_ci95_high']}) |"
                )
    return "\n".join(lines) + "\n"


def write_outputs(args: argparse.Namespace, rows: list[dict], summary: dict, models: list[str]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparisons.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(render_markdown(summary, models), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    eval_rows = load_eval_rows(args.eval_file, args.limit)

    print(f"Loading base model: {args.base_model}")
    base_model, tokenizer = load_policy(args.base_model)
    results = []
    for index, row in enumerate(eval_rows, start=1):
        base_answer, prompt = generate_answer(base_model, tokenizer, row["question"], args.max_new_tokens)
        results.append(
            {
                "id": row["id"],
                "role": row["role"],
                "difficulty": row["difficulty"],
                "question_type": row["question_type"],
                "question": row["question"],
                "expected_points_json": json.dumps(row["expected_points"], ensure_ascii=False),
                "prompt": prompt,
                "base_answer": base_answer,
                "sft_answer": "",
                "dpo_answer": "",
            }
        )
        print(f"Base {index}/{len(eval_rows)}: {row['id']}")

    print(f"Loading SFT adapter: {args.sft_adapter}")
    sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    sft_model.eval()
    for index, row in enumerate(results, start=1):
        row["sft_answer"], _ = generate_answer(sft_model, tokenizer, row["question"], args.max_new_tokens)
        print(f"SFT  {index}/{len(results)}: {row['id']}")

    if args.dpo_adapter:
        base_model = sft_model.unload()  # 卸载 sft adapter, 回到干净的 base 再挂 dpo
        del sft_model
        torch.cuda.empty_cache()
        print(f"Loading DPO adapter: {args.dpo_adapter}")
        dpo_model = PeftModel.from_pretrained(base_model, args.dpo_adapter)
        dpo_model.eval()
        for index, row in enumerate(results, start=1):
            row["dpo_answer"], _ = generate_answer(dpo_model, tokenizer, row["question"], args.max_new_tokens)
            print(f"DPO  {index}/{len(results)}: {row['id']}")

    models = ["base", "sft"] + (["dpo"] if args.dpo_adapter else [])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions": len(results),
        "eval_file": str(args.eval_file),
        "base_model": args.base_model,
        "sft_adapter": args.sft_adapter,
        "dpo_adapter": args.dpo_adapter,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    for m in models:
        summary[m] = model_stats([r[f"{m}_answer"] for r in results])

    if args.rm_model:
        del sft_model
        torch.cuda.empty_cache()
        print(f"Loading reward model: {args.rm_model}")
        rm_model, rm_tokenizer = load_reward_model(args.rm_model, args.rm_adapter)
        for index, row in enumerate(results, start=1):
            for m in models:
                row[f"rm_{m}"] = reward_score(
                    rm_model, rm_tokenizer, row["question"], row[f"{m}_answer"], args.rm_input_format
                )
            print(f"RM   {index}/{len(results)}: {row['id']}")
        summary["rm_model"] = args.rm_model
        summary["rm_adapter"] = args.rm_adapter
        summary["rm_input_format"] = args.rm_input_format
        summary["reward"] = reward_summary(results, args.seed, models)

    write_outputs(args, results, summary, models)
    print(f"Done. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
