"""Generate deterministic Base-vs-SFT comparisons on the frozen evaluation set.

Optional reward-model scoring is deliberately separated from generation. The
reward-model prompt must match the format used during RM training.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument(
        "--sft_adapter", default="Shawnno/qwen2.5-3b-interview-sft-lora"
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "eval" / "results" / "sft_acceptance_v1")
    parser.add_argument("--max_new_tokens", type=int, default=384)
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), prompt


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


def length_summary(rows: list[dict], answer_key: str) -> dict[str, float]:
    values = [len(row[answer_key]) for row in rows]
    return {
        "mean_chars": round(float(np.mean(values)), 2),
        "median_chars": round(safe_median(values), 2),
        "p90_chars": round(float(np.percentile(values, 90)), 2),
        "max_chars": int(max(values)),
    }


def reward_summary(rows: list[dict], seed: int) -> dict[str, float]:
    base = np.array([row["rm_base"] for row in rows], dtype=float)
    sft = np.array([row["rm_sft"] for row in rows], dtype=float)
    delta = sft - base
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(5000, len(rows)))
    bootstrap_delta = delta[indices].mean(axis=1)
    bootstrap_win = (delta[indices] > 0).mean(axis=1)
    return {
        "base_mean": round(float(base.mean()), 4),
        "sft_mean": round(float(sft.mean()), 4),
        "mean_delta": round(float(delta.mean()), 4),
        "mean_delta_ci95_low": round(float(np.percentile(bootstrap_delta, 2.5)), 4),
        "mean_delta_ci95_high": round(float(np.percentile(bootstrap_delta, 97.5)), 4),
        "sft_win_rate": round(float((delta > 0).mean()), 4),
        "tie_rate": round(float(np.isclose(delta, 0.0, atol=1e-6).mean()), 4),
        "win_rate_ci95_low": round(float(np.percentile(bootstrap_win, 2.5)), 4),
        "win_rate_ci95_high": round(float(np.percentile(bootstrap_win, 97.5)), 4),
    }


def write_outputs(args: argparse.Namespace, rows: list[dict], summary: dict) -> None:
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
    lines = [
        "# Base vs SFT Acceptance Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Base: `{summary['base_model']}`",
        f"- SFT adapter: `{summary['sft_adapter']}`",
        f"- Decoding: greedy (`do_sample=false`), max_new_tokens={summary['max_new_tokens']}",
        "",
        "## Length",
        "",
        f"- Base: {summary['base_length']}",
        f"- SFT: {summary['sft_length']}",
    ]
    if "reward_model" in summary:
        lines.extend(
            [
                "",
                "## Reward Model (paired, not an absolute quality score)",
                "",
                f"- RM: `{summary['reward_model']}`",
                f"- Input format: `{summary['rm_input_format']}`",
                f"- Stats: {summary['reward']}",
            ]
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            }
        )
        print(f"Base {index}/{len(eval_rows)}: {row['id']}")

    print(f"Loading SFT adapter: {args.sft_adapter}")
    sft_model = PeftModel.from_pretrained(base_model, args.sft_adapter)
    sft_model.eval()
    for index, row in enumerate(results, start=1):
        row["sft_answer"], _ = generate_answer(sft_model, tokenizer, row["question"], args.max_new_tokens)
        print(f"SFT  {index}/{len(results)}: {row['id']}")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions": len(results),
        "eval_file": str(args.eval_file),
        "base_model": args.base_model,
        "sft_adapter": args.sft_adapter,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "base_length": length_summary(results, "base_answer"),
        "sft_length": length_summary(results, "sft_answer"),
    }

    if args.rm_model:
        del sft_model
        torch.cuda.empty_cache()
        print(f"Loading reward model: {args.rm_model}")
        rm_model, rm_tokenizer = load_reward_model(args.rm_model, args.rm_adapter)
        for index, row in enumerate(results, start=1):
            row["rm_base"] = reward_score(rm_model, rm_tokenizer, row["question"], row["base_answer"], args.rm_input_format)
            row["rm_sft"] = reward_score(rm_model, rm_tokenizer, row["question"], row["sft_answer"], args.rm_input_format)
            print(f"RM   {index}/{len(results)}: {row['id']}")
        summary["reward_model"] = args.rm_model
        summary["rm_adapter"] = args.rm_adapter
        summary["rm_input_format"] = args.rm_input_format
        summary["reward"] = reward_summary(results, args.seed)

    write_outputs(args, results, summary)
    print(f"Done. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
