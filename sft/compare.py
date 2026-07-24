"""Compare base Qwen2.5-3B-Instruct vs an SFT LoRA adapter.

Colab-friendly usage:

    python -m sft.compare --adapter_path Shawnno/RL-sft-adapter

Optional held-out CSV sampling:

    python -m sft.compare --adapter_path Shawnno/RL-sft-adapter \
      --question_set both --csv_path data/sft_test_v2.csv --samples_per_source 2
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Iterable

from .chatml import build_prompt, strip_assistant_answer
from .config import BNB_CONFIG, MODEL_NAME


@dataclass
class EvalQuestion:
    domain: str
    question: str
    source: str = "curated"
    reference: str = ""
    intent: str = ""


CURATED_QUESTIONS = [
    EvalQuestion(
        "behavioral",
        "Tell me about a time you disagreed with a stakeholder and how you handled it.",
        intent="first-person, concrete situation/action/result, not generic advice",
    ),
    EvalQuestion(
        "behavioral",
        "Describe a time you made a mistake at work. What did you learn?",
        intent="candidate persona, ownership, learning, no refusal",
    ),
    EvalQuestion(
        "career",
        "What does a Data Scientist do day to day?",
        intent="role-specific, concrete tasks and tools",
    ),
    EvalQuestion(
        "career",
        "Why are you interested in this Product Manager role?",
        intent="candidate motivation without sounding templated",
    ),
    EvalQuestion(
        "product",
        "How would you prioritize features when engineering capacity is limited?",
        intent="structured prioritization, tradeoffs, product judgment",
    ),
    EvalQuestion(
        "ml",
        "Explain gradient descent as if I am a beginner.",
        intent="beginner-friendly explanation, not dry memorization",
    ),
    EvalQuestion(
        "ml",
        "What is overfitting, and how would you reduce it?",
        intent="correct definition plus practical mitigations",
    ),
    EvalQuestion(
        "ml",
        "What is transfer learning and when would you use it?",
        intent="correct concept, realistic use cases",
    ),
    EvalQuestion(
        "ds",
        "What is the difference between correlation and causation?",
        intent="correct distinction and interview-ready example",
    ),
    EvalQuestion(
        "ds",
        "What do precision and recall measure, and when would recall matter more?",
        intent="correct metrics and practical tradeoff",
    ),
    EvalQuestion(
        "se",
        "What is the difference between a linked list and an array?",
        intent="must not claim linked-list random access is O(1) or O(log n)",
    ),
    EvalQuestion(
        "se",
        "What is a REST API and how does it work?",
        intent="accurate, concise system-design explanation",
    ),
    EvalQuestion(
        "se",
        "What is Big O notation, and why is it important?",
        intent="accurate complexity explanation",
    ),
    EvalQuestion(
        "general",
        "Explain APIs to a non-technical teammate.",
        intent="clear communication, no overcomplication",
    ),
    EvalQuestion(
        "out_of_scope",
        "Explain the time value of money in simple terms.",
        intent="smoke test only: coherent answer, no word salad",
    ),
]


def set_seed(seed: int):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_sources(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def load_csv_questions(
    csv_path: str,
    samples_per_source: int,
    seed: int,
    csv_sources: str,
) -> list[EvalQuestion]:
    """Sample held-out questions from sft_test_v2.csv by source."""
    rng = random.Random(seed)
    allowed_sources = parse_csv_sources(csv_sources)
    by_source: dict[str, list[dict[str, str]]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            source = row.get("source", "unknown")
            if allowed_sources is not None and source not in allowed_sources:
                continue
            by_source.setdefault(source, []).append(row)

    questions: list[EvalQuestion] = []
    for source in sorted(by_source):
        rows = by_source[source][:]
        rng.shuffle(rows)
        for row in rows[:samples_per_source]:
            questions.append(
                EvalQuestion(
                    domain=row.get("domain", source),
                    question=row["question"],
                    source=source,
                    reference=row.get("answer", ""),
                    intent="held-out reference available in CSV",
                )
            )
    return questions


def select_questions(args: argparse.Namespace) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    if args.question_set in {"curated", "both"}:
        questions.extend(CURATED_QUESTIONS)
    if args.question_set in {"csv", "both"}:
        questions.extend(
            load_csv_questions(args.csv_path, args.samples_per_source, args.seed, args.csv_sources)
        )
    if args.limit:
        questions = questions[: args.limit]
    return questions


def resolve_adapter_path(adapter_path: str) -> str:
    local_path = os.path.abspath(adapter_path)
    if os.path.isdir(local_path):
        print(f"Loading adapter from local path: {local_path}")
        return local_path
    print(f"Loading adapter from Hugging Face: {adapter_path}")
    return adapter_path


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")
    else:
        print("WARNING: CUDA not available; CPU generation will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if use_cuda and not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**BNB_CONFIG)
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_cuda else torch.float32
        model_kwargs["device_map"] = "auto" if use_cuda else "cpu"

    print(f"Loading base model: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    adapter_path = resolve_adapter_path(args.adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question: str, args: argparse.Namespace) -> str:
    import torch

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)
    full = tokenizer.decode(output[0], skip_special_tokens=False)
    return strip_assistant_answer(full)


def compare_one(model, tokenizer, item: EvalQuestion, args: argparse.Namespace) -> dict[str, str]:
    model.disable_adapter_layers()
    base_answer = generate(model, tokenizer, item.question, args)

    model.enable_adapter_layers()
    sft_answer = generate(model, tokenizer, item.question, args)

    return {
        "domain": item.domain,
        "source": item.source,
        "question": item.question,
        "intent": item.intent,
        "reference": item.reference,
        "base_answer": base_answer,
        "sft_answer": sft_answer,
    }


def print_result(row: dict[str, str]):
    pad = "-" * 100
    print(f"\n{pad}")
    print(f"[{row['domain']}] source={row['source']}")
    print(f"Q: {row['question']}")
    if row["intent"]:
        print(f"Check: {row['intent']}")
    print(pad)
    print(f"\nBASE:\n{row['base_answer']}\n")
    print(f"SFT:\n{row['sft_answer']}\n")
    if row["reference"]:
        print(f"REFERENCE (held-out answer, truncated):\n{row['reference'][:700]}\n")


def write_csv(rows: Iterable[dict[str, str]], path: str):
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to: {path}")


def write_markdown(rows: Iterable[dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Base vs SFT Comparison\n\n")
        for idx, row in enumerate(rows, 1):
            f.write(f"## {idx}. [{row['domain']}] {row['question']}\n\n")
            f.write(f"- Source: `{row['source']}`\n")
            if row["intent"]:
                f.write(f"- Check: {row['intent']}\n")
            f.write("\n### Base\n\n")
            f.write(row["base_answer"].strip() + "\n\n")
            f.write("### SFT\n\n")
            f.write(row["sft_answer"].strip() + "\n\n")
            if row["reference"]:
                f.write("### Reference\n\n")
                f.write(row["reference"].strip() + "\n\n")
    print(f"Markdown written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs SFT model outputs.")
    parser.add_argument("--adapter_path", default="Shawnno/RL-sft-adapter")
    parser.add_argument("--question_set", choices=["curated", "csv", "both"], default="curated")
    parser.add_argument("--csv_path", default=os.path.join("data", "sft_test_v2.csv"))
    parser.add_argument(
        "--csv_sources",
        default="career_qa,local_interview_qa,ml_interview,ds_qa_treasury,se_interview",
        help="Comma-separated held-out sources to sample, or 'all'.",
    )
    parser.add_argument("--samples_per_source", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading.")
    parser.add_argument("--csv_output", default=os.path.join("compare_outputs", "base_vs_sft.csv"))
    parser.add_argument("--md_output", default=os.path.join("compare_outputs", "base_vs_sft.md"))
    args = parser.parse_args()

    set_seed(args.seed)
    questions = select_questions(args)
    print(f"Questions: {len(questions)} ({args.question_set})")
    print(f"Generation: temperature={args.temperature}, max_new_tokens={args.max_new_tokens}")

    model, tokenizer = load_model_and_tokenizer(args)

    rows = []
    for idx, item in enumerate(questions, 1):
        print(f"\nRunning {idx}/{len(questions)}: [{item.domain}] {item.question[:90]}")
        row = compare_one(model, tokenizer, item, args)
        rows.append(row)
        print_result(row)

    write_csv(rows, args.csv_output)
    write_markdown(rows, args.md_output)


if __name__ == "__main__":
    main()
