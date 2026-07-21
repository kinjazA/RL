"""
Compare base Qwen2.5-3B-Instruct vs SFT-adapter outputs.
Toggles LoRA layers on/off — only one model in memory.

Usage:
    # Local adapter
    python -m sft.compare --adapter_path sft_output

    # HF adapter
    python -m sft.compare --adapter_path Shawnno/RL-sft-adapter
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .config import MODEL_NAME, BNB_CONFIG
from .dataset_utils import CHATML_SYSTEM, CHATML_USER, CHATML_ASSISTANT, CHATML_END, NL, SYSTEM_TEXT

# Test questions covering trained domains — none from train.csv
QUESTIONS = [
    ("ML", "Explain gradient descent as if I'm a beginner."),
    ("ML", "What is transfer learning and when would you use it?"),
    ("HR", "Tell me about a time you failed and what you learned."),
    ("HR", "Why do you want to leave your current job?"),
    ("Career", "What does a Data Scientist do day-to-day?"),
    ("SE", "What is the difference between a linked list and an array?"),
    ("SE", "What is a REST API and how does it work?"),
    ("DS", "What is the difference between correlation and causation?"),
]


def strip_answer(full_text: str) -> str:
    """Extract the assistant's reply from a ChatML generation."""
    marker = f"{CHATML_ASSISTANT}{NL}"
    if marker in full_text:
        return full_text.split(marker, 1)[-1].replace(CHATML_END, "").strip()
    return full_text.replace(CHATML_END, "").strip()


def generate(model, tokenizer, question: str, max_tokens: int = 200) -> str:
    prompt = (
        f"{CHATML_SYSTEM}{NL}{SYSTEM_TEXT}{CHATML_END}{NL}"
        f"{CHATML_USER}{NL}{question}{CHATML_END}{NL}"
        f"{CHATML_ASSISTANT}{NL}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    return strip_answer(full)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs SFT model outputs")
    parser.add_argument("--adapter_path", required=True,
                        help="Path to SFT adapter (local dir or HF repo)")
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU — will be very slow.")
    else:
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}  ({gb:.1f} GB)")

    # ---- Load model once ----
    print("\nLoading base model (4-bit QLoRA) ...")
    bnb = BitsAndBytesConfig(**BNB_CONFIG)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Attach adapter ----
    adapter_path = args.adapter_path
    if os.path.isdir(os.path.abspath(adapter_path)):
        adapter_path = os.path.abspath(adapter_path)
        print(f"Loading adapter from local: {adapter_path}")
    else:
        print(f"Loading adapter from HF: {adapter_path}")

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    # ---- Compare ----
    pad = "-" * 70
    for domain, question in QUESTIONS:
        # Base model (disable adapter)
        model.disable_adapter_layers()
        base_answer = generate(model, tokenizer, question, args.max_tokens)

        # SFT model (enable adapter)
        model.enable_adapter_layers()
        sft_answer = generate(model, tokenizer, question, args.max_tokens)

        print(f"\n{pad}")
        print(f"[{domain}] Q: {question}")
        print(f"{pad}")
        print(f"\n  BASE  │ {base_answer[:400]}")
        print(f"\n  SFT   │ {sft_answer[:400]}")


if __name__ == "__main__":
    main()
