"""
Quick inference test for the trained SFT model.
Usage:
    python -m sft.inference --adapter_path /content/RL/sft_output
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .config import MODEL_NAME, BNB_CONFIG
from .dataset_utils import CHATML_USER, CHATML_ASSISTANT, CHATML_END, NL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True, help="Path to saved LoRA adapter")
    args = parser.parse_args()

    bnb = BitsAndBytesConfig(**BNB_CONFIG)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)

    questions = [
        "What does a Data Scientist do?",
        "Tell me about a time you failed.",
        "Explain gradient descent.",
    ]

    for q in questions:
        prompt = f"{CHATML_USER}{NL}{q}{CHATML_END}{NL}{CHATML_ASSISTANT}{NL}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
        full = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract assistant reply
        marker = f"{CHATML_ASSISTANT}{NL}"
        answer = full.split(marker)[-1].replace(CHATML_END, "").strip()
        print(f"\n{'='*60}\nQ: {q}\nA: {answer[:400]}")


if __name__ == "__main__":
    main()
