"""
SFT training entry point.
Usage:
    python -m sft.train --csv_path /content/RL/data/train.csv --output_dir /content/RL/sft_output
"""

import argparse
import sys

import torch
from trl import SFTTrainer

from .config import TRAINING_ARGS, MAX_SEQ_LENGTH
from .dataset_utils import load_and_format
from .model_utils import load_model_and_tokenizer

# SFTConfig available from trl>=0.9; fall back to TrainingArguments for older versions
try:
    from trl import SFTConfig as _SFTConfig
except ImportError:
    from transformers import TrainingArguments as _SFTConfig


def main():
    parser = argparse.ArgumentParser(description="SFT train Qwen2.5-3B with QLoRA")
    parser.add_argument("--csv_path", required=True, help="Path to merged CSV file")
    parser.add_argument("--output_dir", required=True, help="Where to save the LoRA adapter")
    args = parser.parse_args()

    # ---- Check GPU ----
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. SFT training requires a GPU.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}  ({vram_gb:.1f} GB)")

    # ---- Load data ----
    print("\n[1/3] Loading dataset ...")
    dataset = load_and_format(args.csv_path)

    # ---- Load model ----
    print("\n[2/3] Loading model with 4-bit QLoRA ...")
    model, tokenizer = load_model_and_tokenizer()

    # ---- Tokenize dataset (before trainer to avoid text-column collision) ----
    print("\n[3/3] Tokenizing & training ...")

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    # ---- Training config ----
    sft_kwargs = dict(TRAINING_ARGS)
    sft_kwargs["output_dir"] = args.output_dir
    training_args = _SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test"),
        tokenizer=tokenizer,
    )

    trainer.train()

    # ---- Save ----
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
