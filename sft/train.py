"""
SFT training entry point — v2.
Usage:
    python -m sft.train --csv_path data/sft_train_v2.csv --eval_csv_path data/sft_eval_v2.csv --output_dir sft_output

v2 improvements over v1:
  - Loss only on assistant tokens (DataCollatorForCompletionOnlyLM)
  - NEFTune embedding noise to reduce overfitting
  - System prompt in training examples
  - Oversampling of underrepresented domains (SE, DS)
"""

import argparse
import sys

import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from .config import (
    TRAINING_ARGS,
    MAX_SEQ_LENGTH,
    NEFTUNE_NOISE_ALPHA,
)
from .chatml import CHATML_ASSISTANT, NL
from .dataset_utils import load_and_format
from .model_utils import load_model_and_tokenizer

# Newer TRL (>=0.9) uses SFTConfig; older uses TrainingArguments
try:
    from trl import SFTConfig as _TrainConfig
    _NEW_TRL = True
except ImportError:
    from transformers import TrainingArguments as _TrainConfig
    _NEW_TRL = False


def main():
    parser = argparse.ArgumentParser(description="SFT train Qwen2.5-3B with QLoRA")
    parser.add_argument("--csv_path", required=True, help="Path to train CSV file")
    parser.add_argument("--eval_csv_path", default=None, help="Optional prebuilt eval CSV file")
    parser.add_argument("--output_dir", required=True, help="Where to save the LoRA adapter")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override config learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=None, help="Override config epoch count")
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
    dataset = load_and_format(args.csv_path, eval_csv_path=args.eval_csv_path)

    # ---- Load model ----
    print("\n[2/3] Loading model with 4-bit QLoRA ...")
    model, tokenizer = load_model_and_tokenizer()

    # ---- Data collator: loss only on assistant tokens ----
    response_template = f"{CHATML_ASSISTANT}{NL}"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ---- Train ----
    print("\n[3/3] Training ...")

    sft_kwargs = dict(TRAINING_ARGS)
    sft_kwargs["output_dir"] = args.output_dir
    if args.learning_rate is not None:
        sft_kwargs["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        sft_kwargs["num_train_epochs"] = args.num_train_epochs

    if _NEW_TRL:
        sft_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
        sft_kwargs["dataset_text_field"] = "text"
        sft_kwargs["neftune_noise_alpha"] = NEFTUNE_NOISE_ALPHA
        training_args = _TrainConfig(**sft_kwargs)
        trainer_kwargs = {}
    else:
        training_args = _TrainConfig(**sft_kwargs)
        trainer_kwargs = {
            "max_seq_length": MAX_SEQ_LENGTH,
            "dataset_text_field": "text",
        }

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        data_collator=collator,
        processing_class=tokenizer,
        **trainer_kwargs,
    )

    trainer.train()

    # ---- Save ----
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
