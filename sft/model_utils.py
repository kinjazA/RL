"""
Load Qwen2.5-3B-Instruct with 4-bit QLoRA adapter.
Returns (model, tokenizer). Model is a PeftModel ready for training.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import MODEL_NAME, BNB_CONFIG, LORA_CONFIG


def load_model_and_tokenizer():
    """Load 4-bit quantized model with LoRA adapter. Returns (model, tokenizer)."""

    bnb = BitsAndBytesConfig(**BNB_CONFIG)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
