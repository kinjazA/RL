# SFT configuration for Qwen2.5-3B-Instruct + QLoRA

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---- 4-bit quantization ----
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# ---- LoRA ----
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

# ---- Training ----
TRAINING_ARGS = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 2,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "report_to": "none",
}

MAX_SEQ_LENGTH = 1024
TRAIN_VAL_SPLIT = 0.05
RANDOM_SEED = 42

# ---- Data augmentation ----
OVERSAMPLE_SOURCES = ["se_interview", "ds_qa_treasury"]  # underrepresented domains
OVERSAMPLE_FACTOR = 2  # duplicate each 2x in training set

# Backward-compatible aliases for older notebooks/scripts.
OVERAMPLE_SOURCES = OVERSAMPLE_SOURCES
OVERAMPLE_FACTOR = OVERSAMPLE_FACTOR

# ---- NEFTune ----
NEFTUNE_NOISE_ALPHA = 5  # embedding noise to reduce overfitting (TRL>=0.9)
