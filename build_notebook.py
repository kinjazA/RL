import json

nb = {
    'cells': [],
    'metadata': {
        'colab': {'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'accelerator': 'GPU'
    },
    'nbformat': 4,
    'nbformat_minor': 0
}

def add_md(source):
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [source]
    })

def add_code(source, cell_id=None):
    meta = {}
    if cell_id:
        meta['id'] = cell_id
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': meta,
        'outputs': [],
        'source': [source]
    })

add_md('# SFT Training — Interview Feedback Assistant\n**Phase 1: Supervised Fine-Tuning** on Qwen2.5-3B-Instruct with QLoRA\n\nHardware: T4 GPU (free Colab, 16GB VRAM) is sufficient.\n\nTraining time: ~1.5 hours for 3 epochs on 5,558 QA pairs.')

add_code('''# @title 1. Install Dependencies & Clone Repo
!pip install -q transformers peft bitsandbytes trl datasets accelerate
!git clone https://github.com/kinjazA/RL.git /content/RL
!ls /content/RL/data/''', 'cell-1')

add_code('''# @title 2. Imports & GPU Check
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print(f"GPU: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")''', 'cell-2')

add_code('''# @title 3. Load & Inspect Dataset
df = pd.read_csv("/content/RL/data/merged_sft_dataset_cleaned.csv", encoding="utf-8-sig")
print(f"Total rows: {len(df)}")
print()
print("Source distribution:")
print(df["source"].value_counts().to_string())
print()
print("--- Sample records ---")
for src in df["source"].unique():
    row = df[df["source"] == src].iloc[0]
    print()
    print(f"[{src}]")
    print(f"  Q: {str(row['question'])[:120]}")
    print(f"  A: {str(row['answer'])[:150]}")''', 'cell-3')

add_code('''# @title 4. Format Data for SFT (ChatML)

BOS = "<|im_start|>"
EOS = "<|im_end|>"

def format_row(row):
    q = str(row["question"])
    a = str(row["answer"])
    lines = [
        f"{BOS}user",
        q + EOS,
        f"{BOS}assistant",
        a + EOS,
    ]
    return {"text": "\\n".join(lines)}

records = [format_row(row) for _, row in df.iterrows()]
dataset = Dataset.from_list(records)
dataset = dataset.train_test_split(test_size=0.05, seed=42)

print(f"Train: {len(dataset['train']):,}  |  Val: {len(dataset['test']):,}")
print()
print("--- Sample ---")
print(dataset["train"][0]["text"][:400])''', 'cell-4')

add_code('''# @title 5. Load Qwen2.5-3B-Instruct with 4-bit QLoRA
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"''', 'cell-5')

add_code('''# @title 6. Train
output_dir = "/content/RL/sft_output"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    max_seq_length=1024,
    dataset_text_field="text",
)

trainer.train()
print("Training complete!")''', 'cell-6')

add_code('''# @title 7. Save to Google Drive
from google.colab import drive
drive.mount("/content/drive")

save_path = "/content/drive/MyDrive/RL_sft_adapter"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")
!ls {save_path}''', 'cell-7')

add_code('''# @title 8. Quick Inference Test
test_questions = [
    ("What does a Data Scientist do?", "career"),
    ("Tell me about a time you failed.", "interview"),
    ("Explain gradient descent.", "technical"),
]

model.eval()
BOS = "<|im_start|>"
EOS = "<|im_end|>"
for q, tag in test_questions:
    lines = [
        f"{BOS}user",
        q + EOS,
        f"{BOS}assistant",
    ]
    prompt = "\\n".join(lines) + "\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    marker = f"{BOS}assistant\\n"
    answer = response.split(marker)[-1]
    print()
    print("=" * 60)
    print(f"[{tag}] Q: {q}")
    print(f"A: {answer[:300]}")''', 'cell-8')

with open(r'c:\Users\laiyouhua\Desktop\work\rlProgram\RL\SFT_Training_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook written successfully')
