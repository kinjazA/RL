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

add_md(
    '# SFT Training - Interview Feedback Assistant\n'
    '**Phase 1: Supervised Fine-Tuning** on Qwen2.5-3B-Instruct with QLoRA\n\n'
    'Hardware: T4 GPU (free Colab, 16GB VRAM) is sufficient.\n\n'
    'Training time: ~1.5 hours for 3 epochs on 5,558 QA pairs.'
)

add_code(
    '# @title 1. Install Dependencies & Clone Repo\n'
    '!pip install -q transformers==4.44.0 peft==0.12.0 bitsandbytes==0.43.3 '
    'trl==0.9.6 datasets==2.20.0 accelerate==0.33.0\n'
    '!git clone https://github.com/kinjazA/RL.git /content/RL\n'
    '!ls /content/RL/data/',
    'cell-1'
)

add_code(
    '# @title 2. Imports & GPU Check\n'
    'import torch\n'
    'import pandas as pd\n'
    'from datasets import Dataset\n'
    'from transformers import (\n'
    '    AutoTokenizer,\n'
    '    AutoModelForCausalLM,\n'
    '    BitsAndBytesConfig,\n'
    '    TrainingArguments,\n'
    ')\n'
    'from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n'
    'from trl import SFTTrainer\n'
    '\n'
    'print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
    'props = torch.cuda.get_device_properties(0)\n'
    'print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")',
    'cell-2'
)

add_code(
    '# @title 3. Load & Inspect Dataset\n'
    'df = pd.read_csv("/content/RL/data/merged_sft_dataset_cleaned.csv", encoding="utf-8-sig")\n'
    'print(f"Total rows: {len(df)}")\n'
    'print()\n'
    'print("Source distribution:")\n'
    'print(df["source"].value_counts().to_string())\n'
    'print()\n'
    'print("--- Sample records ---")\n'
    'for src in df["source"].unique():\n'
    '    row = df[df["source"] == src].iloc[0]\n'
    '    print()\n'
    '    print(f"[{src}]")\n'
    '    print(f"  Q: {str(row[\'question\'])[:120]}")\n'
    '    print(f"  A: {str(row[\'answer\'])[:150]}")',
    'cell-3'
)

add_code(
    '# @title 4. Format Data for SFT (ChatML)\n'
    '\n'
    'NEWLINE = chr(10)  # actual newline, avoids escaping issues\n'
    '\n'
    'def format_row(row):\n'
    '    q = str(row["question"])\n'
    '    a = str(row["answer"])\n'
    '    parts = [\n'
    '        "<|im_start|>user",\n'
    '        q + "<|im_end|>",\n'
    '        "<|im_start|>assistant",\n'
    '        a + "<|im_end|>",\n'
    '    ]\n'
    '    return {"text": NEWLINE.join(parts)}\n'
    '\n'
    'records = [format_row(row) for _, row in df.iterrows()]\n'
    'dataset = Dataset.from_list(records)\n'
    'dataset = dataset.train_test_split(test_size=0.05, seed=42)\n'
    '\n'
    'n_tr = len(dataset["train"])\n'
    'n_val = len(dataset["test"])\n'
    'print(f"Train: {n_tr:,}  |  Val: {n_val:,}")\n'
    'print()\n'
    'print("--- Sample ---")\n'
    'print(repr(dataset["train"][0]["text"][:400]))',
    'cell-4'
)

add_code(
    '# @title 5. Load Qwen2.5-3B-Instruct with 4-bit QLoRA\n'
    'MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"\n'
    '\n'
    'bnb_config = BitsAndBytesConfig(\n'
    '    load_in_4bit=True,\n'
    '    bnb_4bit_compute_dtype=torch.bfloat16,\n'
    '    bnb_4bit_quant_type="nf4",\n'
    '    bnb_4bit_use_double_quant=True,\n'
    ')\n'
    '\n'
    'model = AutoModelForCausalLM.from_pretrained(\n'
    '    MODEL_NAME,\n'
    '    quantization_config=bnb_config,\n'
    '    device_map="auto",\n'
    '    trust_remote_code=True,\n'
    ')\n'
    'model = prepare_model_for_kbit_training(model)\n'
    '\n'
    'lora_config = LoraConfig(\n'
    '    r=16,\n'
    '    lora_alpha=32,\n'
    '    target_modules=[\n'
    '        "q_proj", "k_proj", "v_proj", "o_proj",\n'
    '        "gate_proj", "up_proj", "down_proj"\n'
    '    ],\n'
    '    lora_dropout=0.05,\n'
    '    bias="none",\n'
    '    task_type="CAUSAL_LM",\n'
    ')\n'
    'model = get_peft_model(model, lora_config)\n'
    'model.print_trainable_parameters()\n'
    '\n'
    'tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)\n'
    'tokenizer.pad_token = tokenizer.eos_token\n'
    'tokenizer.padding_side = "right"',
    'cell-5'
)

add_code(
    '# @title 6. Train\n'
    'output_dir = "/content/RL/sft_output"\n'
    '\n'
    'n_train = len(dataset["train"])\n'
    'batch_size = 4\n'
    'grad_accum = 4\n'
    'epochs = 3\n'
    'total_steps = (n_train // (batch_size * grad_accum)) * epochs\n'
    'warmup = max(1, int(total_steps * 0.03))\n'
    'print(f"Train samples: {n_train}")\n'
    'print(f"Total steps: {total_steps}  |  Warmup steps: {warmup}")\n'
    '\n'
    'training_args = TrainingArguments(\n'
    '    output_dir=output_dir,\n'
    '    per_device_train_batch_size=batch_size,\n'
    '    gradient_accumulation_steps=grad_accum,\n'
    '    num_train_epochs=epochs,\n'
    '    learning_rate=2e-4,\n'
    '    lr_scheduler_type="cosine",\n'
    '    warmup_steps=warmup,\n'
    '    bf16=True,\n'
    '    logging_steps=10,\n'
    '    save_steps=200,\n'
    '    save_total_limit=2,\n'
    '    eval_strategy="steps",\n'
    '    eval_steps=200,\n'
    '    load_best_model_at_end=True,\n'
    '    metric_for_best_model="eval_loss",\n'
    '    report_to="none",\n'
    '    remove_unused_columns=False,\n'
    ')\n'
    '\n'
    'trainer = SFTTrainer(\n'
    '    model=model,\n'
    '    args=training_args,\n'
    '    train_dataset=dataset["train"],\n'
    '    eval_dataset=dataset["test"],\n'
    '    tokenizer=tokenizer,\n'
    '    max_seq_length=1024,\n'
    '    dataset_text_field="text",\n'
    ')\n'
    '\n'
    'trainer.train()\n'
    'print("Training complete!")',
    'cell-6'
)

add_code(
    '# @title 7. Save to Google Drive\n'
    'from google.colab import drive\n'
    'drive.mount("/content/drive")\n'
    '\n'
    'save_path = "/content/drive/MyDrive/RL_sft_adapter"\n'
    'trainer.model.save_pretrained(save_path)\n'
    'tokenizer.save_pretrained(save_path)\n'
    'print(f"Saved to {save_path}")\n'
    '!ls {save_path}',
    'cell-7'
)

add_code(
    '# @title 8. Quick Inference Test\n'
    'test_questions = [\n'
    '    ("What does a Data Scientist do?", "career"),\n'
    '    ("Tell me about a time you failed.", "interview"),\n'
    '    ("Explain gradient descent.", "technical"),\n'
    ']\n'
    '\n'
    'model.eval()\n'
    'NEWLINE = chr(10)\n'
    'for q, tag in test_questions:\n'
    '    parts = [\n'
    '        "<|im_start|>user",\n'
    '        q + "<|im_end|>",\n'
    '        "<|im_start|>assistant",\n'
    '    ]\n'
    '    prompt = NEWLINE.join(parts) + NEWLINE\n'
    '    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)\n'
    '    with torch.no_grad():\n'
    '        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)\n'
    '    response = tokenizer.decode(outputs[0], skip_special_tokens=False)\n'
    '    answer = response.split("<|im_start|>assistant" + NEWLINE)[-1]\n'
    '    print()\n'
    '    print("=" * 60)\n'
    '    print(f"[{tag}] Q: {q}")\n'
    '    print(f"A: {answer[:300]}")',
    'cell-8'
)

with open(r'c:\Users\laiyouhua\Desktop\work\rlProgram\RL\SFT_Training_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook written successfully')
