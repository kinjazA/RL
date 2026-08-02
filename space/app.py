"""
RLHF Pipeline Demo — Zephyr 7B.

Shows the same question answered by three stages of the RLHF pipeline:
    Base (pre-SFT)  vs  SFT  vs  RLHF (DPO)
then scores all three answers with a reward model (RM).

Deploys to a T4 Hugging Face Space. Models load sequentially in 4-bit to
fit 16GB VRAM. Swap the HF repo IDs in the CONFIG block to demo another
model family.
"""
import gc
import os
import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --------------------------------------------------------------------------
# CONFIG — swap these HF repo IDs to demo a different RLHF model family
# --------------------------------------------------------------------------
MODELS = {
    "Base (pre-SFT)": "mistralai/Mistral-7B-v0.1",
    "SFT": "HuggingFaceH4/zephyr-7b-sft-full",
    "RLHF (DPO)": "HuggingFaceH4/zephyr-7b-beta",
}
RM_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"

# Zephyr SFT/RLHF models were trained with ChatML; the base model wasn't.
MODEL_FORMAT = {
    "Base (pre-SFT)": "plain",
    "SFT": "chatml",
    "RLHF (DPO)": "chatml",
}

GEN_KWARGS = dict(max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)

if os.path.isdir("/data"):  # HF Space persistent storage: download models once
    os.environ["HF_HOME"] = "/data/.cache/huggingface"

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _format_prompt(fmt: str, question: str) -> str:
    if fmt == "chatml":
        return f"<|user|>\n{question}\n<|assistant|>\n"
    return f"Question: {question}\nAnswer:"


def _load_gen(repo: str):
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=qc, device_map="auto")
    model.eval()
    return model, tok


def _unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _load_rm():
    tok = AutoTokenizer.from_pretrained(RM_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RM_MODEL)
    model.eval()
    return model, tok


def _rm_score(model, tok, question: str, answer: str):
    text = f"{question}\n\n{answer}"
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        return model(**enc).logits[0, 0].item()


def score_html(score) -> str:
    if score is None:
        return "<div style='text-align:center;color:#888;padding:8px;'>RM unavailable</div>"
    if score > 1.0:
        color, label = "#22c55e", "High"
    elif score > -1.0:
        color, label = "#eab308", "Mid"
    else:
        color, label = "#ef4444", "Low"
    return (
        "<div style='text-align:center;font-family:system-ui,sans-serif;padding:8px;'>"
        f"<div style='font-size:34px;font-weight:800;color:{color};'>{score:+.2f}</div>"
        f"<div style='font-size:12px;color:#888;'>{label}</div></div>"
    )


def run(question: str):
    if not question.strip():
        return [""] * 3 + [score_html(None)] * 3

    answers, scores = {}, {}
    for label in MODELS:
        model = tok = None
        try:
            model, tok = _load_gen(MODELS[label])
            prompt = _format_prompt(MODEL_FORMAT[label], question.strip())
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, pad_token_id=tok.eos_token_id, **GEN_KWARGS)
            new_tokens = out[0][enc["input_ids"].shape[1]:]
            answers[label] = tok.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            answers[label] = f"[error] {e}"
        finally:
            if model is not None:
                _unload(model)

    try:
        rm, rmtok = _load_rm()
        for label in MODELS:
            scores[label] = _rm_score(rm, rmtok, question.strip(), answers[label])
    except Exception:
        for label in MODELS:
            scores[label] = None
    finally:
        _unload(rm)

    order = list(MODELS)
    return (
        answers[order[0]], answers[order[1]], answers[order[2]],
        score_html(scores[order[0]]), score_html(scores[order[1]]), score_html(scores[order[2]]),
    )


EXAMPLES = [
    "Explain what the RLHF training pipeline is.",
    "Why is the sky blue?",
    "Write a short motivational speech for a student before an exam.",
    "What are the benefits and risks of artificial intelligence?",
    "Explain the difference between supervised fine-tuning and reinforcement learning.",
]

LABELS = list(MODELS)

with gr.Blocks(title="RLHF Pipeline Demo — Zephyr 7B", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        f"""# RLHF Pipeline Demo
### Same question → three pipeline stages → reward-model scores

| | Model | What it is |
|---|---|---|
| **{LABELS[0]}** | `{MODELS[LABELS[0]]}` | before any fine-tuning |
| **{LABELS[1]}** | `{MODELS[LABELS[1]]}` | after supervised fine-tuning |
| **{LABELS[2]}** | `{MODELS[LABELS[2]]}` | after RLHF (DPO) alignment |

All three answers are scored by the reward model `{RM_MODEL}`.
> **First run:** models download (~45GB) and load in 4-bit. The first
> *Generate* can take several minutes; later runs are fast.
> Device: **{_device.upper()}**"""
    )

    q = gr.Textbox(label="Prompt", placeholder="Type any question…", lines=3)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"### {LABELS[0]}")
            out_a = gr.Textbox(label="Base answer", lines=10, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### RM score")
            out_a_s = gr.HTML(label="Base score")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"### {LABELS[1]}")
            out_b = gr.Textbox(label="SFT answer", lines=10, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### RM score")
            out_b_s = gr.HTML(label="SFT score")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"### {LABELS[2]}")
            out_c = gr.Textbox(label="RLHF answer", lines=10, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### RM score")
            out_c_s = gr.HTML(label="RLHF score")

    btn = gr.Button("Run pipeline", variant="primary")
    gr.Examples(EXAMPLES, inputs=q)

    btn.click(run, q, [out_a, out_b, out_c, out_a_s, out_b_s, out_c_s])
    q.submit(run, q, [out_a, out_b, out_c, out_a_s, out_b_s, out_c_s])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
