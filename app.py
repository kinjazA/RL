"""
Interview Answer Assistant — RLHF Pipeline Demo.
Left panel: SFT model answer. Right panel: Reward Model score.
"""
import os
import torch
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_ADAPTER = os.path.join(BASE_DIR, "sft_output")
RM_ADAPTER = os.path.join(BASE_DIR, "rm", "rm_adapter")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ---- ChatML tokens ----
U = "<|im_start|>user"
A = "<|im_start|>assistant"
E = "<|im_end|>"
N = "\n"

# ---- Globals ----
sft_model = None
sft_tokenizer = None
rm_model = None
rm_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_models_loaded = False

# ---------------------------------------------------------------------------
def _load_sft():
    global sft_model, sft_tokenizer
    if sft_model is not None:
        return
    print(f"[SFT] Loading base model + LoRA adapter on {_device} ...")
    if _device == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb, device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cpu",
            trust_remote_code=True,
        )
    sft_model = PeftModel.from_pretrained(base, SFT_ADAPTER)
    sft_model.eval()
    sft_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    sft_tokenizer.pad_token = sft_tokenizer.eos_token
    print("[SFT] Ready.")


def _load_rm():
    """Load RM model. On CPU, skip entirely — not enough RAM for both models."""
    global rm_model, rm_tokenizer
    if rm_model is not None:
        return
    if _device == "cpu":
        print("[RM] Skipped on CPU (not enough RAM for SFT + RM simultaneously).")
        return
    print(f"[RM] Loading base model + LoRA adapter on {_device} ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, num_labels=1,
    )
    base.config.pad_token_id = base.config.eos_token_id
    rm_model = PeftModel.from_pretrained(base, RM_ADAPTER)
    rm_model.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(RM_ADAPTER, trust_remote_code=True)
    print("[RM] Ready.")


def ensure_models():
    """Lazy-load both models on first request."""
    global _models_loaded
    if _models_loaded:
        return
    try:
        _load_sft()
    except Exception as e:
        raise RuntimeError(f"Failed to load SFT model: {e}")
    try:
        _load_rm()
    except Exception as e:
        print(f"[WARN] RM model failed to load: {e}. Running in answer-only mode.")
    _models_loaded = True


# ---------------------------------------------------------------------------
def do_generate(question: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    text = f"{U}{N}{question}{E}{N}{A}{N}"
    inputs = sft_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(sft_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = sft_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=sft_tokenizer.eos_token_id,
        )
    full = sft_tokenizer.decode(out[0], skip_special_tokens=False)
    answer = full.split(f"{A}{N}")[-1].replace(E, "").strip()
    return answer


def do_score(question: str, answer: str) -> float:
    if rm_model is None:
        return None
    text = f"{U}{N}{question}{E}{N}{A}{N}{answer}{E}"
    inputs = rm_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(rm_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = rm_model(**inputs)
    return out.logits[0, 0].item()


# ---------------------------------------------------------------------------
def score_bar(score_val) -> str:
    """Render a coloured gauge for the reward score."""
    if score_val is None:
        return """<div style="text-align:center;padding:20px;color:#888;font-family:system-ui,sans-serif;">
            <div style="font-size:40px;margin-bottom:8px;">🔒</div>
            <p style="font-weight:600;">RM Unavailable</p>
            <p style="font-size:12px;">Reward model needs GPU — upgrade to <b>T4</b> Space for scoring</p>
        </div>"""

    clamped = max(-5.0, min(5.0, score_val))
    pct = (clamped + 5.0) / 10.0 * 100

    if score_val > 1.0:
        color, label = "#22c55e", "High Quality"
    elif score_val > -1.0:
        color, label = "#eab308", "Average"
    else:
        color, label = "#ef4444", "Low Quality"

    return f"""
    <div style="text-align:center;font-family:system-ui,sans-serif;padding:8px 4px;">
        <div style="font-size:52px;font-weight:800;color:{color};line-height:1.1;">{score_val:+.2f}</div>
        <div style="font-size:12px;color:#888;margin:4px 0;">Reward Score</div>
        <div style="margin:12px 8px;background:#e5e7eb;border-radius:6px;height:20px;overflow:hidden;position:relative;">
            <div style="background:linear-gradient(90deg,#ef4444 0%,#eab308 50%,#22c55e 100%);width:100%;height:100%;"></div>
            <div style="position:absolute;top:-3px;left:{pct:.1f}%;width:4px;height:26px;background:#1f2937;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 4px rgba(0,0,0,.3);"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#aaa;padding:0 8px;">
            <span>-5</span><span>0</span><span>+5</span>
        </div>
        <div style="margin-top:12px;display:inline-block;padding:5px 18px;border-radius:20px;background:{color}18;color:{color};font-weight:700;font-size:13px;border:1px solid {color}40;">
            {label}
        </div>
    </div>"""


def run(question: str, max_tokens: int, temperature: float):
    if not question.strip():
        return "", score_bar(0)
    try:
        ensure_models()
        answer = do_generate(question.strip(), max_tokens=max_tokens, temperature=temperature)
        s = do_score(question.strip(), answer)
        return answer, score_bar(s)
    except Exception as exc:
        return f"**Error:** {exc}", score_bar(0)


# ---------------------------------------------------------------------------
EXAMPLES = [
    "What does a Data Scientist do day-to-day?",
    "Tell me about a time you faced a conflict at work and how you resolved it.",
    "Explain gradient descent as if I'm a beginner.",
    "What is the difference between a linked list and an array?",
    "Why do you want to leave your current job?",
]

with gr.Blocks(title="Interview Answer Assistant — RLHF", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # Interview Answer Assistant
    ### RLHF Pipeline: SFT Model + Reward Model side-by-side
    See what the fine-tuned model generates **and** how the reward model judges it.
    > **CPU mode:** RM scoring disabled — only answer generation is available.
    """)

    with gr.Row():
        q = gr.Textbox(
            label="Interview Question",
            placeholder="Type an interview question, e.g. 'Explain overfitting in machine learning'",
            lines=2, scale=4,
        )

    with gr.Row():
        btn = gr.Button("Generate & Score", variant="primary")

    with gr.Accordion("Generation Settings", open=False):
        with gr.Row():
            max_tok = gr.Slider(64, 512, value=256, step=32, label="Max New Tokens")
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")

    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.Markdown("### Model Answer")
            out_answer = gr.Textbox(
                label="Generated Answer", lines=14, interactive=False,
                show_copy_button=True, elem_id="answer-box",
            )
        with gr.Column(scale=1):
            gr.Markdown("### Reward Score")
            out_score = gr.HTML(label="Score Gauge")

    gr.Examples(EXAMPLES, inputs=q, label="Try an Example Question")

    btn.click(run, [q, max_tok, temp], [out_answer, out_score])
    q.submit(run, [q, max_tok, temp], [out_answer, out_score])

    # Status bar
    gr.Markdown(
        f"<div style='text-align:center;color:#888;font-size:11px;margin-top:20px;'>"
        f"Device: **{_device.upper()}** &nbsp;|&nbsp; "
        f"Base: Qwen2.5-3B-Instruct &nbsp;|&nbsp; "
        f"QLoRA rank=16 alpha=32 &nbsp;|&nbsp; "
        f"RM accuracy: 87.6% &nbsp;|&nbsp; margin: 28.9"
        f"</div>"
    )

if __name__ == "__main__":
    print(f"Device: {_device}")
    print("Models will load on first request (lazy init).")
    app.launch(server_name="0.0.0.0", server_port=7860)
