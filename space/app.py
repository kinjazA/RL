"""
面试助手微调效果 Demo — Qwen2.5-3B-Instruct (LoRA SFT).

三栏对比: Base(未微调) vs SFT(你的 LoRA) vs RLHF(未训练)
然后用 reward model 给三个回答打分。

结构说明:
- Base:  Qwen2.5-3B-Instruct 原版
- SFT:   Qwen2.5-3B-Instruct + LoRA adapter (本仓库训练的面试助手)
- RLHF:  当前未训练(项目只做了 SFT), 显示占位说明
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
from peft import PeftModel

# --------------------------------------------------------------------------
# CONFIG — 换成你自己的模型
# --------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"          # 基座(未微调)
LORA_ADAPTER = "Shawnno/qwen2.5-3b-interview-sft-lora"  # 你的 LoRA adapter (HF 或本地路径)
RM_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"

# 三个展示位 -> (加载方式, 标签)
#   None adapter = 纯基座
#   具体 adapter = 基座 + LoRA
#   "not_trained" = 占位(未训练)
STAGES = {
    "Base (未微调)": {"adapter": None, "desc": "Qwen2.5-3B-Instruct 原版"},
    "SFT (面试助手)": {"adapter": LORA_ADAPTER, "desc": "基座 + LoRA SFT 微调"},
    "RLHF (DPO)": {"adapter": "not_trained", "desc": "未训练, 留待下一步"},
}

GEN_KWARGS = dict(max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)

if os.path.isdir("/data"):  # HF Space persistent storage: download models once
    os.environ["HF_HOME"] = "/data/.cache/huggingface"

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_gen(repo: str, adapter=None):
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(repo)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=qc, device_map="auto")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
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
        return ["", "", ""] + [score_html(None)] * 3

    answers, scores = {}, {}
    order = list(STAGES)
    for label in order:
        spec = STAGES[label]
        if spec["adapter"] == "not_trained":
            answers[label] = "（RLHF/DPO 阶段尚未训练，这是下一步计划。）"
            scores[label] = None
            continue
        model = tok = None
        try:
            model, tok = _load_gen(BASE_MODEL, spec["adapter"])
            prompt = _format_prompt(question.strip())
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
        for label in order:
            if STAGES[label]["adapter"] == "not_trained":
                continue
            scores[label] = _rm_score(rm, rmtok, question.strip(), answers[label])
    except Exception:
        for label in order:
            scores[label] = None
    finally:
        _unload(rm)

    return (
        answers[order[0]], answers[order[1]], answers[order[2]],
        score_html(scores[order[0]]), score_html(scores[order[1]]), score_html(scores[order[2]]),
    )


def _format_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"


EXAMPLES = [
    "请解释机器学习中的偏差和方差，它们分别过高时通常会出现什么现象？",
    "请解释JVM的垃圾回收（GC）机制。什么是Minor GC、Major GC和Full GC？",
    "有一份CSV文件包含name、age、score三列，请用Python读取它，按score降序输出前10名。要求处理文件不存在等异常。",
    "请说一个你实际影响工作的短板，以及你在工作中采取的改进措施。",
]

LABELS = list(STAGES)

with gr.Blocks(title="面试助手微调效果 Demo — Qwen2.5-3B", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        f"""# 面试助手微调效果 Demo
### 同一道面试题 → 基座 / SFT微调 / RLHF → reward-model 打分

| | 模型 | 说明 |
|---|---|---|
| **{LABELS[0]}** | `{BASE_MODEL}` | 未微调的原始模型 |
| **{LABELS[1]}** | 基座 + `{LORA_ADAPTER}` | QLoRA SFT 微调后的面试助手 |
| **{LABELS[2]}** | — | 未训练（DPO/RLHF 是下一步） |

三个回答由 reward model `{RM_MODEL}` 打分。
> **首次运行:** 下载基座+adapter (~7GB) 并以 4bit 加载，首次生成较慢，之后很快。
> 设备: **{_device.upper()}**"""
    )

    q = gr.Textbox(label="面试问题", placeholder="输入任意面试题…", lines=3)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"### {LABELS[0]}")
            out_a = gr.Textbox(label="Base 回答", lines=10, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### RM 打分")
            out_a_s = gr.HTML(label="Base score")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"### {LABELS[1]}")
            out_b = gr.Textbox(label="SFT 回答", lines=10, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### RM 打分")
            out_b_s = gr.HTML(label="SFT score")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"### {LABELS[2]}")
            out_c = gr.Textbox(label="RLHF 回答", lines=10, interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### RM 打分")
            out_c_s = gr.HTML(label="RLHF score")

    btn = gr.Button("运行对比", variant="primary")
    gr.Examples(EXAMPLES, inputs=q)

    btn.click(run, q, [out_a, out_b, out_c, out_a_s, out_b_s, out_c_s])
    q.submit(run, q, [out_a, out_b, out_c, out_a_s, out_b_s, out_c_s])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
