"""Gradio demo for the Qwen2.5-3B interview assistant training stages."""

import os

if os.path.isdir("/data"):
    os.environ.setdefault("HF_HOME", "/data/.cache/huggingface")

import gc
import threading
import time
from contextlib import nullcontext

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SFT_ADAPTER = os.getenv(
    "SFT_ADAPTER", "Shawnno/qwen2.5-3b-interview-sft-lora"
)
DPO_ADAPTER = os.getenv(
    "DPO_ADAPTER", "Shawnno/qwen2.5-3b-interview-dpo-lora"
)

STAGES = (
    ("Base", None),
    ("SFT", "sft"),
    ("SFT + DPO", "dpo"),
)
SENTENCE_END = set("。！？….!?")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_tokenizer = None
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def _load_runtime():
    """Load one quantized base model and register both LoRA adapters once."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return _model, _tokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
        model = PeftModel.from_pretrained(
            base_model,
            SFT_ADAPTER,
            adapter_name="sft",
            is_trainable=False,
        )
        model.load_adapter(DPO_ADAPTER, adapter_name="dpo", is_trainable=False)
        model.eval()
        model.config.use_cache = True

        _model = model
        _tokenizer = tokenizer
        return _model, _tokenizer


def ensure_natural_ending(text: str) -> str:
    """Cut a length-limited answer back to its last complete sentence."""
    text = text.rstrip()
    if not text or text[-1] in SENTENCE_END:
        return text
    for index in range(len(text) - 1, -1, -1):
        if text[index] in SENTENCE_END:
            return text[: index + 1]
    return text


def _generate(
    model,
    tokenizer,
    question: str,
    adapter_name: str | None,
    mode: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[str, float]:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": mode == "自由采样",
        "pad_token_id": tokenizer.eos_token_id,
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs.update(temperature=float(temperature), top_p=float(top_p))

    if adapter_name is None:
        adapter_context = model.disable_adapter()
    else:
        model.set_adapter(adapter_name)
        adapter_context = nullcontext()

    started_at = time.perf_counter()
    with adapter_context, torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output[0, input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if new_tokens.shape[0] >= int(max_new_tokens):
        answer = ensure_natural_ending(answer)
    return answer, elapsed


def run(
    question: str,
    mode: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    question = question.strip()
    if not question:
        return "", "", "", "请输入一道面试题。"

    with _generation_lock:
        try:
            model, tokenizer = _load_runtime()
        except Exception as exc:
            message = f"模型加载失败：{type(exc).__name__}: {exc}"
            return message, message, message, message

        answers = []
        timings = []
        for label, adapter_name in STAGES:
            try:
                answer, elapsed = _generate(
                    model,
                    tokenizer,
                    question,
                    adapter_name,
                    mode,
                    temperature,
                    top_p,
                    max_new_tokens,
                )
                answers.append(answer)
                timings.append(f"{label} {elapsed:.1f}s")
            except Exception as exc:
                answers.append(f"生成失败：{type(exc).__name__}: {exc}")
                timings.append(f"{label} failed")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return (*answers, "完成 · " + " · ".join(timings))


EXAMPLES = [
    ["请解释机器学习中的偏差和方差，它们分别过高时通常会出现什么现象？"],
    ["请解释 JVM 的垃圾回收机制。什么是 Minor GC、Major GC 和 Full GC？"],
    ["有一份 CSV 文件包含 name、age、score 三列，请用 Python 读取它，按 score 降序输出前 10 名，并处理文件不存在等异常。"],
    ["请说一个你实际影响工作的短板，以及你在工作中采取的改进措施。"],
]

CSS = """
.model-output textarea { min-height: 360px !important; }
.status-line { color: var(--body-text-color-subdued); }
"""

with gr.Blocks(
    title="Qwen2.5-3B 面试助手",
    theme=gr.themes.Soft(radius_size=gr.themes.sizes.radius_sm),
    css=CSS,
) as app:
    gr.Markdown(
        f"""# Qwen2.5-3B 面试助手
对比同一道题在 **Base → SFT → SFT + DPO** 三个训练阶段的回答。

`{BASE_MODEL}` · 设备：**{_device.upper()}**
"""
    )

    question_input = gr.Textbox(
        label="面试问题",
        placeholder="输入一道面试题",
        lines=3,
    )

    with gr.Accordion("生成参数", open=False):
        mode_input = gr.Radio(
            ["稳定对比", "自由采样"], value="稳定对比", label="生成模式"
        )
        with gr.Row():
            temperature_input = gr.Slider(
                0.1, 1.5, value=0.7, step=0.1, label="Temperature"
            )
            top_p_input = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_tokens_input = gr.Slider(
                128, 600, value=600, step=8, label="最大新 Token"
            )

    run_button = gr.Button("运行三路对比", variant="primary")
    status_output = gr.Markdown("就绪", elem_classes=["status-line"])

    with gr.Row(equal_height=True):
        base_output = gr.Textbox(
            label="Base",
            lines=18,
            interactive=False,
            elem_classes=["model-output"],
        )
        sft_output = gr.Textbox(
            label="SFT",
            lines=18,
            interactive=False,
            elem_classes=["model-output"],
        )
        dpo_output = gr.Textbox(
            label="SFT + DPO",
            lines=18,
            interactive=False,
            elem_classes=["model-output"],
        )

    gr.Examples(EXAMPLES, inputs=question_input, label="示例问题")
    gr.Markdown(
        """### 独立测试集结果
| 指标（64 题，贪心解码） | Base | SFT | SFT + DPO |
|---|---:|---:|---:|
| 回答中位字数 | 883 | 345 | 443 |
| 自然收尾率 | 100% | 100% | 100% |
| 强 5-gram 重复率 | 90.6% | 9.4% | 10.9% |

完整评估结果见项目仓库 `eval/results/dpo_acceptance_v3/`。
"""
    )

    inputs = [
        question_input,
        mode_input,
        temperature_input,
        top_p_input,
        max_tokens_input,
    ]
    outputs = [base_output, sft_output, dpo_output, status_output]
    run_button.click(run, inputs, outputs, concurrency_limit=1)
    question_input.submit(run, inputs, outputs, concurrency_limit=1)

app.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
