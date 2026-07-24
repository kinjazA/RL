"""Shared ChatML helpers for SFT training, inference, demo, and PPO prompts."""

CHATML_SYSTEM = "<|im_start|>system"
CHATML_USER = "<|im_start|>user"
CHATML_ASSISTANT = "<|im_start|>assistant"
CHATML_END = "<|im_end|>"
NL = "\n"

SYSTEM_TEXT = (
    "You are a job candidate in an interview. "
    "Answer in first person with specific examples. "
    "Be concise, factual, and professional."
)


def build_prompt(question: str, system_text: str = SYSTEM_TEXT) -> str:
    """Build the generation prompt up to the assistant turn."""
    question = str(question).strip()
    return (
        f"{CHATML_SYSTEM}{NL}{system_text}{CHATML_END}{NL}"
        f"{CHATML_USER}{NL}{question}{CHATML_END}{NL}"
        f"{CHATML_ASSISTANT}{NL}"
    )


def build_full(question: str, answer: str, system_text: str = SYSTEM_TEXT) -> str:
    """Build a complete single-turn ChatML sample."""
    answer = str(answer).strip()
    return f"{build_prompt(question, system_text)}{answer}{CHATML_END}{NL}"


def strip_assistant_answer(full_text: str) -> str:
    """Extract the final assistant message from a decoded ChatML generation."""
    marker = f"{CHATML_ASSISTANT}{NL}"
    if marker in full_text:
        return full_text.split(marker)[-1].replace(CHATML_END, "").strip()
    return full_text.replace(CHATML_END, "").strip()
