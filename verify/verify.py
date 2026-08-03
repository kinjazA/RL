# 加载训好的 LoRA adapter, 用真题验证面试回答风格
# 用法:
#   python verify.py                      # 用默认 4 个真题
#   python verify.py "你的自定义问题"      # 测自定义问题
#   python verify.py --questions 问题1 问题2 ...
import sys

from llamafactory.chat import ChatModel

DEFAULT_QUESTIONS = [
    # 算法岗 - 概念题 (对应 human_seed 风格样本)
    "请解释机器学习中的偏差和方差，它们分别过高时通常会出现什么现象？",
    # 后端岗 - 概念题
    "请解释JVM的垃圾回收（GC）机制。什么是Minor GC、Major GC和Full GC？它们有什么区别？CMS、G1、ZGC垃圾收集器各有什么特点？",
    # 数据分析岗 - 场景题
    "有一份CSV文件，包含name、age、score三列。请用Python编写程序读取它，按score从高到低排序，输出前10名学生的名字。要求处理文件不存在等异常情况。",
    # 行为面试题 - 开放题
    "请说一个你实际影响工作的短板，以及你在工作中采取的改进措施。",
]

# 换成你训练出来的 adapter 路径
ADAPTER = "saves/Qwen2.5-3B-Instruct/lora/sft-cli/checkpoint-1212"


def main():
    if "--questions" in sys.argv:
        i = sys.argv.index("--questions")
        questions = sys.argv[i + 1:]
    elif len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = DEFAULT_QUESTIONS

    chat = ChatModel({
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_name_or_path": ADAPTER,
        "template": "qwen",
        "infer_backend": "huggingface",
        "trust_remote_code": True,
    })

    print("=" * 60)
    print("微调模型面试回答验证 (Qwen2.5-3B-Instruct + LoRA)")
    print("=" * 60)

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"[问题 {i}] {q}")
        print(f"{'='*60}")
        result = chat.chat([{"role": "user", "content": q}])[0]
        print(f"[回答] {result.response_text}")
        print()


if __name__ == "__main__":
    main()
