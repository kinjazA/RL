import sys, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # space/

import app  # reuses _load_runtime + _generate (same path the live demo would use)

QUESTIONS = [
    "请解释机器学习中的偏差和方差，它们分别过高时通常会出现什么现象？",
    "请解释 JVM 的垃圾回收机制。什么是 Minor GC、Major GC 和 Full GC？",
    "有一份 CSV 文件包含 name、age、score 三列，请用 Python 读取它，按 score 降序输出前 10 名，并处理文件不存在等异常。",
    "请说一个你实际影响工作的短板，以及你在工作中采取的改进措施。",
    "什么是数据库的索引？为什么它能加快查询？它有什么代价？",
]

OUT = os.path.join(HERE, "answers.json")

model, tokenizer = app._load_runtime()

results = {}
for q in QUESTIONS:
    results[q] = {}
    for label, adapter_name in app.STAGES:
        answer, elapsed = app._generate(
            model, tokenizer, q, adapter_name, "稳定对比", 0.7, 0.9, 512
        )
        results[q][label] = {"answer": answer, "elapsed": round(elapsed, 1)}
        print(f"[{label}] {q[:16]}... ({elapsed:.1f}s, {len(answer)} chars)", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("SAVED", OUT, flush=True)
