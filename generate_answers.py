"""
用强模型按 human_seed 风格批量生成面试回答.

输入: data/rlhf_interview_all_questions_merged.csv
       (3230 道带元数据的面试题; 其中 ~150 条已有答案, 其余答案为空)
输出: 完整保留原列, 只把空答案按风格填上.

风格来源: 文件里 source_type=human_seed 的答案 (ALG_001 偏差方差那种:
          先概念定义 -> 再说现象(冒号展开) -> 一句点本质; 平实口语, 像面试现场作答).
生成时把每道题的 interview_intent + expected_points 一起喂给模型当指导.

用法:
    export LLM_API_KEY=<你的 key>
    python generate_answers.py                                 # 默认 deepseek
    python generate_answers.py --provider qwen --limit 5       # 先跑 5 条试口味
    python generate_answers.py --workers 4

会断点续写: 重复跑自动跳过已有答案的行.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------------------------------------------------------- 配置
PROVIDERS = {
    "deepseek": {"base": "https://api.deepseek.com", "model": "deepseek-chat"},
    "qwen": {"base": "https://dashscope.aliyuncs.com/compatible-mode", "model": "qwen-plus"},
    "moonshot": {"base": "https://api.moonshot.cn", "model": "moonshot-v1-8k"},
    "ollama": {"base": "http://localhost:11434", "model": "llama3.1:8b"},  # 本地, 无需 key
}

DEFAULT_INPUT = "data/rlhf_interview_all_questions_merged.csv"
MIN_ANSWER_LEN = 50  # 答案短于此的视为"需要生成"

# 从 human_seed 里挑几条当风格样本 (固定挑前几条, 保证可复现)
STYLE_SHOTS = [
    (
        "请解释机器学习中的偏差和方差，它们分别过高时通常会出现什么现象？",
        "偏差表示模型的假设与真实规律之间的差距。偏差过高通常说明模型太简单，会出现欠拟合：训练集和验证集效果都比较差，而且两者差距不大。方差表示模型对训练数据波动的敏感程度。方差过高通常说明模型过度记住了训练数据，会出现过拟合：训练集效果很好，但验证集明显变差，换一批数据结果波动也比较大。本质上就是在模型复杂度和泛化能力之间做平衡。",
    ),
    (
        "L1正则化和L2正则化有什么区别？在什么场景下你会优先选择其中一种？",
        "L1正则化惩罚参数绝对值，容易把部分参数压到0，因此能产生稀疏模型并实现特征选择，适合高维稀疏特征或希望模型更易解释的场景；L2正则化惩罚参数平方，通常不会把参数变为0，但能让参数整体变小，适合特征之间存在共线或希望模型更稳定的场景。如果目标是降维或解释性，我优先L1；如果只是防过拟合，L2更稳妥。",
    ),
]

SYSTEM_PROMPT = f"""你是一个在面试现场回答问题的求职者。用下面的风格回答问题。

要求：
1. 直接回答，不绕弯、不铺垫。
2. 先给核心概念的定义或直觉；再说"过高/异常时"会出现什么现象（用冒号具体展开）；最后用一句话点本质。
3. 专业术语用得准，但用大白话解释清楚。
4. 语气平实、口语化，像人现场口头作答；不要"首先/其次/最后"模板词，不要堆砌分点，不要 AI 腔。
5. 长度约 150-300 字。

风格参考：
问题：{STYLE_SHOTS[0][0]}
回答：{STYLE_SHOTS[0][1]}

问题：{STYLE_SHOTS[1][0]}
回答：{STYLE_SHOTS[1][1]}
"""


def load_rows(in_path: str):
    return list(csv.DictReader(open(in_path, encoding="utf-8-sig")))


def make_client(provider: str, key: str):
    base = PROVIDERS[provider]["base"]
    model = PROVIDERS[provider]["model"]
    return base, model, {"Authorization": f"Bearer {key}"}


def generate(base: str, model: str, headers: dict, row: dict):
    user = f"问题：{row['question']}\n"
    if row.get("interview_intent"):
        user += f"考察意图：{row['interview_intent']}\n"
    if row.get("expected_points"):
        user += f"期望要点：{row['expected_points']}\n"
    user += "回答："
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.45,
        "top_p": 0.9,
        "max_tokens": 500,
    }
    for attempt in range(3):
        try:
            r = requests.post(base + "/v1/chat/completions", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"生成失败: {row['question'][:30]}… -> {e}")
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default="data/rlhf_answers_filled.csv")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    ap.add_argument("--limit", type=int, default=0, help="只生成前 N 条(0=全部)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    key = os.environ.get("LLM_API_KEY", "").strip()
    if args.provider != "ollama" and not key:
        print("[error] 请先设置环境变量 LLM_API_KEY (DeepSeek: https://platform.deepseek.com; ollama 本地无需 key)")
        sys.exit(1)

    base, model, headers = make_client(args.provider, key)
    rows = load_rows(args.input)
    fields = list(rows[0].keys())
    kid = lambda r: r.get("id") or r["question"]

    # 源文件里已有答案的 -> 保留原样
    preserved = {kid(r): r for r in rows if len(str(r.get("answer", "")).strip()) >= MIN_ANSWER_LEN}
    # 断点续爬: 之前输出里已生成的 -> 也算完成(但不覆盖源文件已有的金标答案)
    if os.path.exists(args.out):
        for r in load_rows(args.out):
            if len(str(r.get("answer", "")).strip()) >= MIN_ANSWER_LEN:
                preserved.setdefault(kid(r), r)

    todo = [r for r in rows if kid(r) not in preserved]
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print("没有需要生成的答案(可能已全部完成)。")
        return
    print(f"待生成: {len(todo)} | 已有/已生成: {len(preserved)} | 模型: {model}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ok = 0
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        # 先写所有已完成的(保持输入顺序)
        for r in rows:
            if kid(r) in preserved:
                writer.writerow(preserved[kid(r)])
        done_ids = set(preserved)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(generate, base, model, headers, r): r for r in todo}
            for fut in as_completed(futs):
                r = futs[fut]
                if kid(r) in done_ids:
                    continue
                try:
                    r["answer"] = fut.result()
                    writer.writerow(r)
                    f.flush()
                    done_ids.add(kid(r))
                    ok += 1
                    print(f"[{ok}/{len(todo)}] {r['question'][:45]}")
                except Exception as e:
                    print(f"[x] {r['question'][:35]} -> {e}")

    print(f"\n完成。本次填了 {ok} 条答案 -> {args.out} (共 {ok + len(preserved)} 行)")


if __name__ == "__main__":
    main()
