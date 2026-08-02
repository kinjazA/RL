"""
把 data/rlhf_answers_filled.csv 转成 LLaMA-Factory 可用的 SFT 数据集.

- 格式: alpaca JSON (instruction=问题, output=答案)
- 输出:
    data/sft_train.json      全部数据 (alpaca)
    data/dataset_info.json   LLaMA-Factory 数据集注册文件

用法:
    python convert_llamafactory.py
    python convert_llamafactory.py --input data/rlhf_answers_filled.csv
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/rlhf_answers_filled.csv")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--with_system", action="store_true",
                    help="给每条加 system 风格提示(默认不加,模型从输出里学风格)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, encoding="utf-8-sig")))
    print(f"读入 {len(rows)} 行")

    def to_alpaca(r):
        d = {
            "instruction": r["question"].strip(),
            "input": "",
            "output": r["answer"].strip(),
        }
        if args.with_system:
            d["system"] = (
                "你是面试现场的求职者。回答要简洁、专业、口语化：先给概念定义，"
                "再说现象（冒号展开），最后点本质。不要 AI 腔、不要模板词。"
            )
        return d

    data = [to_alpaca(r) for r in rows]
    os.makedirs(args.out_dir, exist_ok=True)

    p = os.path.join(args.out_dir, "sft_train.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"  {p}: {len(data)} 条")

    info = {
        "sft_train": {
            "file_name": "sft_train.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  data/dataset_info.json: 注册文件")


if __name__ == "__main__":
    main()
