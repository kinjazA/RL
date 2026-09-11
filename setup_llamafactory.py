# 把 RL 项目的 SFT / DPO 数据接入 LLaMA-Factory:
#   1. 拷贝 sft_train.json (SFT) + preference_pairs_llamafactory.json (DPO) 到 LLaMA-Factory/data/
#   2. 把 "sft_train" 与 "interview_preference_pairs" 并入 LLaMA-Factory/data/dataset_info.json (幂等, 可重复跑)
import json
import os
import shutil

RL_ROOT = os.path.dirname(__file__)
RL_DATA = os.path.join(RL_ROOT, "data")
RM_DATA = os.path.join(RL_ROOT, "rm", "artifacts", "full_v1")
LF_ROOT = os.environ.get("LLAMAFACTORY_ROOT", r"C:\Users\leeze\Documents\GitHub\LLaMA-Factory")

# (源文件, 目标文件名) —— 目标统一放 LF_ROOT/data/
DATASETS = [
    (os.path.join(RL_DATA, "sft_train.json"), "sft_train.json"),
    (os.path.join(RM_DATA, "preference_pairs_llamafactory.json"), "preference_pairs_llamafactory.json"),
]
DST_INFO = os.path.join(LF_ROOT, "data", "dataset_info.json")

# 需要注册进 dataset_info.json 的条目 (幂等)
ENTRIES = {
    "sft_train": {
        "file_name": "sft_train.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    },
    "interview_preference_pairs": {
        "file_name": "preference_pairs_llamafactory.json",
        "ranking": True,
        "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected"},
    },
}

# 1. 拷贝数据文件
for src, name in DATASETS:
    if not os.path.exists(src):
        raise SystemExit(f"找不到源数据: {src}")
    shutil.copy2(src, os.path.join(LF_ROOT, "data", name))
    print(f"[1/x] 数据已拷贝: {name}")

# 2. 合并注册表 (幂等: 已有该 key 就跳过)
with open(DST_INFO, encoding="utf-8") as f:
    info = json.load(f)
for key, entry in ENTRIES.items():
    if key in info:
        print(f"[2/x] {key} 已存在, 无需合并")
    else:
        info[key] = entry
        with open(DST_INFO, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"[2/x] 已合并 {key} ->", DST_INFO)

print("完成! 现在可以运行: llamafactory-cli train sft_qwen3b.yaml 或 dpo_qwen3b.yaml")
