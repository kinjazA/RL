# 把 RL 项目的 SFT 数据接入 LLaMA-Factory:
#   1. 拷贝 sft_train.json 到 LLaMA-Factory/data/
#   2. 把 "sft_train" 条目合并进 LLaMA-Factory/data/dataset_info.json (幂等, 可重复跑)
import json
import os
import shutil

RL_DATA = os.path.join(os.path.dirname(__file__), "data")
LF_ROOT = os.environ.get("LLAMAFACTORY_ROOT", r"C:\Users\leeze\Documents\GitHub\LLaMA-Factory")

SRC_JSON = os.path.join(RL_DATA, "sft_train.json")
DST_JSON = os.path.join(LF_ROOT, "data", "sft_train.json")
SRC_INFO = os.path.join(RL_DATA, "dataset_info.json")
DST_INFO = os.path.join(LF_ROOT, "data", "dataset_info.json")

ENTRY = {
    "sft_train": {
        "file_name": "sft_train.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
}

# 1. 拷贝数据文件
if not os.path.exists(SRC_JSON):
    raise SystemExit(f"找不到源数据: {SRC_JSON}")
shutil.copy2(SRC_JSON, DST_JSON)
print(f"[1/2] 数据已拷贝: {DST_JSON}")

# 2. 合并注册表 (幂等: 已有 sft_train 就跳过)
with open(DST_INFO, encoding="utf-8") as f:
    info = json.load(f)
if "sft_train" in info:
    print("[2/2] sft_train 已存在, 无需合并")
else:
    info.update(ENTRY)
    with open(DST_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[2/2] 已合并 sft_train ->", DST_INFO)

print("完成! 现在可以运行: llamafactory-cli webui")
