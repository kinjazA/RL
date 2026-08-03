# Base vs SFT Acceptance Summary

- Questions: 64
- Base: `Qwen/Qwen2.5-3B`
- SFT adapter: `Shawnno/qwen2.5-3b-interview-sft-lora`
- Decoding: greedy (`do_sample=false`), max_new_tokens=384

## Length

- Base: {'mean_chars': 584.25, 'median_chars': 557.5, 'p90_chars': 746.8, 'max_chars': 1921}
- SFT: {'mean_chars': 272.44, 'median_chars': 227.0, 'p90_chars': 432.5, 'max_chars': 754}
