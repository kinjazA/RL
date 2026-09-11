# Acceptance Summary

- Questions: 64
- Base: `Qwen/Qwen2.5-3B-Instruct`
- SFT adapter: `Shawnno/qwen2.5-3b-interview-sft-lora`
- DPO adapter: `C:/Users/leeze/Documents/GitHub/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/dpo-cli`
- Decoding: greedy (`do_sample=false`), max_new_tokens=384

## Length

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| mean_chars | 688.2 | 359.1 | 467.6 |
| median_chars | 684.5 | 345.0 | 477.0 |
| p90_chars | 728.4 | 478.5 | 564.0 |
| max_chars | 917 | 578 | 681 |

## Length buckets & style diagnostics

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| pct_150_300 (%) | 0.0 | 18.8 | 0.0 |
| pct_150_450 (%) | 1.6 | 84.4 | 42.2 |
| pct_le_550 (%) | 1.6 | 96.9 | 78.1 |
| natural_ending (%) | 9.4 | 100.0 | 82.8 |
| strong 5-gram repeat (%) | 90.6 | 9.4 | 25.0 |
| extreme 5-gram repeat (%) | 56.2 | 0.0 | 0.0 |
