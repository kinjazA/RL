# Acceptance Summary

- Questions: 64
- Base: `Qwen/Qwen2.5-3B-Instruct`
- SFT adapter: `C:/Users/leeze/Documents/GitHub/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/sft-cli`
- DPO adapter: `C:/Users/leeze/Documents/GitHub/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/dpo-cli-v2`
- Decoding: greedy (`do_sample=false`), max_new_tokens=384

## Length

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| mean_chars | 688.2 | 359.1 | 448.8 |
| median_chars | 684.5 | 345.0 | 443.0 |
| p90_chars | 728.4 | 478.5 | 547.8 |
| max_chars | 917 | 578 | 682 |

## Length buckets & style diagnostics

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| pct_150_300 (%) | 0.0 | 18.8 | 3.1 |
| pct_150_450 (%) | 1.6 | 84.4 | 53.1 |
| pct_le_550 (%) | 1.6 | 96.9 | 92.2 |
| natural_ending (%) | 9.4 | 100.0 | 85.9 |
| strong 5-gram repeat (%) | 90.6 | 9.4 | 10.9 |
| extreme 5-gram repeat (%) | 56.2 | 0.0 | 1.6 |
