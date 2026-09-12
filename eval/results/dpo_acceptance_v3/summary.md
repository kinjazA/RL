# Acceptance Summary

- Questions: 64
- Base: `Qwen/Qwen2.5-3B-Instruct`
- SFT adapter: `C:/Users/leeze/Documents/GitHub/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/sft-cli`
- DPO adapter: `C:/Users/leeze/Documents/GitHub/LLaMA-Factory/saves/Qwen2.5-3B-Instruct/lora/dpo-cli-v2`
- Decoding: greedy (`do_sample=false`), max_new_tokens=600

## Length

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| mean_chars | 908.3 | 359.1 | 454.9 |
| median_chars | 883.0 | 345.0 | 443.0 |
| p90_chars | 1069.1 | 478.5 | 570.5 |
| max_chars | 1829 | 578 | 716 |

## Length buckets & style diagnostics

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| pct_150_300 (%) | 0.0 | 18.8 | 3.1 |
| pct_150_450 (%) | 1.6 | 84.4 | 53.1 |
| pct_le_550 (%) | 1.6 | 96.9 | 84.4 |
| natural_ending (%) | 100.0 | 100.0 | 100.0 |
| strong 5-gram repeat (%) | 90.6 | 9.4 | 10.9 |
| extreme 5-gram repeat (%) | 71.9 | 0.0 | 1.6 |
