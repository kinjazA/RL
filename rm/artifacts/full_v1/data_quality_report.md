# Preference Data Quality Report

- Selected prompts: 3230
- Scored candidates: 17251
- Retained preference pairs: 1210 (37.5%)
- Teacher judge: `Skywork/Skywork-Reward-V2-Qwen3-4B`
- Margin band: [0.5, 8.0]
- Chosen sources: {'sft_reference': 1210}
- Skipped: {'fewer_than_two_viable_answers': 334, 'no_hard_negative_in_margin_band': 1686}
- Reward-margin distribution: {'count': 1210, 'mean': 5.1134, 'median': 5.25, 'p10': 2.5, 'p90': 7.375, 'min': 0.5, 'max': 8.0}

`manual_audit.csv` contains every selected pair for review. Do not train a reward model before auditing a stratified sample from that file.
