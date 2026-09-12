# Preference Data Quality Report

- Selected prompts: 3230
- Scored candidates: 17251
- Retained preference pairs: 805 (24.9%)
- Teacher judge: `Skywork/Skywork-Reward-V2-Qwen3-4B`
- Margin band: [0.5, 8.0]
- Length ratio limit: 1.3x
- Chosen sources: {'sft_reference': 805}
- Skipped: {'fewer_than_two_viable_answers': 334, 'no_hard_negative_in_margin_band': 1686, 'no_length_matched_negative': 405}
- Reward-margin distribution: {'count': 805, 'mean': 5.1051, 'median': 5.25, 'p10': 2.5, 'p90': 7.375, 'min': 0.5, 'max': 8.0}
- Length-ratio distribution: {'count': 805, 'mean': 1.1347, 'median': 1.1277, 'p10': 1.0255, 'p90': 1.263, 'min': 1.0, 'max': 1.2996}

`manual_audit.csv` contains every selected pair for review. Do not train a reward model before auditing a stratified sample from that file.
