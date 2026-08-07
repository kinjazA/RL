# Reward-Model Preference Data

This directory builds synthetic, auditable preference pairs for a later reward-model run. It does **not** modify `data/sft_train.json`, and it never reads the frozen 64-question SFT acceptance set in `eval/sft_test_v1.json`.

The flow is:

```text
SFT reference answer + sampled SFT answers
    -> Skywork-Reward-V2 scores each answer within the same question
    -> quality filters and hard-negative selection
    -> RM/DPO-compatible chosen-rejected pairs
```

## Selection rules

- One answer is sampled from the SFT adapter for every configured temperature. The defaults are `0.3,0.5,0.7,0.9,1.1`.
- The original SFT response is preferred as `chosen`, provided it passes basic quality checks.
- The `rejected` answer is the closest lower-scoring answer inside the configured reward-margin band. This produces a useful hard negative, rather than a blank response or a repetition loop.
- Empty, very short, very long, and strongly repetitive outputs are excluded before pair selection.
- When the reference cannot form a valid pair, the highest-scoring sampled answer may be used as `chosen`; the output records that provenance. Pass `--no_sample_chosen_fallback` to disable this fallback.
- Scores and score gaps are only used inside each prompt. They are not treated as global quality scores across prompts.

## Colab Pilot

Use a GPU runtime. `colab_preference_data.ipynb` is ready to upload directly to Colab. The first run should be a 200-prompt pilot, not all 3230 prompts:

```bash
git clone https://github.com/kinjazA/RL.git
cd RL
pip install -q "transformers>=4.52.3" "peft>=0.14" accelerate bitsandbytes sentencepiece

python rm/build_preference_data.py \
  --output_dir rm/artifacts/pilot_v1 \
  --limit 200
```

The script is resumable. On an interruption, run the same command again. The individual stages can also be run separately:

```bash
python rm/build_preference_data.py --output_dir rm/artifacts/pilot_v1 --limit 200 --stage generate
python rm/build_preference_data.py --output_dir rm/artifacts/pilot_v1 --limit 200 --stage score
python rm/build_preference_data.py --output_dir rm/artifacts/pilot_v1 --limit 200 --stage build
```

## Required Review Before Training

The pilot produces these files under its output directory:

- `candidates.jsonl`: reference and sampled answers, with seeds and decoding parameters.
- `scored_candidates.jsonl`: candidate records plus the Skywork reward score.
- `preference_pairs.jsonl`: canonical prompt/chosen/rejected pairs plus selection metadata.
- `preference_pairs_llamafactory.json`: ranking data ready to register in LLaMA-Factory.
- `dataset_info.json`: the corresponding LLaMA-Factory dataset registration fragment.
- `manual_audit.csv`: every selected pair for manual review.
- `data_quality_report.json` / `.md`: pair retention, role balance, chosen provenance, rejected-pair reasons, and score-margin distribution.

Review a stratified sample of `manual_audit.csv` before scaling up. Check factual correctness, role fit, answer style, and whether the rejected answer is plausible but genuinely worse. Then adjust `--min_margin` / `--max_margin` only from the pilot report, use a new output directory, and run the full dataset with `--limit 0`.

The generated pairs can train a reward model or a direct-preference method later. Building these pairs alone is not a justification to run PPO.
