# SFT v1 Independent Evaluation

## Conclusion

This evaluation confirms that the QLoRA SFT adapter produces a clear improvement over the raw Base model for the project's primary goal: concise, interview-style Chinese answers. The SFT model should be retained as the current SFT v1 baseline and used for subsequent data and decoding iteration.

The gain is strongest in answer usability. Compared with the Base model, SFT answers are substantially shorter, reach a natural ending more often, and remove most of the Base model's repeated lists and irrelevant continuations. The remaining repetition cases are a long-tail quality issue for the next iteration, not evidence that the overall SFT direction is wrong.

## Evaluation Setup

- 64 frozen independent questions, covering 8 existing roles with 8 questions per role.
- Base: `Qwen/Qwen2.5-3B`.
- SFT adapter: `Shawnno/qwen2.5-3b-interview-sft-lora`.
- Decoding: greedy (`do_sample=false`), `max_new_tokens=384`.
- The test questions and their scoring points were not used in SFT training.
- This run did not load a reward model. RM scores and win rates will be added in a separate pass once the RM path and input format are confirmed.

## Quantitative Results

| Metric | Base | SFT | SFT Change |
|---|---:|---:|---:|
| Mean answer length | 584 chars | 272 chars | -53.4% |
| Median answer length | 558 chars | 227 chars | -59.3% |
| Answers in 150-300 chars | 3/64 (4.7%) | 33/64 (51.6%) | +46.9 pp |
| Answers in 150-450 chars | 17/64 (26.6%) | 48/64 (75.0%) | +48.4 pp |
| Answers no longer than 550 chars | 31/64 (48.4%) | 61/64 (95.3%) | +46.9 pp |
| Natural sentence ending | 34/64 (53.1%) | 57/64 (89.1%) | +35.9 pp |
| Strong repeated 5-gram pattern | 51/64 (79.7%) | 15/64 (23.4%) | -56.3 pp |
| Extreme repeated 5-gram pattern | 27/64 (42.2%) | 7/64 (10.9%) | -31.3 pp |

Strong repetition means that a continuous 5-character pattern appears at least 4 times in one answer. Extreme repetition uses a threshold of at least 8 times. These are conservative mechanical diagnostics, not quality scores.

## Observations

- The Base model frequently continues into long enumerations, repeated separators, emojis, or irrelevant text. SFT suppresses this behavior on most questions.
- SFT answers are generally more direct and more suitable for an interview setting. Data-analysis metric diagnosis, data-lineage troubleshooting, and test mock-versus-end-to-end tradeoffs are representative improvements.
- The SFT model still has 15 strong repetition cases, including 7 extreme cases. They concentrate in several multi-constraint scenario questions and should be used as targeted regression cases for the next data cleanup or decoding pass.
- Some technical answers remain incomplete or imprecise. This evaluation therefore demonstrates a strong formatting and usability improvement relative to Base; it is not yet a substitute for a factual-correctness benchmark.

## Artifacts

- `comparisons.csv`: all 64 questions, prompts, Base answers, SFT answers, and hidden scoring points for offline review.
- `summary.json` and `summary.md`: model, decoding, and length metadata from the Colab run.
- `../../sft_test_v1.json`: frozen evaluation questions.
- `../../audit_overlap.py`: lexical overlap audit against the 3230-row SFT training corpus.

## Next Step

Run the same 64 pairs through the trained RM with the exact RM training input format. Report paired `mean_delta`, SFT win rate, and bootstrap confidence intervals alongside this report. Keep the seven extreme repetition cases as non-negotiable regression tests before considering any PPO or preference-optimization stage.
