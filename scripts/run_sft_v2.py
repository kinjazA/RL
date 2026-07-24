"""One-command SFT v2 training launcher.

Run from the RL/ directory:

    python scripts/run_sft_v2.py

The script prepares the curated SFT data, builds group-isolated splits, verifies
assistant-token loss masking, and launches QLoRA SFT training.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(cmd: list[str], *, skip: bool = False):
    if skip:
        print(f"[skip] {' '.join(cmd)}")
        return
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="sft_output", help="Where to save the trained LoRA adapter.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--max_per_question", type=int, default=3)
    parser.add_argument("--eval_frac", type=float, default=0.08)
    parser.add_argument("--test_frac", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_data_prep", action="store_true", help="Reuse existing data/sft_*_v2.csv files.")
    parser.add_argument("--skip_collator_check", action="store_true", help="Skip TRL collator sanity check.")
    args = parser.parse_args()

    py = sys.executable
    train_csv = os.path.join("data", "sft_train_v2.csv")
    eval_csv = os.path.join("data", "sft_eval_v2.csv")

    print("SFT v2 launcher")
    print(f"Root: {ROOT}")
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Output: {args.output_dir}")

    run(
        [
            py,
            "scripts/sft_prepare_data.py",
            "--max_per_question",
            str(args.max_per_question),
        ],
        skip=args.skip_data_prep,
    )
    run([py, "scripts/sft_data_report.py"], skip=args.skip_data_prep)
    run(
        [
            py,
            "scripts/build_sft_splits.py",
            "--eval_frac",
            str(args.eval_frac),
            "--test_frac",
            str(args.test_frac),
            "--seed",
            str(args.seed),
        ],
        skip=args.skip_data_prep,
    )
    run(
        [py, "-m", "sft.checks.verify_collator", "--csv_path", train_csv],
        skip=args.skip_collator_check,
    )
    run(
        [
            py,
            "-m",
            "sft.train",
            "--csv_path",
            train_csv,
            "--eval_csv_path",
            eval_csv,
            "--output_dir",
            args.output_dir,
            "--learning_rate",
            str(args.learning_rate),
            "--num_train_epochs",
            str(args.num_train_epochs),
        ]
    )


if __name__ == "__main__":
    main()
