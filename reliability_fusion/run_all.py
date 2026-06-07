import argparse
import subprocess
import sys
from pathlib import Path

from reliability_fusion.config import RESULTS_DIR


def run_command(command):
    print("=" * 100)
    print("Running command:")
    print(" ".join(command))
    print("=" * 100)

    completed = subprocess.run(command)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: {' '.join(command)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs for each language model.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum validation accuracy improvement for early stopping.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run evaluation using existing checkpoints.",
    )

    parser.add_argument(
        "--clear_results",
        action="store_true",
        help="Remove previous reliability_fusion_results.csv before evaluation.",
    )

    args = parser.parse_args()

    results_csv = RESULTS_DIR / "reliability_fusion_results.csv"

    if args.clear_results and results_csv.exists():
        results_csv.unlink()
        print(f"Removed old results file: {results_csv}")

    python_exe = sys.executable

    if not args.skip_training:
        for train_language in ["English", "German"]:
            run_command(
                [
                python_exe,
                "-m",
                "reliability_fusion.train",
                "--train_language",
                train_language,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--min_delta",
                str(args.min_delta),
            ]
            )

    experiments = [
        # English-trained model
        ("English", "English", False),  # P3
        ("English", "English", True),   # P4
        ("English", "German", False),   # P5
        ("English", "German", True),    # P6

        # German-trained model
        ("German", "German", False),    # P3
        ("German", "German", True),     # P4
        ("German", "English", False),   # P5
        ("German", "English", True),    # P6
    ]

    for train_language, test_language, audio_only in experiments:
        command = [
            python_exe,
            "-m",
            "reliability_fusion.evaluate",
            "--train_language",
            train_language,
            "--test_language",
            test_language,
            "--batch_size",
            str(args.batch_size),
            "--output_csv",
            str(results_csv),
        ]

        if audio_only:
            command.append("--audio_only")

        run_command(command)

    print("=" * 100)
    print("All experiments finished.")
    print(f"Results saved to: {results_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()