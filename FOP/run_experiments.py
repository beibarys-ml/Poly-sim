"""
run_experiments.py — Baseline comparison and synthetic data analysis (Tasks 3 & 4)

Runs three FOP training experiments and prints a comparison table:
  A) Original German data only
  B) Synthetic German data only
  C) Mixed (original + synthetic)

Bonus (Task 4):
  synthetic_data_sweep() — varies synthetic sample count per identity to find
  the saturation point where additional synthetic data yields < 0.5% accuracy gain.

Usage:
  # Run all three comparison experiments:
  python run_experiments.py

  # Run bonus sweep analysis:
  python run_experiments.py --sweep

  # Specify custom CSV paths:
  python run_experiments.py --orig_csv feature_tracker/v3_train_German.csv
"""

import argparse
import copy
import csv
import logging
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure the FOP directory is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from model import FOP
from utils.featLoader import LoadData
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.earlystop import EarlyStopping

logger = logging.getLogger("run_experiments")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(csv_path: str, config: ExperimentConfig, shuffle: bool = False):
    dataset = LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def train_and_evaluate(
    train_csv: str,
    test_csv: str,
    unseen_csv: str,
    config: ExperimentConfig,
    experiment_name: str = "experiment",
) -> dict[float, dict]:
    """
    Train FOP for all alphas in config.alpha_list and return accuracy results.

    Returns:
        dict keyed by alpha → {'seen': float, 'unseen': float, 'best_epoch': int}
    """
    torch.manual_seed(config.seed)
    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    _, train_loader   = make_loader(train_csv,   config, shuffle=True)
    test_dataset,   _ = make_loader(test_csv,    config, shuffle=False)
    unseen_dataset, _ = make_loader(unseen_csv,  config, shuffle=False)

    audio_sample, face_sample, _ = next(iter(train_loader))

    results = {}

    for alpha in config.alpha_list:
        logger.info("[%s] Training with alpha=%.3f", experiment_name, alpha)

        model = FOP(
            config=config,
            face_dim=face_sample.shape[1],
            voice_dim=audio_sample.shape[1],
        )

        trainer  = Trainer(model, config)
        evaluator = Evaluator(model, config)
        stopper  = EarlyStopping(
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta,
        )

        best_seen, best_unseen, best_epoch = 0.0, 0.0, 0

        save_dir = Path("checkpoints") / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"alpha{alpha}_best.pt"

        for epoch in range(config.max_epochs):
            loss = trainer.train_epoch(train_loader, alpha, epoch=epoch)

            acc_seen   = evaluator.accuracy(test_dataset)
            acc_unseen = evaluator.accuracy(unseen_dataset)

            monitor = acc_seen if config.early_stop_metric == "seen" else acc_unseen

            if monitor > (best_seen if config.early_stop_metric == "seen" else best_unseen):
                best_seen, best_unseen, best_epoch = acc_seen, acc_unseen, epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "seen": acc_seen,
                    "unseen": acc_unseen,
                }, str(ckpt_path))

            logger.info(
                "[%s][α=%.3f] Epoch %03d | Loss %.4f | Seen %.2f | Unseen %.2f",
                experiment_name, alpha, epoch, loss, acc_seen, acc_unseen,
            )

            if config.early_stop and stopper.step(monitor):
                logger.info("Early stop at epoch %d", epoch)
                break

        results[alpha] = {
            "seen":       best_seen,
            "unseen":     best_unseen,
            "best_epoch": best_epoch,
        }
        logger.info(
            "[%s][α=%.3f] Best → Seen %.2f | Unseen %.2f (epoch %d)",
            experiment_name, alpha, best_seen, best_unseen, best_epoch,
        )

    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_comparison(
    orig_train_csv:  str = "feature_tracker/v3_train_German.csv",
    synth_train_csv: str = "feature_tracker/v3_train_German_synthetic.csv",
    mixed_train_csv: str = "feature_tracker/v3_train_German_mixed.csv",
    test_csv:        str = "feature_tracker/v3_test_German.csv",
    unseen_csv:      str = "feature_tracker/v3_test_English.csv",
) -> None:
    """
    Run experiments A, B, C and print a side-by-side comparison table.
    Skips an experiment if its CSV file does not exist.
    """
    config = ExperimentConfig()

    experiments = {
        "A_original":  orig_train_csv,
        "B_synthetic": synth_train_csv,
        "C_mixed":     mixed_train_csv,
    }

    all_results = {}

    for name, train_csv in experiments.items():
        if not os.path.exists(train_csv):
            logger.warning("CSV not found, skipping experiment %s: %s", name, train_csv)
            continue

        exp_config = copy.deepcopy(config)
        all_results[name] = train_and_evaluate(
            train_csv=train_csv,
            test_csv=test_csv,
            unseen_csv=unseen_csv,
            config=exp_config,
            experiment_name=name,
        )

    _print_comparison_table(all_results)


def _print_comparison_table(all_results: dict) -> None:
    """Pretty-print a comparison table across experiments and alphas."""
    if not all_results:
        print("No results to display.")
        return

    print("\n" + "=" * 72)
    print(f"{'Experiment':<20} {'Alpha':>8} {'Seen Acc':>12} {'Unseen Acc':>12} {'Best Epoch':>12}")
    print("-" * 72)

    for exp_name, alpha_results in sorted(all_results.items()):
        for alpha, metrics in sorted(alpha_results.items()):
            print(
                f"{exp_name:<20} {alpha:>8.3f} "
                f"{metrics['seen']:>11.2f}% "
                f"{metrics['unseen']:>11.2f}% "
                f"{metrics['best_epoch']:>12d}"
            )
        print("-" * 72)

    print("=" * 72 + "\n")

    # Save to CSV for easy reporting
    out_path = "results_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "alpha", "seen_acc", "unseen_acc", "best_epoch"])
        for exp_name, alpha_results in sorted(all_results.items()):
            for alpha, metrics in sorted(alpha_results.items()):
                writer.writerow([
                    exp_name, alpha,
                    f"{metrics['seen']:.4f}",
                    f"{metrics['unseen']:.4f}",
                    metrics["best_epoch"],
                ])
    logger.info("Results saved → %s", out_path)


# ---------------------------------------------------------------------------
# Task 4 (Bonus) — Synthetic data sweep analysis
# ---------------------------------------------------------------------------

def _sample_csv_by_identity(
    csv_path: str,
    fraction: float,
    seed: int = 42,
) -> list[list[str]]:
    """
    Sample `fraction` of rows per identity from a CSV file.
    Returns a list of rows (including header) to write as a sub-sampled CSV.
    """
    rng = random.Random(seed)

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_by_label: dict[str, list] = defaultdict(list)
        for row in reader:
            label = row[2]
            rows_by_label[label].append(row)

    sampled = [header]
    for label, rows in rows_by_label.items():
        n = max(1, math.ceil(len(rows) * fraction))
        sampled.extend(rng.sample(rows, n))

    return sampled


def synthetic_data_sweep(
    synth_train_csv: str = "feature_tracker/v3_train_German_synthetic.csv",
    orig_train_csv:  str = "feature_tracker/v3_train_German.csv",
    test_csv:        str = "feature_tracker/v3_test_German.csv",
    unseen_csv:      str = "feature_tracker/v3_test_English.csv",
    fractions:       list[float] = (0.10, 0.25, 0.50, 0.75, 1.00),
    threshold:       float = 0.5,
) -> None:
    """
    Task 4 (Bonus): Determine how much synthetic data per identity is needed.

    For each fraction in `fractions`:
      1. Sample that fraction of synthetic rows per identity
      2. Combine with original training data
      3. Train FOP and record seen/unseen accuracy

    Plots accuracy vs. fraction and prints the saturation point (the smallest
    fraction where adding more synthetic data yields < `threshold`% accuracy gain).

    Args:
        synth_train_csv : CSV with all synthetic German training rows
        orig_train_csv  : CSV with original German training rows
        test_csv        : CSV for seen-language evaluation
        unseen_csv      : CSV for unseen-language evaluation
        fractions       : fractions of synthetic data to evaluate
        threshold       : minimum acc gain (%) to consider non-saturated
    """
    if not os.path.exists(synth_train_csv):
        logger.error("Synthetic CSV not found: %s", synth_train_csv)
        return

    config = ExperimentConfig()
    config.early_stop_patience = 5   # faster sweeps
    config.max_epochs = 100

    sweep_results = {}   # fraction → {'seen': float, 'unseen': float}
    tmp_dir = Path("sweep_tmp")
    tmp_dir.mkdir(exist_ok=True)

    for frac in fractions:
        logger.info("=== Sweep: %.0f%% synthetic data ===", frac * 100)

        # Sub-sample synthetic rows
        synth_rows = _sample_csv_by_identity(synth_train_csv, frac, seed=config.seed)

        # Combine with original
        with open(orig_train_csv, newline="") as f:
            orig_rows = list(csv.reader(f))

        combined = orig_rows + synth_rows[1:]  # skip duplicate header from synth

        tmp_csv = tmp_dir / f"mixed_frac{int(frac*100):03d}.csv"
        with open(str(tmp_csv), "w", newline="") as f:
            csv.writer(f).writerows(combined)

        alpha_results = train_and_evaluate(
            train_csv=str(tmp_csv),
            test_csv=test_csv,
            unseen_csv=unseen_csv,
            config=copy.deepcopy(config),
            experiment_name=f"sweep_{int(frac*100)}pct",
        )

        # Use the first (or only) alpha result
        first_alpha = list(alpha_results.keys())[0]
        sweep_results[frac] = alpha_results[first_alpha]

    # --- Print sweep table ---
    print("\n" + "=" * 60)
    print(f"{'Synthetic %':>12} {'Seen Acc':>12} {'Unseen Acc':>12}")
    print("-" * 60)
    for frac in sorted(sweep_results):
        m = sweep_results[frac]
        print(f"{frac*100:>11.0f}% {m['seen']:>11.2f}% {m['unseen']:>11.2f}%")
    print("=" * 60)

    # --- Find saturation point ---
    sorted_fracs = sorted(sweep_results)
    saturation_frac = sorted_fracs[-1]  # default: all data needed

    for i in range(1, len(sorted_fracs)):
        prev = sweep_results[sorted_fracs[i - 1]]["seen"]
        curr = sweep_results[sorted_fracs[i]]["seen"]
        if curr - prev < threshold:
            saturation_frac = sorted_fracs[i - 1]
            break

    print(f"\nSaturation point: {saturation_frac*100:.0f}% of synthetic data per identity")
    print(f"(Adding more than {saturation_frac*100:.0f}% yields < {threshold}% accuracy gain)\n")

    # --- Save sweep results ---
    sweep_csv = "results_sweep.csv"
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["synth_fraction", "seen_acc", "unseen_acc", "best_epoch"])
        for frac in sorted(sweep_results):
            m = sweep_results[frac]
            writer.writerow([frac, f"{m['seen']:.4f}", f"{m['unseen']:.4f}", m["best_epoch"]])
    logger.info("Sweep results saved → %s", sweep_csv)

    # --- Optional matplotlib plot ---
    try:
        import matplotlib.pyplot as plt

        fracs = [f * 100 for f in sorted(sweep_results)]
        seen_accs   = [sweep_results[f / 100]["seen"]   for f in fracs]
        unseen_accs = [sweep_results[f / 100]["unseen"] for f in fracs]

        plt.figure(figsize=(8, 5))
        plt.plot(fracs, seen_accs,   marker="o", label="Seen (German)")
        plt.plot(fracs, unseen_accs, marker="s", label="Unseen (English)")
        plt.axvline(x=saturation_frac * 100, color="gray", linestyle="--",
                    label=f"Saturation @ {saturation_frac*100:.0f}%")
        plt.xlabel("Synthetic Data per Identity (%)")
        plt.ylabel("Accuracy (%)")
        plt.title("Synthetic Data Amount vs. Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results_sweep.png", dpi=150)
        logger.info("Plot saved → results_sweep.png")
        plt.show()

    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FOP baseline comparison experiments")
    parser.add_argument("--sweep", action="store_true",
                        help="Run Task 4 bonus: synthetic data amount sweep analysis")
    parser.add_argument("--orig_csv",  default="feature_tracker/v3_train_German.csv")
    parser.add_argument("--synth_csv", default="feature_tracker/v3_train_German_synthetic.csv")
    parser.add_argument("--mixed_csv", default="feature_tracker/v3_train_German_mixed.csv")
    parser.add_argument("--test_csv",  default="feature_tracker/v3_test_German.csv")
    parser.add_argument("--unseen_csv",default="feature_tracker/v3_test_English.csv")
    args = parser.parse_args()

    if args.sweep:
        synthetic_data_sweep(
            synth_train_csv=args.synth_csv,
            orig_train_csv=args.orig_csv,
            test_csv=args.test_csv,
            unseen_csv=args.unseen_csv,
        )
    else:
        run_comparison(
            orig_train_csv=args.orig_csv,
            synth_train_csv=args.synth_csv,
            mixed_train_csv=args.mixed_csv,
            test_csv=args.test_csv,
            unseen_csv=args.unseen_csv,
        )
