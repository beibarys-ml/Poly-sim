"""
run_experiments.py — Baseline comparison + synthetic data sweep (Tasks 3 & 4)

Experiments:
  A) Original German data only
  B) Synthetic German data only  (skipped if CSV missing)
  C) Mixed (original + synthetic) (skipped if CSV missing)

Task 4 (Bonus): --sweep  sweeps synthetic data fraction per identity.

All results saved under ./results/ (or --results_dir):
  results/
    {exp_name}/results.csv    ← per-epoch metrics
    {exp_name}/summary.csv    ← best metrics
    comparison.csv            ← side-by-side table
    sweep.csv                 ← sweep results (--sweep only)

Usage:
  python run_experiments.py
  python run_experiments.py --sweep
  python run_experiments.py --results_dir ./my_results
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from main import main as _train

logger = logging.getLogger("run_experiments")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s][%(name)s] %(message)s")


# ---------------------------------------------------------------------------
# Comparison runner (Task 3)
# ---------------------------------------------------------------------------

def run_comparison(
    orig_train_csv:  str = "feature_tracker/v3_train_German.csv",
    synth_train_csv: str = "feature_tracker/v3_train_German_synthetic.csv",
    mixed_train_csv: str = "feature_tracker/v3_train_German_mixed.csv",
    val_csv:         str = "feature_tracker/v3_val_German.csv",
    test_csv:        str = "feature_tracker/v3_test_German.csv",
    unseen_csv:      str = "feature_tracker/v3_test_English.csv",
    results_dir:     str = "./results",
) -> None:
    """
    Run A / B / C experiments and write comparison.csv.
    Experiments whose train CSV is missing are skipped automatically.
    """
    base_cfg = ExperimentConfig()

    experiments = {
        "A_original":  orig_train_csv,
        "B_synthetic": synth_train_csv,
        "C_mixed":     mixed_train_csv,
    }

    all_results: dict[str, dict] = {}

    for name, train_csv in experiments.items():
        if not os.path.exists(train_csv):
            logger.warning("Skipping %s — CSV not found: %s", name, train_csv)
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg.results_dir = os.path.join(results_dir, name)

        logger.info("=== Experiment: %s ===", name)
        all_results[name] = _train(
            train_csv  = train_csv,
            val_csv    = val_csv,
            test_csv   = test_csv,
            unseen_csv = unseen_csv,
            run_id     = name,
            config     = cfg,
        )

    _save_comparison_table(all_results, results_dir)


def _save_comparison_table(all_results: dict, results_dir: str) -> None:
    if not all_results:
        print("No results to display.")
        return

    print("\n" + "=" * 82)
    print(f"{'Experiment':<20} {'Alpha':>8} {'Seen':>10} {'Val':>10} {'Unseen':>10} {'Epoch':>8}")
    print("-" * 82)
    for exp_name, alpha_results in sorted(all_results.items()):
        for alpha, m in sorted(alpha_results.items()):
            print(
                f"{exp_name:<20} {alpha:>8.4f} "
                f"{m['seen']:>9.2f}% "
                f"{m.get('val', 0.0):>9.2f}% "
                f"{m['unseen']:>9.2f}% "
                f"{m['epoch']:>8d}"
            )
        print("-" * 82)
    print("=" * 82 + "\n")

    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "comparison.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "alpha", "seen_acc", "val_acc", "unseen_acc", "best_epoch"])
        for exp_name, alpha_results in sorted(all_results.items()):
            for alpha, m in sorted(alpha_results.items()):
                w.writerow([
                    exp_name, alpha,
                    round(m["seen"], 4),
                    round(m.get("val", 0.0), 4),
                    round(m["unseen"], 4),
                    m["epoch"],
                ])
    logger.info("Comparison table -> %s", out)


# ---------------------------------------------------------------------------
# Synthetic data sweep (Task 4 Bonus)
# ---------------------------------------------------------------------------

def _sample_csv_rows(csv_path: str, fraction: float, seed: int) -> list[list]:
    """Return header + stratified sample of `fraction` rows per label."""
    rng = random.Random(seed)
    by_label: dict[str, list] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            by_label[row[2]].append(row)

    sampled = [header]
    for rows in by_label.values():
        n = max(1, math.ceil(len(rows) * fraction))
        sampled.extend(rng.sample(rows, min(n, len(rows))))
    return sampled


def synthetic_data_sweep(
    synth_train_csv: str = "feature_tracker/v3_train_German_synthetic.csv",
    orig_train_csv:  str = "feature_tracker/v3_train_German.csv",
    val_csv:         str = "feature_tracker/v3_val_German.csv",
    test_csv:        str = "feature_tracker/v3_test_German.csv",
    unseen_csv:      str = "feature_tracker/v3_test_English.csv",
    fractions:       tuple = (0.10, 0.25, 0.50, 0.75, 1.00),
    saturation_threshold: float = 0.5,
    results_dir:     str = "./results",
) -> None:
    """
    For each fraction of synthetic data per identity, combine with all original
    training data, train FOP, and record accuracy.

    Finds the saturation point where adding more synthetic data yields
    < saturation_threshold% accuracy gain.
    """
    if not os.path.exists(synth_train_csv):
        logger.error("Synthetic CSV not found: %s", synth_train_csv)
        return

    base_cfg = ExperimentConfig()
    base_cfg.early_stop_patience = 5
    base_cfg.max_epochs = 100

    tmp_dir = os.path.join(results_dir, "sweep_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    sweep_results: dict[float, dict] = {}

    # Read original train rows once
    with open(orig_train_csv, newline="", encoding="utf-8") as f:
        orig_rows = list(csv.reader(f))

    for frac in fractions:
        logger.info("=== Sweep %.0f%% synthetic ===", frac * 100)

        synth_rows = _sample_csv_rows(synth_train_csv, frac, seed=base_cfg.seed)
        combined = orig_rows + synth_rows[1:]  # skip duplicate header

        tmp_csv = os.path.join(tmp_dir, f"mixed_frac{int(frac*100):03d}.csv")
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(combined)

        cfg = copy.deepcopy(base_cfg)
        cfg.results_dir = os.path.join(results_dir, f"sweep_{int(frac*100):03d}pct")

        alpha_results = _train(
            train_csv  = tmp_csv,
            val_csv    = val_csv,
            test_csv   = test_csv,
            unseen_csv = unseen_csv,
            run_id     = f"sweep_{int(frac*100)}pct",
            config     = cfg,
        )
        sweep_results[frac] = alpha_results[list(alpha_results.keys())[0]]

    # ---- print & save sweep table ----
    print("\n" + "=" * 62)
    print(f"{'Synth %':>10} {'Seen':>10} {'Val':>10} {'Unseen':>10}")
    print("-" * 62)
    for frac in sorted(sweep_results):
        m = sweep_results[frac]
        print(f"{frac*100:>9.0f}% {m['seen']:>9.2f}% {m.get('val',0):>9.2f}% {m['unseen']:>9.2f}%")
    print("=" * 62)

    # Saturation point
    sorted_fracs = sorted(sweep_results)
    saturation = sorted_fracs[-1]
    for i in range(1, len(sorted_fracs)):
        gain = sweep_results[sorted_fracs[i]]["seen"] - sweep_results[sorted_fracs[i-1]]["seen"]
        if gain < saturation_threshold:
            saturation = sorted_fracs[i - 1]
            break
    print(f"\nSaturation at {saturation*100:.0f}% synthetic data per identity "
          f"(gain < {saturation_threshold}% beyond this)\n")

    sweep_csv = os.path.join(results_dir, "sweep.csv")
    os.makedirs(results_dir, exist_ok=True)
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["synth_fraction", "seen_acc", "val_acc", "unseen_acc", "best_epoch"])
        for frac in sorted(sweep_results):
            m = sweep_results[frac]
            w.writerow([frac, round(m["seen"], 4), round(m.get("val", 0), 4),
                        round(m["unseen"], 4), m["epoch"]])
    logger.info("Sweep results -> %s", sweep_csv)

    try:
        import matplotlib.pyplot as plt
        fracs_pct = [f * 100 for f in sorted(sweep_results)]
        plt.figure(figsize=(8, 5))
        plt.plot(fracs_pct, [sweep_results[f/100]["seen"]   for f in fracs_pct], "o-", label="Seen (German)")
        plt.plot(fracs_pct, [sweep_results[f/100]["unseen"] for f in fracs_pct], "s-", label="Unseen (English)")
        plt.axvline(saturation * 100, color="gray", ls="--",
                    label=f"Saturation @ {saturation*100:.0f}%")
        plt.xlabel("Synthetic Data per Identity (%)")
        plt.ylabel("Accuracy (%)")
        plt.title("Effect of Synthetic Data Amount on Accuracy")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "sweep.png")
        plt.savefig(plot_path, dpi=150)
        logger.info("Sweep plot -> %s", plot_path)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOP baseline experiments")
    parser.add_argument("--sweep",       action="store_true",
                        help="Run Task 4 bonus: synthetic data sweep")
    parser.add_argument("--orig_csv",    default="feature_tracker/v3_train_German.csv")
    parser.add_argument("--synth_csv",   default="feature_tracker/v3_train_German_synthetic.csv")
    parser.add_argument("--mixed_csv",   default="feature_tracker/v3_train_German_mixed.csv")
    parser.add_argument("--val_csv",     default="feature_tracker/v3_val_German.csv")
    parser.add_argument("--test_csv",    default="feature_tracker/v3_test_German.csv")
    parser.add_argument("--unseen_csv",  default="feature_tracker/v3_test_English.csv")
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    if args.sweep:
        synthetic_data_sweep(
            synth_train_csv = args.synth_csv,
            orig_train_csv  = args.orig_csv,
            val_csv         = args.val_csv,
            test_csv        = args.test_csv,
            unseen_csv      = args.unseen_csv,
            results_dir     = args.results_dir,
        )
    else:
        run_comparison(
            orig_train_csv  = args.orig_csv,
            synth_train_csv = args.synth_csv,
            mixed_train_csv = args.mixed_csv,
            val_csv         = args.val_csv,
            test_csv        = args.test_csv,
            unseen_csv      = args.unseen_csv,
            results_dir     = args.results_dir,
        )
