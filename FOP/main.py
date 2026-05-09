"""
main.py — FOP baseline training entry point

Results are saved to:
  {config.results_dir}/
    results.csv               ← per-epoch metrics for every alpha run
    summary.csv               ← best metrics per alpha run
    {version}_{lang}_alpha{a}_best.pt  ← model checkpoint

Usage:
  python main.py
  python main.py --device cpu          # force CPU (for testing)
  python main.py --seed 42             # override seed
  python main.py --train_csv feature_tracker/v3_train_German_mixed.csv
"""

import argparse
import csv
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from model import FOP
from utils.earlystop import EarlyStopping
from utils.evaluator import Evaluator
from utils.featLoader import LoadData
from utils.trainer import Trainer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Lock all sources of randomness for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(config: ExperimentConfig) -> logging.Logger:
    logger = logging.getLogger("FOP")
    logger.setLevel(config.log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s][%(name)s] %(message)s"))
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_loader(
    csv_path: str,
    config: ExperimentConfig,
    shuffle: bool = False,
) -> tuple:
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
        pin_memory=(config.device == "cuda"),
        drop_last=False,
    )
    return dataset, loader


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    metrics: dict,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch":          epoch,
        "metrics":        metrics,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":         vars(config),
    }, save_path)


# ---------------------------------------------------------------------------
# CSV result writers
# ---------------------------------------------------------------------------

def _ensure_csv(path: str, header: list[str]) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_epoch_row(path: str, row: list) -> None:
    _ensure_csv(path, [
        "run_id", "alpha", "epoch", "loss",
        "acc_seen", "acc_val", "acc_unseen",
    ])
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def append_summary_row(path: str, row: list) -> None:
    _ensure_csv(path, [
        "run_id", "alpha",
        "best_epoch", "best_acc_seen", "best_acc_val", "best_acc_unseen",
        "seed", "fusion", "embedding_dim", "lr", "batch_size",
    ])
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    train_csv: str = None,
    val_csv:   str = None,
    test_csv:  str = None,
    unseen_csv: str = None,
    run_id: str = "baseline",
    config: ExperimentConfig = None,
) -> dict:
    """
    Train and evaluate the FOP model.

    Args:
        train_csv  : override train CSV path (default: config-derived)
        val_csv    : override val CSV path
        test_csv   : override seen-test CSV path
        unseen_csv : override unseen-test CSV path
        run_id     : label written to results CSVs (e.g. 'original', 'mixed')
        config     : ExperimentConfig instance; created fresh if None

    Returns:
        dict keyed by alpha → {'seen', 'val', 'unseen', 'best_epoch'}
    """
    if config is None:
        config = ExperimentConfig()

    set_all_seeds(config.seed)
    logger = setup_logger(config)

    # ---- resolve CSV paths ----
    tracker = os.path.join(config.home_dir, "feature_tracker")
    train_csv  = train_csv  or os.path.join(tracker, f"{config.version}_train_{config.seen_lang}.csv")
    val_csv    = val_csv    or os.path.join(tracker, f"{config.version}_val_{config.seen_lang}.csv")
    test_csv   = test_csv   or os.path.join(tracker, f"{config.version}_test_{config.seen_lang}.csv")
    unseen_csv = unseen_csv or os.path.join(tracker, f"{config.version}_test_{config.unseen_lang}.csv")

    logger.info("=== FOP Experiment — run_id=%s ===", run_id)
    logger.info("seed=%d | device=%s | fusion=%s | version=%s | seen=%s | unseen=%s",
                config.seed, config.device, config.fusion,
                config.version, config.seen_lang, config.unseen_lang)

    # ---- data ----
    _, train_loader     = make_loader(train_csv,  config, shuffle=True)
    val_dataset,   _    = make_loader(val_csv,    config, shuffle=False)
    test_dataset,  _    = make_loader(test_csv,   config, shuffle=False)
    unseen_dataset, _   = make_loader(unseen_csv, config, shuffle=False)

    audio_sample, face_sample, _ = next(iter(train_loader))
    logger.info("Feature dims | audio=%d | face=%d", audio_sample.shape[1], face_sample.shape[1])

    # ---- result file paths ----
    results_csv = os.path.join(config.results_dir, "results.csv")
    summary_csv = os.path.join(config.results_dir, "summary.csv")
    os.makedirs(config.results_dir, exist_ok=True)

    all_results = {}

    for alpha in config.alpha_list:
        logger.info("--- alpha=%.4f ---", alpha)
        set_all_seeds(config.seed)   # reset seed for each alpha for fair comparison

        model = FOP(config=config, face_dim=face_sample.shape[1], voice_dim=audio_sample.shape[1])
        trainer   = Trainer(model, config)
        evaluator = Evaluator(model, config)
        stopper   = EarlyStopping(
            patience=config.early_stop_patience,
            min_delta=config.early_stop_min_delta,
        )

        ckpt_path = os.path.join(
            config.results_dir,
            f"{config.version}_{config.seen_lang}_alpha{alpha}_best.pt",
        )

        best = {"seen": 0.0, "val": 0.0, "unseen": 0.0, "epoch": 0}

        for epoch in range(config.max_epochs):
            loss = trainer.train_epoch(train_loader, alpha, epoch=epoch)

            acc_seen   = evaluator.accuracy(test_dataset)
            acc_val    = evaluator.accuracy(val_dataset)
            acc_unseen = evaluator.accuracy(unseen_dataset)

            # which metric drives early stopping and checkpointing
            monitor = {"seen": acc_seen, "val": acc_val, "unseen": acc_unseen}.get(
                config.early_stop_metric, acc_seen
            )

            if monitor > best.get(config.early_stop_metric, -1):
                best = {"seen": acc_seen, "val": acc_val, "unseen": acc_unseen, "epoch": epoch}
                save_checkpoint(model, trainer.opt, config, epoch,
                                best, ckpt_path)

            logger.info(
                "[a=%.4f] E%03d | loss=%.4f | seen=%.2f | val=%.2f | unseen=%.2f",
                alpha, epoch, loss, acc_seen, acc_val, acc_unseen,
            )

            append_epoch_row(results_csv, [
                run_id, alpha, epoch, round(loss, 6),
                round(acc_seen, 4), round(acc_val, 4), round(acc_unseen, 4),
            ])

            if config.early_stop and stopper.step(monitor):
                logger.info("Early stop at epoch %d (best=%.2f)", epoch, stopper.best_score)
                break

        all_results[alpha] = best
        append_summary_row(summary_csv, [
            run_id, alpha,
            best["epoch"], round(best["seen"], 4),
            round(best["val"], 4), round(best["unseen"], 4),
            config.seed, config.fusion, config.embedding_dim,
            config.lr, config.batch_size,
        ])
        logger.info("[a=%.4f] Best -> seen=%.2f | val=%.2f | unseen=%.2f (epoch %d)",
                    alpha, best["seen"], best["val"], best["unseen"], best["epoch"])

    logger.info("=== Done. Results saved to %s ===", config.results_dir)
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FOP baseline")
    parser.add_argument("--train_csv",  default=None)
    parser.add_argument("--val_csv",    default=None)
    parser.add_argument("--test_csv",   default=None)
    parser.add_argument("--unseen_csv", default=None)
    parser.add_argument("--run_id",     default="baseline")
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--device",     default=None, choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--max_epochs", type=int,   default=None)
    parser.add_argument("--results_dir",default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.seed       is not None: cfg.seed       = args.seed
    if args.device     is not None: cfg.device     = args.device
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.max_epochs is not None: cfg.max_epochs = args.max_epochs
    if args.results_dir is not None: cfg.results_dir = args.results_dir

    main(
        train_csv  = args.train_csv,
        val_csv    = args.val_csv,
        test_csv   = args.test_csv,
        unseen_csv = args.unseen_csv,
        run_id     = args.run_id,
        config     = cfg,
    )
