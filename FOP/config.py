from dataclasses import dataclass, field
from typing import List, Tuple
import logging
import os


@dataclass
class ExperimentConfig:
    # -----------------------------------------------------------------------
    # Paths — auto-resolved so the project works on any machine
    # -----------------------------------------------------------------------
    home_dir: str = os.path.dirname(os.path.abspath(__file__))

    # -----------------------------------------------------------------------
    # Reproducibility — all randomness is controlled by a single seed
    # -----------------------------------------------------------------------
    seed: int = 42

    # -----------------------------------------------------------------------
    # Hardware
    # -----------------------------------------------------------------------
    device: str = "cuda"        # "cuda" | "cpu"
    num_workers: int = 4        # set to 0 on Windows if multiprocessing errors occur

    # -----------------------------------------------------------------------
    # Training hyperparameters
    # -----------------------------------------------------------------------
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 300
    alpha_list: Tuple[float, ...] = (0.0,)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    embedding_dim: int = 512
    fusion: str = "linear"      # "linear" | "gated"

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    version: str = "v3"
    seen_lang: str = "German"   # "English" | "Hindi" | "German"

    # -----------------------------------------------------------------------
    # Missing modality simulation (training-time augmentation)
    # -----------------------------------------------------------------------
    train_missing_modality: str = "face"   # "face" | "audio" | None
    missing_ratio: float = 0.0             # fraction of batch to zero out

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    debug: bool = False

    @property
    def log_level(self) -> int:
        return logging.DEBUG if self.debug else logging.INFO

    # -----------------------------------------------------------------------
    # Early stopping
    # -----------------------------------------------------------------------
    early_stop: bool = True
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.2
    early_stop_metric: str = "seen"        # "seen" | "unseen" | "val"

    # -----------------------------------------------------------------------
    # Results output
    # -----------------------------------------------------------------------
    results_dir: str = "./results"         # CSV + checkpoint root

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------
    @property
    def resolved_num_classes(self) -> int:
        _map = {"v1": 70, "v2": 84, "v3": 36}
        if self.version not in _map:
            raise ValueError(f"Unknown version '{self.version}'")
        return _map[self.version]

    @property
    def unseen_lang(self) -> str:
        _map = {
            ("v1", "Urdu"):   "English",
            ("v2", "Hindi"):  "English",
            ("v3", "German"): "English",
        }
        key = (self.version, self.seen_lang)
        if key not in _map:
            raise ValueError(
                f"No unseen_lang mapping for version='{self.version}', "
                f"seen_lang='{self.seen_lang}'."
            )
        return _map[key]
