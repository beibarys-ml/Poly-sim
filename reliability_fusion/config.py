from pathlib import Path

# Project root: reliability-fusion-project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CSV metadata
FEATURE_TRACKER_DIR = PROJECT_ROOT / "feature_tracker"

TRAIN_CSV = {
    "English": FEATURE_TRACKER_DIR / "v3_train_English.csv",
    "German": FEATURE_TRACKER_DIR / "v3_train_German.csv",
}

VAL_CSV = {
    "English": FEATURE_TRACKER_DIR / "v3_val_English.csv",
    "German": FEATURE_TRACKER_DIR / "v3_val_German.csv",
}

TEST_CSV = {
    "English": FEATURE_TRACKER_DIR / "v3_test_English.csv",
    "German": FEATURE_TRACKER_DIR / "v3_test_German.csv",
}

# External professor baseline feature root.
# This folder is NOT copied into our clean project.
FOP_ROOT = Path(r"C:\Users\beiba\Desktop\baseline\polysim-main\FOP")

# Model dimensions from your feature inspection
AUDIO_DIM = 192      # ECAPA-TDNN embedding
FACE_DIM = 512       # FaceNet embedding
HIDDEN_DIM = 512

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3

# Face-dropout probability during training.
# This simulates missing visual modality.
FACE_DROPOUT_P = 0.3

# Paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Reproducibility
SEED = 42