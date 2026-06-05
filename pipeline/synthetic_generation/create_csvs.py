"""
create_csvs.py — Build CSV split files for the FOP baseline (Task 2 / Task 3)

Scans the ecappafeats/ and facenetfeats/ directories and generates the
three-column CSV files consumed by featLoader.LoadData:

    ecappa_feats_path, facenet_feats_path, label

Outputs (written to feature_tracker/):
  v3_train_German.csv           — original German training data
  v3_test_German.csv            — original German test data
  v3_test_English.csv           — original English test data
  v3_train_German_synthetic.csv — synthetic German data only
  v3_train_German_mixed.csv     — original + synthetic German training data

Label mapping: speaker IDs sorted alphabetically → 0-indexed integers.
  id0001 → 0, id0002 → 1, … (consistent with any existing embeddings)

Usage:
  python create_csvs.py --fop_root /path/to/Poly-sim/FOP
"""

import argparse
import csv
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Speaker → label mapping
# ---------------------------------------------------------------------------

def build_speaker_map_from_csv(csv_path: Path) -> dict[str, int]:
    """
    Build speaker_id -> label mapping from the original feature_tracker CSV.
    This preserves the exact label mapping used by the provided dataset.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    if "identity" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain 'identity' and 'label' columns."
        )

    speaker_map = (
        df[["identity", "label"]]
        .drop_duplicates()
        .set_index("identity")["label"]
        .astype(int)
        .to_dict()
    )

    return speaker_map


# ---------------------------------------------------------------------------
# Embedding scanning
# ---------------------------------------------------------------------------

def scan_embeddings(
    fop_root: Path,
    language: str,
    version: str = "v3",
    speaker_map: dict[str, int] = None,
    synthetic: bool = False,
) -> list[tuple[str, str, int]]:
    """
    Walk ecappafeats and facenetfeats for a given language and pair them.

    Pairing strategy:
      - One ECAPA file per (speaker_id, video_id): the 00000.npy file
      - Multiple FaceNet files per (speaker_id, video_id): one per frame
      - Each ECAPA file is paired with every FaceNet file in the same clip,
        matching the original CSV structure

    Args:
        fop_root     : Path to the FOP directory
        language     : e.g. 'German' or 'English'
        version      : dataset version, e.g. 'v3'
        speaker_map  : dict mapping speaker_id → label; built automatically if None
        synthetic    : if True, scan ecappafeats_synthetic/ and facenetfeats_synthetic/

    Returns:
        List of (ecappa_rel_path, facenet_rel_path, label) tuples
    """
    prefix = "_synthetic" if synthetic else ""
    ecappa_base = fop_root / f"ecappafeats{prefix}" / version / "voices"
    facenet_base = fop_root / f"facenetfeats{prefix}" / version / "faces"

    if not ecappa_base.exists():
        logger.warning("Directory not found: %s", ecappa_base)
        return []

    if speaker_map is None:
        speaker_map = build_speaker_map(fop_root, version)

    rows = []

    for speaker_dir in sorted(ecappa_base.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name

        if speaker_id not in speaker_map:
            logger.warning("Speaker %s not in speaker_map, skipping", speaker_id)
            continue

        label = speaker_map[speaker_id]
        lang_dir = speaker_dir / language

        if not lang_dir.exists():
            continue

        for video_dir in sorted(lang_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name

            # The ECAPA embedding for this clip
            ecappa_file = video_dir / "00000.npy"
            if not ecappa_file.exists():
                logger.warning("Missing ECAPA file: %s", ecappa_file)
                continue

            # All FaceNet embeddings for this clip
            facenet_dir = facenet_base / speaker_id / language / video_id
            if not facenet_dir.exists():
                logger.warning("Missing FaceNet dir: %s", facenet_dir)
                continue

            face_files = sorted(facenet_dir.glob("*.npy"))
            if not face_files:
                logger.warning("No FaceNet files in: %s", facenet_dir)
                continue

            # Relative paths from fop_root
            ecappa_rel = str(ecappa_file.relative_to(fop_root))
            for ff in face_files:
                facenet_rel = str(ff.relative_to(fop_root))
                rows.append((ecappa_rel, facenet_rel, label))

    logger.info(
        "Scanned %s / %s%s: %d rows, %d speakers",
        language, version, " (synthetic)" if synthetic else "",
        len(rows),
        len({r[2] for r in rows}),
    )
    return rows


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def split_rows(
    rows: list[tuple[str, str, int]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Stratified split: for each speaker, hold out `test_ratio` of their clips
    as test data.  Split is done at the video-clip level to avoid data leakage.
    """
    from collections import defaultdict

    # Group rows by (label, video_id inferred from ecappa path)
    clips: dict[tuple[int, str], list] = defaultdict(list)
    for row in rows:
        ecappa_path, facenet_path, label = row
        # video_id is the parent folder of 00000.npy
        video_id = Path(ecappa_path).parent.name
        clips[(label, video_id)].append(row)

    rng = random.Random(seed)
    train_rows, test_rows = [], []

    # Group clips by label for stratification
    by_label: dict[int, list] = defaultdict(list)
    for key in clips:
        label = key[0]
        by_label[label].append(key)

    for label, clip_keys in by_label.items():
        rng.shuffle(clip_keys)
        n_test = max(1, round(len(clip_keys) * test_ratio))
        test_keys = set(clip_keys[:n_test])
        for key in clip_keys:
            target = test_rows if key in test_keys else train_rows
            target.extend(clips[key])

    return train_rows, test_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[tuple[str, str, int]], out_path: str) -> None:
    """
    Write CSV in the same format as the provided feature_tracker files.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    header = [
        "audio_path",
        "face_path",
        "identity",
        "facenet_feats_path",
        "label",
        "ecappa_feats_path",
        "wavlmsv_feats_path",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ecappa, facenet, label in rows:
            ecappa = ecappa.replace("\\", "/")
            facenet = facenet.replace("\\", "/")

            # Example:
            # ecappafeats_synthetic/v3/voices/id0001/German/h_vamljclHE/00000.npy
            parts = Path(ecappa).parts

            try:
                speaker_id = parts[3]  # id0001
            except Exception:
                speaker_id = str(label)

            writer.writerow([
                "",          # audio_path; optional, not used by your loader
                "",          # face_path; optional, not used by your loader
                speaker_id,
                ecappa_path_to_feature_style(facenet),
                int(label),
                ecappa_path_to_feature_style(ecappa),
                "",          # wavlmsv_feats_path; not extracted
            ])

    logger.info("CSV written → %s (%d rows)", out_path, len(rows))


def ecappa_path_to_feature_style(path: str) -> str:
    """
    Match existing feature_tracker path style.
    Adds './' if not already present.
    """
    path = path.replace("\\", "/")
    if not path.startswith("./"):
        path = "./" + path
    return path


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_all_csvs(
    fop_root: str,
    version: str = "v3",
    seen_lang: str = "German",
    unseen_lang: str = "English",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Build synthetic and mixed CSV files while keeping original feature_tracker
    CSVs unchanged.
    """
    root = Path(fop_root)
    tracker_dir = root / "feature_tracker"
    tracker_dir.mkdir(exist_ok=True)

    original_train_csv = tracker_dir / f"{version}_train_{seen_lang}.csv"

    if not original_train_csv.exists():
        raise FileNotFoundError(f"Original train CSV not found: {original_train_csv}")

    speaker_map = build_speaker_map_from_csv(original_train_csv)
    logger.info("Speaker map loaded from original CSV: %d identities", len(speaker_map))

    # Scan synthetic embeddings only
    synth_rows = scan_embeddings(
        root,
        seen_lang,
        version,
        speaker_map,
        synthetic=True,
    )

    if not synth_rows:
        logger.info(
            "No synthetic embeddings found. Make sure ecappafeats_synthetic/ "
            "and facenetfeats_synthetic/ exist."
        )
        return

    synthetic_csv = tracker_dir / f"{version}_train_{seen_lang}_synthetic.csv"
    write_csv(synth_rows, str(synthetic_csv))

    # Create mixed CSV = original train + synthetic rows
    import pandas as pd

    original_df = pd.read_csv(original_train_csv)
    synthetic_df = pd.read_csv(synthetic_csv)

    mixed_df = pd.concat([original_df, synthetic_df], ignore_index=True)

    mixed_csv = tracker_dir / f"{version}_train_{seen_lang}_mixed.csv"
    mixed_df.to_csv(mixed_csv, index=False)

    logger.info("Mixed CSV written → %s (%d rows)", mixed_csv, len(mixed_df))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CSV split files for FOP baseline")
    parser.add_argument("--fop_root", default=".", help="Path to the FOP directory")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--seen_lang", default="German")
    parser.add_argument("--unseen_lang", default="English")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of clips held out for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_all_csvs(
        fop_root=args.fop_root,
        version=args.version,
        seen_lang=args.seen_lang,
        unseen_lang=args.unseen_lang,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("\nAll CSVs built successfully.")
