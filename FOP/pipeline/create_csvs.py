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

def build_speaker_map(ecappa_root: Path, version: str = "v3") -> dict[str, int]:
    """
    Derive the speaker_id → label mapping by sorting all speaker IDs found
    under ecappafeats/{version}/voices/ alphabetically.

    This ensures the mapping is deterministic and consistent across runs.
    """
    voices_dir = ecappa_root / "ecappafeats" / version / "voices"
    if not voices_dir.exists():
        raise FileNotFoundError(f"Voices directory not found: {voices_dir}")

    speaker_ids = sorted(p.name for p in voices_dir.iterdir() if p.is_dir())
    return {sid: idx for idx, sid in enumerate(speaker_ids)}


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
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Normalise path separators for cross-platform compatibility
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ecappa_feats_path", "facenet_feats_path", "label"])
        for ecappa, facenet, label in rows:
            writer.writerow([
                ecappa.replace("\\", "/"),
                facenet.replace("\\", "/"),
                label,
            ])
    logger.info("CSV written → %s  (%d rows)", out_path, len(rows))


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
    Build all CSV split files required for the FOP baseline experiments.

    Creates under {fop_root}/feature_tracker/:
      v3_train_{seen_lang}.csv
      v3_test_{seen_lang}.csv
      v3_test_{unseen_lang}.csv
      v3_train_{seen_lang}_synthetic.csv   (if synthetic embeddings exist)
      v3_train_{seen_lang}_mixed.csv       (original + synthetic, if synthetic exists)
    """
    root = Path(fop_root)
    tracker_dir = root / "feature_tracker"
    tracker_dir.mkdir(exist_ok=True)

    speaker_map = build_speaker_map(root, version)
    logger.info("Speaker map: %d identities", len(speaker_map))

    # --- Original seen language (German) ---
    seen_rows = scan_embeddings(root, seen_lang, version, speaker_map, synthetic=False)
    if seen_rows:
        train_rows, test_rows = split_rows(seen_rows, test_ratio=test_ratio, seed=seed)
        write_csv(train_rows, str(tracker_dir / f"{version}_train_{seen_lang}.csv"))
        write_csv(test_rows,  str(tracker_dir / f"{version}_test_{seen_lang}.csv"))
    else:
        logger.warning("No original %s embeddings found — skipping seen-lang CSVs", seen_lang)

    # --- Original unseen language (English) ---
    unseen_rows = scan_embeddings(root, unseen_lang, version, speaker_map, synthetic=False)
    if unseen_rows:
        # All unseen data is used as test set (zero-shot evaluation)
        write_csv(unseen_rows, str(tracker_dir / f"{version}_test_{unseen_lang}.csv"))
    else:
        logger.warning("No original %s embeddings found — skipping unseen-lang CSV", unseen_lang)

    # --- Synthetic seen language ---
    synth_rows = scan_embeddings(root, seen_lang, version, speaker_map, synthetic=True)
    if synth_rows:
        write_csv(synth_rows, str(tracker_dir / f"{version}_train_{seen_lang}_synthetic.csv"))

        # Mixed: original train + all synthetic
        if seen_rows:
            mixed_rows = train_rows + synth_rows
            write_csv(mixed_rows, str(tracker_dir / f"{version}_train_{seen_lang}_mixed.csv"))
    else:
        logger.info(
            "No synthetic embeddings found (run generate_dataset.py first). "
            "Skipping synthetic and mixed CSVs."
        )


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
