"""
generate_csvs.py — Build train / val / test CSV splits from existing .npy embeddings

Scans ecappafeats/ and facenetfeats/ and produces:
  feature_tracker/v3_train_German.csv      (70% of German clips)
  feature_tracker/v3_val_German.csv        (10% of German clips)
  feature_tracker/v3_test_German.csv       (20% of German clips — seen)
  feature_tracker/v3_test_English.csv      (all English clips — unseen / zero-shot)

CSV format (3 columns, no header needed by featLoader but written for clarity):
  ecappa_feats_path, facenet_feats_path, label

Label assignment: speaker IDs sorted alphabetically → 0-indexed integers.
  id0001→0, id0002→1, id0004→2, … (deterministic)

Pairing strategy: cross-product of ecappa utterances × facenet frames within the
same (speaker, language, video_id) clip. This mirrors the original dataset structure.

Usage:
  python generate_csvs.py                        # uses defaults
  python generate_csvs.py --version v3 --seed 42
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def build_speaker_map(ecappa_root: Path, version: str) -> dict[str, int]:
    """Sort all speaker IDs alphabetically → deterministic 0-indexed labels."""
    voices_dir = ecappa_root / version / "voices"
    speaker_ids = sorted(p.name for p in voices_dir.iterdir() if p.is_dir())
    return {sid: idx for idx, sid in enumerate(speaker_ids)}


# ---------------------------------------------------------------------------
# Pairing: ecappa × facenet within a clip
# ---------------------------------------------------------------------------

def pair_clip(
    ecappa_clip_dir: Path,
    facenet_clip_dir: Path,
    ecappa_root: Path,    # base for relative paths
    label: int,
) -> list[tuple[str, str, int]]:
    """
    Cross-product: every ecappa utterance file × every facenet frame file
    in the same clip directory.
    """
    ecappa_files = sorted(ecappa_clip_dir.glob("*.npy"))
    facenet_files = sorted(facenet_clip_dir.glob("*.npy"))

    if not ecappa_files or not facenet_files:
        return []

    rows = []
    for ef in ecappa_files:
        for ff in facenet_files:
            rows.append((
                str(ef.relative_to(ecappa_root)).replace("\\", "/"),
                str(ff.relative_to(ecappa_root)).replace("\\", "/"),
                label,
            ))
    return rows


# ---------------------------------------------------------------------------
# Scan all clips for a language
# ---------------------------------------------------------------------------

def scan_language(
    fop_root: Path,
    language: str,
    version: str,
    speaker_map: dict[str, int],
) -> dict[tuple[str, str], list[tuple[str, str, int]]]:
    """
    Returns a dict keyed by (speaker_id, video_id) → list of CSV rows.
    Keying by clip allows stratified train/val/test splitting.
    """
    ecappa_lang_root = fop_root / "ecappafeats" / version / "voices"
    facenet_lang_root = fop_root / "facenetfeats" / version / "faces"

    clips: dict[tuple[str, str], list] = {}

    for speaker_id, label in sorted(speaker_map.items()):
        ecappa_lang_dir = ecappa_lang_root / speaker_id / language
        facenet_lang_dir = facenet_lang_root / speaker_id / language

        if not ecappa_lang_dir.exists() or not facenet_lang_dir.exists():
            continue

        for video_dir in sorted(ecappa_lang_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            facenet_video_dir = facenet_lang_dir / video_id

            if not facenet_video_dir.exists():
                continue

            rows = pair_clip(video_dir, facenet_video_dir, fop_root, label)
            if rows:
                clips[(speaker_id, video_id)] = rows

    return clips


# ---------------------------------------------------------------------------
# Stratified split at clip level
# ---------------------------------------------------------------------------

def stratified_split(
    clips: dict[tuple[str, str], list],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split clips into train / val / test at the clip level, stratified by speaker.
    Returns (train_rows, val_rows, test_rows).
    """
    rng = random.Random(seed)

    # Group clip keys by speaker
    by_speaker: dict[str, list] = defaultdict(list)
    for (speaker_id, video_id) in clips:
        by_speaker[speaker_id].append((speaker_id, video_id))

    train_rows, val_rows, test_rows = [], [], []

    for speaker_id in sorted(by_speaker):
        clip_keys = by_speaker[speaker_id]
        rng.shuffle(clip_keys)

        n = len(clip_keys)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(0, round(n * val_ratio))
        # test gets the remainder

        for i, key in enumerate(clip_keys):
            rows = clips[key]
            if i < n_train:
                train_rows.extend(rows)
            elif i < n_train + n_val:
                val_rows.extend(rows)
            else:
                test_rows.extend(rows)

    return train_rows, val_rows, test_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[tuple[str, str, int]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ecappa_feats_path", "facenet_feats_path", "label"])
        for ecappa, facenet, label in rows:
            writer.writerow([ecappa, facenet, label])
    print(f"  Wrote {len(rows):>5} rows -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_csvs(
    fop_root: str = ".",
    version: str = "v3",
    seen_lang: str = "German",
    unseen_lang: str = "English",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    root = Path(fop_root).resolve()
    tracker = root / "feature_tracker"

    print(f"Scanning embeddings in: {root}")
    speaker_map = build_speaker_map(root / "ecappafeats", version)
    print(f"Found {len(speaker_map)} speakers: {list(speaker_map.keys())}")

    # ---- Seen language (German) — train / val / test split ----
    print(f"\nScanning {seen_lang} clips …")
    seen_clips = scan_language(root, seen_lang, version, speaker_map)
    print(f"  {len(seen_clips)} clips, "
          f"{sum(len(v) for v in seen_clips.values())} pairs total")

    train, val, test_seen = stratified_split(
        seen_clips, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    write_csv(train,     str(tracker / f"{version}_train_{seen_lang}.csv"))
    write_csv(val,       str(tracker / f"{version}_val_{seen_lang}.csv"))
    write_csv(test_seen, str(tracker / f"{version}_test_{seen_lang}.csv"))

    # ---- Unseen language (English) — all clips go to test ----
    print(f"\nScanning {unseen_lang} clips …")
    unseen_clips = scan_language(root, unseen_lang, version, speaker_map)
    all_unseen = [row for rows in unseen_clips.values() for row in rows]
    print(f"  {len(unseen_clips)} clips, {len(all_unseen)} pairs total")

    write_csv(all_unseen, str(tracker / f"{version}_test_{unseen_lang}.csv"))

    print(f"\nDone. CSVs written to: {tracker}")
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  (seen)  : {len(test_seen)}")
    print(f"  Test  (unseen): {len(all_unseen)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/val/test CSV splits")
    parser.add_argument("--fop_root",    default=".",      help="Path to FOP directory")
    parser.add_argument("--version",     default="v3")
    parser.add_argument("--seen_lang",   default="German")
    parser.add_argument("--unseen_lang", default="English")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    generate_all_csvs(
        fop_root    = args.fop_root,
        version     = args.version,
        seen_lang   = args.seen_lang,
        unseen_lang = args.unseen_lang,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )
