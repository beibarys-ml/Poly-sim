"""
generate_dataset.py — Batch dubbing + embedding extraction runner

For each source video:
  1. Run the full dubbing pipeline (ASR → translate → TTS → lip-sync)
  2. Extract ECAPA-TDNN and FaceNet embeddings from the generated audio/video
  3. Save embeddings in the standard directory layout
  4. Collect CSV rows for downstream create_csvs.py

Expected input layout:
  video_dir/
    {speaker_id}/
      {video_id}.mp4   (e.g. id0001/h_vamljclHE.mp4)

Usage:
  python generate_dataset.py \
    --video_dir /path/to/raw_videos \
    --output_root /path/to/Poly-sim/FOP \
    --tgt_lang de \
    --tgt_language_label German
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

from pipeline import run_pipeline
from extract_embeddings import save_embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Single video processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    speaker_id: str,
    label: int,
    tgt_lang: str,
    tgt_language_label: str,
    output_root: str,
    pipeline_output_dir: str = "./pipeline_output",
    wav2lip_dir: str = "Wav2Lip",
    wav2lip_checkpoint: str = "Wav2Lip/checkpoints/wav2lip_gan.pth",
    whisper_model_size: str = "base",
) -> list[tuple[str, str, int]]:
    """
    Process one video: dub it, extract embeddings, return CSV rows.

    Each video can yield multiple FaceNet embeddings (one per frame), each
    paired with the single ECAPA embedding for that clip — mirroring the
    one-to-one structure of the original dataset CSVs.

    Args:
        video_path          : path to source video
        speaker_id          : e.g. 'id0001'
        label               : integer class label
        tgt_lang            : ISO 639-1 target language ('de')
        tgt_language_label  : human-readable label for directory ('German')
        output_root         : FOP project root where ecappafeats/ and facenetfeats/ live
        pipeline_output_dir : working directory for intermediate pipeline files
        wav2lip_dir         : path to Wav2Lip repository
        wav2lip_checkpoint  : path to Wav2Lip checkpoint
        whisper_model_size  : Whisper model size

    Returns:
        List of (ecappa_rel_path, facenet_rel_path, label) tuples ready for CSV
    """
    video_id = Path(video_path).stem

    logger.info("Processing %s / %s / %s", speaker_id, video_id, tgt_language_label)

    # Stage 1–5: dubbing pipeline
    result = run_pipeline(
        video_path=video_path,
        tgt_lang=tgt_lang,
        output_dir=pipeline_output_dir,
        src_lang="auto",
        wav2lip_dir=wav2lip_dir,
        wav2lip_checkpoint=wav2lip_checkpoint,
        whisper_model_size=whisper_model_size,
    )

    # Extract + save embeddings
    ecapa_abs, facenet_abs_list = save_embeddings(
        audio_path=result["tts_audio"],
        video_path=result["output_video"],
        speaker_id=speaker_id,
        language=tgt_language_label,
        video_id=video_id,
        root_dir=output_root,
    )

    # Convert absolute paths to paths relative to output_root (for CSV)
    root = Path(output_root).resolve()
    ecapa_rel = str(Path(ecapa_abs).resolve().relative_to(root))

    rows = []
    for facenet_abs in facenet_abs_list:
        facenet_rel = str(Path(facenet_abs).resolve().relative_to(root))
        rows.append((ecapa_rel, facenet_rel, label))

    return rows


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_process(
    video_dir: str,
    speaker_map: dict[str, int],
    tgt_lang: str = "de",
    tgt_language_label: str = "German",
    output_root: str = ".",
    pipeline_output_dir: str = "./pipeline_output",
    wav2lip_dir: str = "Wav2Lip",
    wav2lip_checkpoint: str = "Wav2Lip/checkpoints/wav2lip_gan.pth",
    whisper_model_size: str = "base",
) -> list[tuple[str, str, int]]:
    """
    Process all videos under video_dir, returning CSV rows for all clips.

    Expected video_dir layout:
        {video_dir}/{speaker_id}/{video_id}.mp4

    Args:
        video_dir          : root directory containing speaker sub-folders
        speaker_map        : dict mapping speaker_id → integer class label
        tgt_lang           : ISO 639-1 target language code
        tgt_language_label : directory label (e.g. 'German')
        output_root        : FOP project root
        pipeline_output_dir: working directory for pipeline intermediates
        wav2lip_dir        : path to Wav2Lip repository
        wav2lip_checkpoint : path to Wav2Lip .pth checkpoint
        whisper_model_size : Whisper model size

    Returns:
        Flat list of (ecappa_rel_path, facenet_rel_path, label) CSV rows
    """
    all_rows = []
    errors = []

    video_root = Path(video_dir)

    for speaker_id, label in speaker_map.items():
        speaker_dir = video_root / speaker_id
        if not speaker_dir.exists():
            logger.warning("Speaker directory not found: %s", speaker_dir)
            continue

        video_files = sorted(speaker_dir.glob("*.mp4"))
        if not video_files:
            logger.warning("No .mp4 files found for %s", speaker_id)
            continue

        for video_path in video_files:
            try:
                rows = process_video(
                    video_path=str(video_path),
                    speaker_id=speaker_id,
                    label=label,
                    tgt_lang=tgt_lang,
                    tgt_language_label=tgt_language_label,
                    output_root=output_root,
                    pipeline_output_dir=pipeline_output_dir,
                    wav2lip_dir=wav2lip_dir,
                    wav2lip_checkpoint=wav2lip_checkpoint,
                    whisper_model_size=whisper_model_size,
                )
                all_rows.extend(rows)
                logger.info("Done: %s/%s → %d rows", speaker_id, video_path.name, len(rows))

            except Exception as exc:
                logger.error("Failed %s/%s: %s", speaker_id, video_path.name, exc)
                errors.append({"speaker_id": speaker_id, "video": str(video_path), "error": str(exc)})

    if errors:
        err_path = os.path.join(pipeline_output_dir, "errors.json")
        os.makedirs(pipeline_output_dir, exist_ok=True)
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        logger.warning("%d errors logged → %s", len(errors), err_path)

    logger.info("Batch complete: %d total CSV rows from %d speakers", len(all_rows), len(speaker_map))
    return all_rows


def save_csv_rows(rows: list[tuple[str, str, int]], out_path: str) -> None:
    """Write CSV rows to file with the standard 3-column format."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ecappa_feats_path", "facenet_feats_path", "label"])
        writer.writerows(rows)
    logger.info("CSV saved → %s  (%d rows)", out_path, len(rows))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch dub videos and extract embeddings")
    parser.add_argument("--video_dir", required=True,
                        help="Root dir with speaker sub-folders containing .mp4 files")
    parser.add_argument("--speaker_map", required=True,
                        help="JSON file mapping speaker_id → label, e.g. {'id0001':0,'id0002':1}")
    parser.add_argument("--output_root", default=".",
                        help="FOP project root where ecappafeats/ and facenetfeats/ live")
    parser.add_argument("--tgt_lang", default="de")
    parser.add_argument("--tgt_language_label", default="German")
    parser.add_argument("--out_csv", default="./feature_tracker/v3_train_German_synthetic.csv")
    parser.add_argument("--pipeline_output_dir", default="./pipeline_output")
    parser.add_argument("--wav2lip_dir", default="Wav2Lip")
    parser.add_argument("--wav2lip_checkpoint", default="Wav2Lip/checkpoints/wav2lip_gan.pth")
    parser.add_argument("--whisper_model", default="base")
    args = parser.parse_args()

    with open(args.speaker_map) as f:
        speaker_map = json.load(f)

    rows = batch_process(
        video_dir=args.video_dir,
        speaker_map=speaker_map,
        tgt_lang=args.tgt_lang,
        tgt_language_label=args.tgt_language_label,
        output_root=args.output_root,
        pipeline_output_dir=args.pipeline_output_dir,
        wav2lip_dir=args.wav2lip_dir,
        wav2lip_checkpoint=args.wav2lip_checkpoint,
        whisper_model_size=args.whisper_model,
    )

    save_csv_rows(rows, args.out_csv)
    print(f"\nDone. {len(rows)} rows written to {args.out_csv}")
