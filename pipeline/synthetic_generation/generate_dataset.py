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

      python generate_dataset.py \
    --video_dir D:/data/videos/id0001/English/h_vamljclHE\
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

def rel_or_abs(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return "./" + str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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
) -> list:
    """
    Process one video: dub it, extract embeddings, return CSV rows.
    """
    video_path_obj = Path(video_path)
    video_id = f"{video_path_obj.parent.name}_{video_path_obj.stem}"

    logger.info("Processing %s / %s / %s", speaker_id, video_id, tgt_language_label)

    # --------------------------------------------------
    # Stage 1-5: dubbing pipeline
    # --------------------------------------------------
    try:
        result = run_pipeline(
            video_path=video_path,
            tgt_lang=tgt_lang,
            output_dir=pipeline_output_dir,
            src_lang="auto",
            wav2lip_dir=wav2lip_dir,
            wav2lip_checkpoint=wav2lip_checkpoint,
            whisper_model_size=whisper_model_size,
        )

        if result["output_video"] is None:
            raise RuntimeError("Pipeline finished but did not produce a lip-synced video.")

        video_for_facenet = result["output_video"]

    except Exception as exc:
        logger.warning(
            "Wav2Lip failed for %s. Falling back to original video for FaceNet. Error: %s",
            video_path,
            exc,
        )

        result = run_pipeline(
            video_path=video_path,
            tgt_lang=tgt_lang,
            output_dir=pipeline_output_dir,
            src_lang="auto",
            wav2lip_dir=wav2lip_dir,
            wav2lip_checkpoint=wav2lip_checkpoint,
            whisper_model_size=whisper_model_size,
            skip_lipsync=True,
        )

        video_for_facenet = video_path

    # --------------------------------------------------
    # Extract + save synthetic embeddings
    # --------------------------------------------------
    ecapa_abs, facenet_abs_list = save_embeddings(
        audio_path=result["tts_audio"],
        video_path=video_for_facenet,
        speaker_id=speaker_id,
        language=tgt_language_label,
        video_id=video_id,
        root_dir=output_root,
        synthetic=True,
    )

    # --------------------------------------------------
    # Convert paths to CSV format
    # --------------------------------------------------
    root = Path(output_root).resolve()

    ecapa_rel = rel_or_abs(ecapa_abs, root)
    audio_rel = rel_or_abs(result["tts_audio"], root)
    video_rel = rel_or_abs(video_for_facenet, root)

    rows = []

    for facenet_abs in facenet_abs_list:
        facenet_rel = rel_or_abs(facenet_abs, root)

        rows.append([
            audio_rel,
            video_rel,
            speaker_id,
            facenet_rel,
            int(label),
            ecapa_rel,
            "",  # wavlmsv_feats_path not extracted
        ])

    return rows


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_process(
    video_dir: str,
    speaker_map: dict[str, int],
    src_language_label: str = "English",
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

        source_lang_dir = speaker_dir / src_language_label
        if not source_lang_dir.exists():
            logger.warning(
                "Source language directory not found for %s: %s",
                speaker_id,
                source_lang_dir,
            )
            continue

        video_files = sorted(
            list(source_lang_dir.rglob("*.mp4")) +
            list(source_lang_dir.rglob("*.avi"))
            )

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


def save_csv_rows(rows, out_path: str) -> None:
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
    parser.add_argument("--src_language_label", default="English")
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
    src_language_label=args.src_language_label,
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
