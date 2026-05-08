"""
extract_embeddings.py — ECAPA-TDNN + FaceNet embedding extraction (Task 2)

Extracts:
  - Audio embeddings using SpeechBrain's ECAPA-TDNN (spkrec-ecapa-voxceleb)
  - Visual embeddings using FaceNet (InceptionResnetV1, pretrained on VGGFace2)

Saves embeddings as .npy files mirroring the existing directory structure:
  ecappafeats/v3/voices/{speaker_id}/{language}/{video_id}/00000.npy
  facenetfeats/v3/faces/{speaker_id}/{language}/{video_id}/{frame_idx:09d}.npy

Usage:
  python extract_embeddings.py \
    --audio path/to/audio.wav \
    --video path/to/video.mp4 \
    --speaker_id id0001 \
    --language German \
    --video_id abc123 \
    --root_dir ../
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from facenet_pytorch import MTCNN, InceptionResnetV1
from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# ECAPA-TDNN — audio speaker embeddings
# ---------------------------------------------------------------------------

_ecapa_model = None

def _get_ecapa():
    global _ecapa_model
    if _ecapa_model is None:
        logger.info("Loading SpeechBrain ECAPA-TDNN …")
        _ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE},
        )
    return _ecapa_model


def extract_ecapa(audio_path: str) -> np.ndarray:
    """
    Extract a single speaker embedding from an audio file.

    The model produces frame-level embeddings; we mean-pool them to get one
    fixed-size vector per utterance, matching the convention used for the
    existing 00000.npy files.

    Args:
        audio_path: path to wav file (any sample rate — resampled internally)

    Returns:
        np.ndarray of shape (192,)  [ECAPA output dimension]
    """
    model = _get_ecapa()

    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # SpeechBrain expects (1, T) float tensor on the correct device
    waveform = waveform.mean(dim=0, keepdim=True).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)  # (1, 1, D)

    return embedding.squeeze().cpu().numpy()  # (D,)


# ---------------------------------------------------------------------------
# FaceNet — visual face embeddings
# ---------------------------------------------------------------------------

_mtcnn = None
_facenet = None

def _get_facenet():
    global _mtcnn, _facenet
    if _mtcnn is None:
        logger.info("Loading MTCNN face detector …")
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            device=DEVICE,
            keep_all=False,
            post_process=True,
        )
    if _facenet is None:
        logger.info("Loading FaceNet (InceptionResnetV1, VGGFace2) …")
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return _mtcnn, _facenet


def extract_facenet(video_path: str, max_frames: int = 0) -> list[tuple[int, np.ndarray]]:
    """
    Extract per-frame face embeddings from a video.

    Frames without a detectable face are skipped. The frame index is kept so
    filenames can match the original convention (e.g. 000001550.npy).

    Args:
        video_path : path to video file
        max_frames : if > 0, stop after this many successfully extracted frames

    Returns:
        List of (frame_idx, embedding) where embedding has shape (512,)
    """
    mtcnn, facenet = _get_facenet()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    results = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Detect + align face; returns (160, 160, 3) tensor or None
        face_tensor = mtcnn(frame_rgb)

        if face_tensor is not None:
            with torch.no_grad():
                emb = facenet(face_tensor.unsqueeze(0).to(DEVICE))  # (1, 512)
            results.append((frame_idx, emb.squeeze().cpu().numpy()))

            if max_frames > 0 and len(results) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d face embeddings from %s", len(results), video_path)
    return results


# ---------------------------------------------------------------------------
# Save helpers — mirrors existing .npy directory layout
# ---------------------------------------------------------------------------

def save_ecapa_embedding(
    embedding: np.ndarray,
    root_dir: str,
    speaker_id: str,
    language: str,
    video_id: str,
) -> str:
    """
    Save ECAPA embedding as:
      {root_dir}/ecappafeats/v3/voices/{speaker_id}/{language}/{video_id}/00000.npy
    """
    out_dir = Path(root_dir) / "ecappafeats" / "v3" / "voices" / speaker_id / language / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "00000.npy"
    np.save(str(out_path), embedding)
    logger.info("ECAPA embedding saved → %s", out_path)
    return str(out_path)


def save_facenet_embeddings(
    frame_embeddings: list[tuple[int, np.ndarray]],
    root_dir: str,
    speaker_id: str,
    language: str,
    video_id: str,
) -> list[str]:
    """
    Save FaceNet embeddings as:
      {root_dir}/facenetfeats/v3/faces/{speaker_id}/{language}/{video_id}/{frame_idx:09d}.npy
    """
    out_dir = Path(root_dir) / "facenetfeats" / "v3" / "faces" / speaker_id / language / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for frame_idx, emb in frame_embeddings:
        out_path = out_dir / f"{frame_idx:09d}.npy"
        np.save(str(out_path), emb)
        saved_paths.append(str(out_path))

    logger.info("Saved %d FaceNet embeddings → %s", len(saved_paths), out_dir)
    return saved_paths


def save_embeddings(
    audio_path: str,
    video_path: str,
    speaker_id: str,
    language: str,
    video_id: str,
    root_dir: str,
) -> tuple[str, list[str]]:
    """
    Extract and save both ECAPA and FaceNet embeddings for one video.

    Args:
        audio_path : wav file (TTS-generated or original)
        video_path : mp4 file (lip-synced or original)
        speaker_id : e.g. 'id0001'
        language   : e.g. 'German'
        video_id   : unique clip identifier, e.g. 'h_vamljclHE'
        root_dir   : project root (FOP directory)

    Returns:
        (ecapa_path, [facenet_paths])
    """
    audio_emb = extract_ecapa(audio_path)
    ecapa_path = save_ecapa_embedding(audio_emb, root_dir, speaker_id, language, video_id)

    face_embs = extract_facenet(video_path)
    facenet_paths = save_facenet_embeddings(face_embs, root_dir, speaker_id, language, video_id)

    return ecapa_path, facenet_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ECAPA + FaceNet embeddings")
    parser.add_argument("--audio", required=True, help="Path to audio wav file")
    parser.add_argument("--video", required=True, help="Path to video mp4 file")
    parser.add_argument("--speaker_id", required=True, help="Speaker ID, e.g. id0001")
    parser.add_argument("--language", required=True, help="Language label, e.g. German")
    parser.add_argument("--video_id", required=True, help="Unique clip ID, e.g. h_vamljclHE")
    parser.add_argument("--root_dir", default="..", help="Project root (FOP parent dir)")
    args = parser.parse_args()

    ecapa, facenets = save_embeddings(
        audio_path=args.audio,
        video_path=args.video,
        speaker_id=args.speaker_id,
        language=args.language,
        video_id=args.video_id,
        root_dir=args.root_dir,
    )

    print(f"\nECAPA  → {ecapa}")
    print(f"FaceNet → {len(facenets)} frame embeddings saved")
