"""
pipeline.py — End-to-end dubbing pipeline (Task 1)

Stages:
  1. extract_audio      : strip wav from mp4 using ffmpeg
  2. transcribe         : Whisper ASR → (text, detected_language)
  3. translate          : GoogleTranslator (deep_translator) → translated text
  4. clone_and_synthesize: Coqui XTTS v2 → TTS audio preserving speaker voice
  5. lip_sync           : Wav2Lip → lip-synced output video

Usage:
  python pipeline.py --video path/to/video.mp4 --tgt_lang de --output_dir ./output
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import ffmpeg
import whisper
from deep_translator import GoogleTranslator
from TTS.api import TTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Stage 1 — Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, out_wav: str) -> str:
    """Extract mono 16 kHz wav from a video file using ffmpeg."""
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(out_wav, acodec="pcm_s16le", ar=16000, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info("Audio extracted → %s", out_wav)
    return out_wav


# ---------------------------------------------------------------------------
# Stage 2 — Speech recognition + language detection
# ---------------------------------------------------------------------------

_whisper_model = None

def _get_whisper(model_size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model '%s' …", model_size)
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe(audio_path: str, model_size: str = "base") -> tuple[str, str]:
    """
    Run Whisper on audio_path.

    Returns:
        text          : transcribed text
        detected_lang : ISO 639-1 language code (e.g. 'en', 'de')
    """
    model = _get_whisper(model_size)
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    detected_lang = result.get("language", "unknown")
    logger.info("Transcribed (%s): %s", detected_lang, text[:80])
    return text, detected_lang


# ---------------------------------------------------------------------------
# Stage 3 — Translation
# ---------------------------------------------------------------------------

def translate(text: str, src_lang: str = "en", tgt_lang: str = "de") -> str:
    """
    Translate text using Google Translate via deep_translator.
    No API key required.

    Args:
        src_lang: ISO 639-1 source language code
        tgt_lang: ISO 639-1 target language code
    """
    translated = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
    logger.info("Translated (%s→%s): %s", src_lang, tgt_lang, translated[:80])
    return translated


# ---------------------------------------------------------------------------
# Stage 4 — Voice cloning + TTS
# ---------------------------------------------------------------------------

_tts_model = None

def _get_tts():
    global _tts_model
    if _tts_model is None:
        logger.info("Loading Coqui XTTS v2 model …")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts_model


# Maps ISO 639-1 codes to XTTS language strings
_XTTS_LANG_MAP = {
    "de": "de",
    "en": "en",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    "zh": "zh-cn",
    "hu": "hu",
    "ko": "ko",
    "ja": "ja",
}


def clone_and_synthesize(
    text: str,
    ref_audio: str,
    tgt_lang: str = "de",
    out_wav: str = "tts_output.wav",
) -> str:
    """
    Generate speech in tgt_lang using Coqui XTTS v2, cloning the voice
    from ref_audio (a short sample of the original speaker).

    Args:
        text      : translated text to synthesize
        ref_audio : path to reference wav for voice cloning
        tgt_lang  : ISO 639-1 target language code
        out_wav   : output wav path

    Returns:
        Path to generated wav file.
    """
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    xtts_lang = _XTTS_LANG_MAP.get(tgt_lang, tgt_lang)
    tts = _get_tts()
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_audio,
        language=xtts_lang,
        file_path=out_wav,
    )
    logger.info("TTS audio saved → %s", out_wav)
    return out_wav


# ---------------------------------------------------------------------------
# Stage 5 — Lip synchronization
# ---------------------------------------------------------------------------

def lip_sync(
    video_path: str,
    tts_audio: str,
    out_video: str,
    wav2lip_dir: str = "Wav2Lip",
    checkpoint: str = "Wav2Lip/checkpoints/wav2lip_gan.pth",
) -> str:
    """
    Align tts_audio to the speaker's mouth movements in video_path using Wav2Lip.

    Wav2Lip must be cloned separately:
        git clone https://github.com/Rudrabha/Wav2Lip

    Download pretrained checkpoint from:
        https://github.com/Rudrabha/Wav2Lip#getting-the-weights

    Args:
        video_path  : original video (face visible)
        tts_audio   : generated TTS wav
        out_video   : output video path
        wav2lip_dir : path to the cloned Wav2Lip repository
        checkpoint  : path to wav2lip_gan.pth checkpoint

    Returns:
        Path to lip-synced video.
    """
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    inference_script = os.path.join(wav2lip_dir, "inference.py")

    cmd = [
        "python", inference_script,
        "--checkpoint_path", checkpoint,
        "--face", video_path,
        "--audio", tts_audio,
        "--outfile", out_video,
        "--nosmooth",
    ]

    logger.info("Running Wav2Lip: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.stdout:
        logger.debug(result.stdout)
    if result.stderr:
        logger.debug(result.stderr)

    logger.info("Lip-synced video saved → %s", out_video)
    return out_video


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    tgt_lang: str = "de",
    output_dir: str = "./output",
    src_lang: str = "auto",
    wav2lip_dir: str = "Wav2Lip",
    wav2lip_checkpoint: str = "Wav2Lip/checkpoints/wav2lip_gan.pth",
    whisper_model_size: str = "base",
) -> dict:
    """
    Run the full dubbing pipeline on a single video.

    Args:
        video_path           : path to input video (.mp4)
        tgt_lang             : ISO 639-1 code for target language (default: 'de')
        output_dir           : root directory for all intermediate + final outputs
        src_lang             : source language ('auto' lets Whisper detect it)
        wav2lip_dir          : path to cloned Wav2Lip repository
        wav2lip_checkpoint   : path to Wav2Lip checkpoint file
        whisper_model_size   : Whisper model variant ('tiny'|'base'|'small'|'medium'|'large')

    Returns:
        dict with keys:
            original_audio  : extracted wav from source video
            transcription   : ASR text
            detected_lang   : language detected by Whisper
            translation     : translated text
            tts_audio       : voice-cloned TTS wav
            output_video    : final lip-synced video
    """
    stem = Path(video_path).stem
    work_dir = os.path.join(output_dir, stem)
    os.makedirs(work_dir, exist_ok=True)

    # Stage 1 — extract audio
    original_audio = extract_audio(video_path, os.path.join(work_dir, "original.wav"))

    # Stage 2 — transcribe
    text, detected_lang = transcribe(original_audio, model_size=whisper_model_size)

    # Resolve source language: use detected if 'auto'
    resolved_src = detected_lang if src_lang == "auto" else src_lang

    # Stage 3 — translate
    translation = translate(text, src_lang=resolved_src, tgt_lang=tgt_lang)

    # Stage 4 — voice-cloned TTS
    tts_audio = clone_and_synthesize(
        text=translation,
        ref_audio=original_audio,
        tgt_lang=tgt_lang,
        out_wav=os.path.join(work_dir, "tts_audio.wav"),
    )

    # Stage 5 — lip sync
    output_video = lip_sync(
        video_path=video_path,
        tts_audio=tts_audio,
        out_video=os.path.join(work_dir, "lipsynced.mp4"),
        wav2lip_dir=wav2lip_dir,
        checkpoint=wav2lip_checkpoint,
    )

    return {
        "original_audio": original_audio,
        "transcription": text,
        "detected_lang": detected_lang,
        "translation": translation,
        "tts_audio": tts_audio,
        "output_video": output_video,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dubbing pipeline: video → translated & lip-synced video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--tgt_lang", default="de", help="Target language ISO code (default: de)")
    parser.add_argument("--src_lang", default="auto", help="Source language ISO code (default: auto-detect)")
    parser.add_argument("--output_dir", default="./output", help="Root output directory")
    parser.add_argument("--wav2lip_dir", default="Wav2Lip", help="Path to Wav2Lip repository")
    parser.add_argument("--wav2lip_checkpoint", default="Wav2Lip/checkpoints/wav2lip_gan.pth")
    parser.add_argument("--whisper_model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        tgt_lang=args.tgt_lang,
        output_dir=args.output_dir,
        src_lang=args.src_lang,
        wav2lip_dir=args.wav2lip_dir,
        wav2lip_checkpoint=args.wav2lip_checkpoint,
        whisper_model_size=args.whisper_model,
    )

    print("\n=== Pipeline complete ===")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
