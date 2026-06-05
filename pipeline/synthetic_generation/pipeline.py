"""
pipeline.py — End-to-end dubbing pipeline (Task 1)

Stages:
  1. extract_audio       : strip wav from mp4 using ffmpeg
  2. transcribe          : Whisper ASR → (text, detected_language)
  3. translate           : GoogleTranslator (deep_translator) → translated text
  4. clone_and_synthesize: edge-tts (Microsoft TTS) → translated speech audio
                           Note: Coqui XTTS v2 is the preferred voice-cloning
                           backend; switch to it by installing `TTS` and setting
                           TTS_BACKEND=coqui in the environment.
  5. lip_sync            : Wav2Lip → lip-synced output video
                           (skipped automatically if checkpoint is missing;
                            pass --skip_lipsync to always skip)

Usage:
  python pipeline.py --video path/to/video.mp4 --tgt_lang de --output_dir ./output
  python pipeline.py --video D:/data/videos/id0001/English/h_vamljclHE/00000.avi --tgt_lang de --output_dir ./output
  python pipeline.py --video path/to/video.mp4 --tgt_lang de --skip_lipsync

  python C:/Users/beiba/Desktop/baseline-polysim/Poly-sim/FOP/pipeline/pipeline.py `
  --video "D:/data/videos/id0001/English/h_vamljclHE/00000.avi" `
  --tgt_lang de `
  --output_dir "C:/Users/beiba/Desktop/baseline-polysim/Poly-sim/FOP/pipeline_output" `
  --wav2lip_dir "C:/Users/beiba/Desktop/baseline-polysim/Poly-sim/FOP/Wav2Lip" `
  --wav2lip_checkpoint "C:/Users/beiba/Desktop/baseline-polysim/Poly-sim/FOP/Wav2Lip/checkpoints/wav2lip_gan.pt"

  """

import argparse
import asyncio
import logging
import os
import subprocess
from pathlib import Path
import torch
import sys

import whisper
from deep_translator import GoogleTranslator

SCRIPT_DIR = Path(__file__).resolve().parent
FOP_ROOT = SCRIPT_DIR.parent
DEFAULT_WAV2LIP_DIR = FOP_ROOT / "Wav2Lip"
DEFAULT_WAV2LIP_CKPT = DEFAULT_WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"

# Ensure the conda env's Library/bin is on PATH so ffmpeg-python can find ffmpeg.exe
_conda_lib_bin = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "Library", "bin"))
if _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Stage 1 — Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, out_wav: str) -> str:
    """Extract mono 16 kHz wav from a video file using ffmpeg."""
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)

    # Use imageio-ffmpeg's self-contained binary — avoids DLL issues on Windows.
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    logger.info("ffmpeg binary: %s", ffmpeg_bin)

    cmd = [ffmpeg_bin, "-y", "-i", video_path,
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}):\n"
                           f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    logger.info("Audio extracted → %s", out_wav)
    return out_wav

def convert_audio_to_wav(input_audio: str, output_wav: str) -> str:
    """
    Convert generated audio to real mono 16 kHz WAV.
    This is needed because edge-tts outputs MP3 data, even if the file extension is .wav.
    """
    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", input_audio,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    if result.returncode != 0:
        raise RuntimeError(
            f"Audio conversion failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    logger.info("Converted generated audio → %s", output_wav)
    return output_wav

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

    Loads the wav via torchaudio to bypass Whisper's internal ffmpeg call,
    which avoids DLL conflicts on Windows.

    Returns:
        text          : transcribed text
        detected_lang : ISO 639-1 language code (e.g. 'en', 'de')
    """
    import numpy as np
    import soundfile as sf

    model = _get_whisper(model_size)

    # Load wav with soundfile (pure Python/C, no DLL issues on Windows).
    # extract_audio already produced a 16 kHz mono wav so no resampling needed.
    audio_np, _ = sf.read(audio_path, dtype="float32", always_2d=False)

    result = model.transcribe(audio_np)
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
# Stage 4 — TTS  (edge-tts backend; Coqui XTTS v2 is the voice-cloning ideal)
# ---------------------------------------------------------------------------
#
# edge-tts uses Microsoft Azure Neural TTS voices via the Edge browser API.
# It produces high-quality speech and supports 40+ languages.
# It does NOT clone the original speaker's voice — for true voice cloning,
# install Coqui TTS (`pip install TTS`) and set TTS_BACKEND=coqui.
#
# Voice selection: best German neural voices are Conrad (male) and Amala (female).
# Override with TTS_VOICE env var, e.g. TTS_VOICE=de-DE-AmalaNeural

# Maps ISO 639-1 codes → sensible default edge-tts voice
_tts_model = None

def _get_xtts():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        logger.info("Loading Coqui XTTS v2 model...")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _tts_model


_XTTS_LANG_MAP = {
    "en": "en",
    "de": "de",
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
    Generate translated speech using Coqui XTTS v2.
    ref_audio is the original speaker audio and is used as the voice prompt.
    """
    import torch

    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)

    xtts_lang = _XTTS_LANG_MAP.get(tgt_lang, tgt_lang)
    tts = _get_xtts()

    logger.info(
        "TTS via Coqui XTTS v2 | language=%s | ref_audio=%s | text=%s...",
        xtts_lang,
        ref_audio,
        text[:60],
    )

    tts.tts_to_file(
        text=text,
        speaker_wav=ref_audio,
        language=xtts_lang,
        file_path=out_wav,
    )

    logger.info("XTTS audio saved → %s", out_wav)
    return out_wav


# ---------------------------------------------------------------------------
# Stage 5 — Lip synchronization
# ---------------------------------------------------------------------------

def lip_sync(
    video_path: str,
    tts_audio: str,
    out_video: str,
    wav2lip_dir: str = str(DEFAULT_WAV2LIP_DIR),
    checkpoint: str = str(DEFAULT_WAV2LIP_CKPT),
) -> str:
    """
    Align tts_audio to the speaker's mouth movements in video_path using Wav2Lip.
    """
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)

    wav2lip_dir = os.path.abspath(wav2lip_dir)
    checkpoint = os.path.abspath(checkpoint)
    inference_script = os.path.join(wav2lip_dir, "inference.py")

    if not os.path.isfile(inference_script):
        raise FileNotFoundError(f"Wav2Lip inference.py not found: {inference_script}")

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Wav2Lip checkpoint not found: {checkpoint}")

    cmd = [
        sys.executable,
        inference_script,
        "--checkpoint_path", checkpoint,
        "--face", video_path,
        "--audio", tts_audio,
        "--outfile", out_video,
        "--nosmooth",
    ]

    logger.info("Running Wav2Lip: %s", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(wav2lip_dir) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(wav2lip_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Wav2Lip failed with return code {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

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
    wav2lip_dir: str = str(DEFAULT_WAV2LIP_DIR),
    wav2lip_checkpoint: str = str(DEFAULT_WAV2LIP_CKPT),
    whisper_model_size: str = "base",
    skip_lipsync: bool = False,
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
        skip_lipsync         : skip Stage 5; output_video will be None

    Returns:
        dict with keys:
            original_audio  : extracted wav from source video
            transcription   : ASR text
            detected_lang   : language detected by Whisper
            translation     : translated text
            tts_audio       : synthesized TTS wav
            output_video    : lip-synced video, or None if skip_lipsync=True
    """
    video_path_obj = Path(video_path)
    video_id = f"{video_path_obj.parent.name}_{video_path_obj.stem}"
    speaker_id = video_path_obj.parents[2].name

    work_dir = os.path.join(output_dir, speaker_id, video_id)
    os.makedirs(work_dir, exist_ok=True)

    # Stage 1 — extract audio
    original_audio = extract_audio(video_path, os.path.join(work_dir, "original.wav"))

    # Stage 2 — transcribe
    text, detected_lang = transcribe(original_audio, model_size=whisper_model_size)

    # Resolve source language: use detected if 'auto'
    resolved_src = detected_lang if src_lang == "auto" else src_lang

    # Stage 3 — translate
    translation = translate(text, src_lang=resolved_src, tgt_lang=tgt_lang)

    # Stage 4 — TTS
    tts_audio_mp3 = clone_and_synthesize(
    text=translation,
    ref_audio=original_audio,
    tgt_lang=tgt_lang,
    out_wav=os.path.join(work_dir, "tts_audio.wav"),
)

    tts_audio = convert_audio_to_wav(
        input_audio=tts_audio_mp3,
        output_wav=os.path.join(work_dir, "tts_audio_16k.wav"),
    )

    # Stage 5 — lip sync (optional)
    output_video = None
    ckpt_exists = os.path.isfile(wav2lip_checkpoint)
    if skip_lipsync:
        logger.info("Stage 5 skipped (--skip_lipsync)")
    elif not ckpt_exists:
        logger.warning(
            "Stage 5 skipped: checkpoint not found at '%s'. "
            "Download wav2lip_gan.pth and re-run without --skip_lipsync.",
            wav2lip_checkpoint,
        )
    else:
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
    parser.add_argument("--wav2lip_dir", default=str(DEFAULT_WAV2LIP_DIR), help="Path to Wav2Lip repository")
    parser.add_argument("--wav2lip_checkpoint", default=str(DEFAULT_WAV2LIP_CKPT))
    parser.add_argument("--whisper_model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--skip_lipsync", action="store_true",
                        help="Skip Wav2Lip stage (useful when checkpoint is unavailable)")
    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        tgt_lang=args.tgt_lang,
        output_dir=args.output_dir,
        src_lang=args.src_lang,
        wav2lip_dir=args.wav2lip_dir,
        wav2lip_checkpoint=args.wav2lip_checkpoint,
        whisper_model_size=args.whisper_model,
        skip_lipsync=args.skip_lipsync,
    )

    print("\n=== Pipeline complete ===")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
