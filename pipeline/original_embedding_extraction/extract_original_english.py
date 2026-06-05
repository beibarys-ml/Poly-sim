import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ecapa_model = None
_mtcnn = None
_facenet = None


def get_ecapa():
    global _ecapa_model
    if _ecapa_model is None:
        print("[INFO] Loading SpeechBrain ECAPA-TDNN...")
        _ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE},
        )
    return _ecapa_model


def extract_audio_from_video(video_path: str, out_wav: str) -> str:
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)

    import imageio_ffmpeg
    import subprocess

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        out_wav,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return out_wav


def extract_ecapa(audio_path: str) -> np.ndarray:
    model = get_ecapa()

    audio_np, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)

    if sample_rate != 16000:
        import math
        gcd = math.gcd(16000, sample_rate)
        audio_np = resample_poly(
            audio_np,
            16000 // gcd,
            sample_rate // gcd,
        ).astype(np.float32)

    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_batch(waveform)

    return emb.squeeze().cpu().numpy()


def get_facenet():
    global _mtcnn, _facenet

    if _mtcnn is None:
        print("[INFO] Loading MTCNN...")
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            post_process=True,
            device=DEVICE,
        )

    if _facenet is None:
        print("[INFO] Loading FaceNet...")
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    return _mtcnn, _facenet


def extract_facenet(video_path: str, frame_step: int = 25):
    mtcnn, facenet = get_facenet()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    results = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(frame_rgb)

        if face_tensor is not None:
            with torch.no_grad():
                emb = facenet(face_tensor.unsqueeze(0).to(DEVICE))
            results.append((frame_idx, emb.squeeze().cpu().numpy()))

        frame_idx += 1

    cap.release()
    return results


def rel_path(path: Path, root: Path) -> str:
    return "./" + str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def save_ecapa(emb, fop_root, speaker_id, language, video_id):
    out_dir = (
        Path(fop_root)
        / "ecappafeats_original_extracted"
        / "v3"
        / "voices"
        / speaker_id
        / language
        / video_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "00000.npy"
    np.save(out_path, emb)

    return out_path


def save_facenet(face_embs, fop_root, speaker_id, language, video_id):
    out_dir = (
        Path(fop_root)
        / "facenetfeats_original_extracted"
        / "v3"
        / "faces"
        / speaker_id
        / language
        / video_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for frame_idx, emb in face_embs:
        out_path = out_dir / f"{frame_idx:09d}.npy"
        np.save(out_path, emb)
        saved.append(out_path)

    return saved


def process_video(video_path, fop_root, speaker_id, label, language, frame_step):
    video_path = Path(video_path)
    video_id = f"{video_path.parent.name}_{video_path.stem}"

    print(f"[INFO] Processing {speaker_id} / {language} / {video_id}")

    work_dir = Path(fop_root) / "pipeline_output_original" / speaker_id / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_wav = work_dir / "original.wav"

    extract_audio_from_video(str(video_path), str(audio_wav))

    audio_emb = extract_ecapa(str(audio_wav))
    ecapa_path = save_ecapa(audio_emb, fop_root, speaker_id, language, video_id)

    face_embs = extract_facenet(str(video_path), frame_step=frame_step)

    if len(face_embs) == 0:
        print(f"[WARNING] No face detected every {frame_step} frames. Retrying every 2 frames.")
        face_embs = extract_facenet(str(video_path), frame_step=2)

    if len(face_embs) == 0:
        raise RuntimeError(f"No face embeddings extracted from {video_path}")

    facenet_paths = save_facenet(face_embs, fop_root, speaker_id, language, video_id)

    rows = []
    root = Path(fop_root)

    for face_path in facenet_paths:
        rows.append([
            "",
            "",
            speaker_id,
            rel_path(face_path, root),
            int(label),
            rel_path(ecapa_path, root),
            "",
        ])

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--fop_root", required=True)
    parser.add_argument("--speaker_map", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--out_csv", default="feature_tracker/v3_train_English_original_extracted.csv")
    parser.add_argument("--frame_step", type=int, default=25)
    parser.add_argument("--max_videos_per_speaker", type=int, default=0)

    args = parser.parse_args()

    video_root = Path(args.video_dir)
    fop_root = Path(args.fop_root)

    with open(args.speaker_map, "r") as f:
        speaker_map = json.load(f)

    all_rows = []
    errors = []

    for speaker_id, label in speaker_map.items():
        source_dir = video_root / speaker_id / args.language

        if not source_dir.exists():
            print(f"[WARNING] Missing folder: {source_dir}")
            continue

        videos = sorted(
            list(source_dir.rglob("*.avi")) +
            list(source_dir.rglob("*.mp4"))
        )

        if args.max_videos_per_speaker > 0:
            videos = videos[:args.max_videos_per_speaker]

        print(f"\n[INFO] Speaker {speaker_id}: {len(videos)} videos")

        for video_path in videos:
            try:
                rows = process_video(
                    video_path=video_path,
                    fop_root=fop_root,
                    speaker_id=speaker_id,
                    label=label,
                    language=args.language,
                    frame_step=args.frame_step,
                )
                all_rows.extend(rows)

            except Exception as exc:
                print(f"[ERROR] Failed {speaker_id} / {video_path}: {exc}")
                errors.append({
                    "speaker_id": speaker_id,
                    "video": str(video_path),
                    "error": str(exc),
                })

    out_csv = fop_root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "audio_path",
        "face_path",
        "identity",
        "facenet_feats_path",
        "label",
        "ecappa_feats_path",
        "wavlmsv_feats_path",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\n[INFO] Saved CSV: {out_csv}")
    print(f"[INFO] Rows: {len(all_rows)}")
    print(f"[INFO] Errors: {len(errors)}")

    if errors:
        error_path = fop_root / "pipeline_output_original" / "original_english_extraction_errors.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)

        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)

        print(f"[INFO] Errors saved to: {error_path}")


if __name__ == "__main__":
    main()
