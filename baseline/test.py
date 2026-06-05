import argparse
import torch
import pandas as pd

from config import ExperimentConfig
from utils.featLoader import LoadData
from model import FOP
from utils.evaluator import Evaluator


def load_dataset(csv_path, config):
    return LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )


def evaluate_p3_p4_p5_p6(checkpoint_path, output_csv=None):
    config = ExperimentConfig()
    config.debug = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(config.device)

    # Official provided test sets
    seen_csv = f"./feature_tracker/{config.version}_test_German.csv"
    unseen_csv = f"./feature_tracker/{config.version}_test_English_original_extracted.csv"

    seen_dataset = load_dataset(seen_csv, config)
    unseen_dataset = load_dataset(unseen_csv, config)

    face_dim = seen_dataset.face_feats.shape[1]
    audio_dim = seen_dataset.audio_feats.shape[1]

    model = FOP(
        config=config,
        face_dim=face_dim,
        voice_dim=audio_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    evaluator = Evaluator(model, config)

    # Get tensors
    seen_face, seen_audio, seen_labels = evaluator._get_tensors(seen_dataset)
    unseen_face, unseen_audio, unseen_labels = evaluator._get_tensors(unseen_dataset)

    seen_face = seen_face.to(device)
    seen_audio = seen_audio.to(device)
    seen_labels = seen_labels.to(device)

    unseen_face = unseen_face.to(device)
    unseen_audio = unseen_audio.to(device)
    unseen_labels = unseen_labels.to(device)

    # P3: German, face + audio
    p3 = evaluator.accuracy_from_tensors(
        seen_face,
        seen_audio,
        seen_labels,
    )

    # P4: German, audio only = zero face
    p4 = evaluator.accuracy_from_tensors(
        torch.zeros_like(seen_face),
        seen_audio,
        seen_labels,
    )

    # P5: English, face + audio
    p5 = evaluator.accuracy_from_tensors(
        unseen_face,
        unseen_audio,
        unseen_labels,
    )

    # P6: English, audio only = zero face
    p6 = evaluator.accuracy_from_tensors(
        torch.zeros_like(unseen_face),
        unseen_audio,
        unseen_labels,
    )

    avg = (p3 + p4 + p5 + p6) / 4.0

    results = {
        "checkpoint": checkpoint_path,
        "P3_German_face_audio": p3,
        "P4_German_audio_only": p4,
        "P5_English_face_audio": p5,
        "P6_English_audio_only": p6,
        "Average": avg,
    }

    print("\n=== P3-P6 Evaluation ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"P3 German face+audio : {p3:.2f}")
    print(f"P4 German audio-only : {p4:.2f}")
    print(f"P5 English face+audio: {p5:.2f}")
    print(f"P6 English audio-only: {p6:.2f}")
    print(f"Average              : {avg:.2f}")

    if output_csv is not None:
        pd.DataFrame([results]).to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint",
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save results CSV",
    )

    args = parser.parse_args()

    evaluate_p3_p4_p5_p6(
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
    )