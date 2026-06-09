import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reliability_fusion.config import (
    AUDIO_DIM,
    FACE_DIM,
    HIDDEN_DIM,
    DROPOUT,
    BATCH_SIZE,
    FOP_ROOT,
    TEST_CSV,
    CHECKPOINT_DIR,
    RESULTS_DIR,
)

from reliability_fusion.model_reliability import ReliabilityFusionNet
from reliability_fusion.utils.featLoader import PolySimFeatureDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_setting_name(train_language: str, test_language: str, audio_only: bool) -> str:
    same_language = train_language == test_language

    if same_language and not audio_only:
        return "P3"
    if same_language and audio_only:
        return "P4"
    if (not same_language) and not audio_only:
        return "P5"
    if (not same_language) and audio_only:
        return "P6"

    raise ValueError("Invalid setting combination.")


def load_checkpoint(checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, audio_only: bool):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    audio_rel_values = []
    face_rel_values = []

    all_preds = []
    all_labels = []

    for batch in dataloader:
        audio = batch["audio"].float().to(device)
        face = batch["face"].float().to(device)
        labels = batch["label"].long().to(device)

        logits, diagnostics = model(
            audio,
            face,
            force_audio_only=audio_only,
            return_reliability=True,
        )

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        audio_rel_values.append(diagnostics["s_audio_mean"].detach().cpu())
        face_rel_values.append(diagnostics["s_face_mean"].detach().cpu())

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    audio_rel = torch.cat(audio_rel_values).mean().item()
    face_rel = torch.cat(face_rel_values).mean().item()

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "audio_reliability_mean": audio_rel,
        "face_reliability_mean": face_rel,
        "num_samples": total_samples,
        "preds": preds,
        "labels": labels,
    }


def append_result_csv(result_row: dict, output_csv: Path):
    output_csv.parent.mkdir(exist_ok=True)

    file_exists = output_csv.exists()

    fieldnames = [
        "train_language",
        "test_language",
        "setting",
        "modality",
        "accuracy",
        "loss",
        "audio_reliability_mean",
        "face_reliability_mean",
        "num_samples",
        "checkpoint",
    ]

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({k: result_row[k] for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_language",
        type=str,
        required=True,
        choices=["English", "German"],
        help="Language of the trained checkpoint.",
    )

    parser.add_argument(
        "--test_language",
        type=str,
        required=True,
        choices=["English", "German"],
        help="Language split used for evaluation.",
    )

    parser.add_argument(
        "--audio_only",
        action="store_true",
        help="Evaluate with missing face modality. Used for P4/P6.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(RESULTS_DIR / "reliability_fusion_results.csv"),
    )

    args = parser.parse_args()

    device = get_device()

    train_language = args.train_language
    test_language = args.test_language
    audio_only = args.audio_only

    setting = get_setting_name(
        train_language=train_language,
        test_language=test_language,
        audio_only=audio_only,
    )

    modality = "audio_only" if audio_only else "audio_face"

    if args.checkpoint is None:
        checkpoint_path = CHECKPOINT_DIR / f"reliability_fusion_stageA_h512_fd02_lam02_{train_language}_best.pt"
    else:
        checkpoint_path = Path(args.checkpoint)

    checkpoint = load_checkpoint(checkpoint_path, device)

    num_classes = int(checkpoint["num_classes"])

    model = ReliabilityFusionNet(
        audio_dim=AUDIO_DIM,
        face_dim=FACE_DIM,
        num_classes=num_classes,
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        dropout=checkpoint.get("dropout", DROPOUT),
        face_dropout_p=checkpoint.get("face_dropout_p", 0.0),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = PolySimFeatureDataset(
        csv_path=TEST_CSV[test_language],
        fop_root=FOP_ROOT,
        audio_only=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()

    result = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        audio_only=audio_only,
    )

    print("=" * 80)
    print(f"Train language: {train_language}")
    print(f"Test language: {test_language}")
    print(f"Setting: {setting}")
    print(f"Modality: {modality}")
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 80)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Mean audio reliability: {result['audio_reliability_mean']:.4f}")
    print(f"Mean face reliability: {result['face_reliability_mean']:.4f}")
    print(f"Num samples: {result['num_samples']}")
    print("=" * 80)

    result_row = {
        "train_language": train_language,
        "test_language": test_language,
        "setting": setting,
        "modality": modality,
        "accuracy": result["accuracy"],
        "loss": result["loss"],
        "audio_reliability_mean": result["audio_reliability_mean"],
        "face_reliability_mean": result["face_reliability_mean"],
        "num_samples": result["num_samples"],
        "checkpoint": str(checkpoint_path),
    }

    append_result_csv(result_row, Path(args.output_csv))

    print(f"Result appended to: {args.output_csv}")


if __name__ == "__main__":
    main()