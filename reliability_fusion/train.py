import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reliability_fusion.config import (
    AUDIO_DIM,
    FACE_DIM,
    HIDDEN_DIM,
    DROPOUT,
    FACE_DROPOUT_P,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
    FOP_ROOT,
    TRAIN_CSV,
    VAL_CSV,
    CHECKPOINT_DIR,
)

from reliability_fusion.model_reliability import ReliabilityFusionNet
from reliability_fusion.utils.featLoader import PolySimFeatureDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_num_classes(csv_path: Path) -> int:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return int(df["label"].max()) + 1


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        audio = batch["audio"].float().to(device)
        face = batch["face"].float().to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad()

        logits = model(audio, face, force_audio_only=False)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device, audio_only: bool = False):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    audio_rel_values = []
    face_rel_values = []

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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    audio_rel = torch.cat(audio_rel_values).mean().item()
    face_rel = torch.cat(face_rel_values).mean().item()

    return avg_loss, avg_acc, audio_rel, face_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_language",
        type=str,
        required=True,
        choices=["English", "German"],
        help="Language split used for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Early stopping patience based on validation accuracy.",
)

    parser.add_argument(
    "--min_delta",
    type=float,
    default=0.0,
    help="Minimum validation accuracy improvement required to reset patience.",
)
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    train_language = args.train_language

    train_csv = TRAIN_CSV[train_language]
    val_csv = VAL_CSV[train_language]

    num_classes = infer_num_classes(train_csv)

    print("=" * 80)
    print(f"Training language: {train_language}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"FOP root: {FOP_ROOT}")
    print(f"Audio dim: {AUDIO_DIM}")
    print(f"Face dim: {FACE_DIM}")
    print(f"Num classes: {num_classes}")
    print(f"Device: {device}")
    print("=" * 80)

    train_dataset = PolySimFeatureDataset(
        csv_path=train_csv,
        fop_root=FOP_ROOT,
        audio_only=False,
    )

    val_dataset = PolySimFeatureDataset(
        csv_path=val_csv,
        fop_root=FOP_ROOT,
        audio_only=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ReliabilityFusionNet(
        audio_dim=AUDIO_DIM,
        face_dim=FACE_DIM,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        face_dropout_p=FACE_DROPOUT_P,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = CHECKPOINT_DIR / f"reliability_fusion_{train_language}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_audio_rel, val_face_rel = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            audio_only=False,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"val rel audio {val_audio_rel:.4f} | "
            f"val rel face {val_face_rel:.4f}"
        )

        improved = val_acc > best_val_acc + args.min_delta

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_language": train_language,
                    "num_classes": num_classes,
                    "audio_dim": AUDIO_DIM,
                    "face_dim": FACE_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "face_dropout_p": FACE_DROPOUT_P,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{args.patience} epochs."
            )

        if epochs_without_improvement >= args.patience:
            print("=" * 80)
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best val acc: {best_val_acc:.4f}"
            )
            print("=" * 80)
            break


    print("=" * 80)
    print(f"Finished training {train_language} model.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()