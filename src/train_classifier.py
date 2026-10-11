import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from tqdm import tqdm

from src.dataset import CrackDataset
from src.preprocessing import train_transform, inference_transform

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(freeze_backbone=True):
    """
    Build a MobileNetV3-Large classifier with transfer learning.
    Freezing the feature extractor enables fast, stable training on CPU.
    """
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(features, 2)
    return model.to(DEVICE)


def calculate_class_weights(train_dataset):
    """Compute inverse-frequency class weights to handle SDNET2018 class imbalance."""
    labels = train_dataset.df["label"].values
    num_neg = int(np.sum(labels == 0))
    num_pos = int(np.sum(labels == 1))
    total = len(labels)

    weight_neg = total / (2.0 * max(num_neg, 1))
    weight_pos = total / (2.0 * max(num_pos, 1))
    weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32).to(DEVICE)
    print(f"Class counts in train: Non-cracked={num_neg}, Cracked={num_pos}")
    print(f"Computed loss class weights: [Non-cracked: {weight_neg:.3f}, Cracked: {weight_pos:.3f}]")
    return weights


def train_one_epoch(model, loader, criterion, optimizer, max_batches=None):
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(total_samples, 1)


def evaluate(model, loader, criterion, max_batches=None):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(total_samples, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, bal_acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large Crack Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze pretrained feature extractor")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false", help="Fine-tune full network")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch (for quick testing)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit validation batches (for quick testing)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, freeze_backbone={args.freeze_backbone}")

    # Datasets and Loaders
    train_dataset = CrackDataset("train", train_transform)
    val_dataset = CrackDataset("validation", inference_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Model and Loss
    model = build_model(freeze_backbone=args.freeze_backbone)
    class_weights = calculate_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "classifier.pth"

    best_bal_acc = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, max_batches=args.max_train_batches
        )
        val_loss, val_acc, val_bal_acc, val_f1 = evaluate(
            model, val_loader, criterion, max_batches=args.max_val_batches
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Bal Acc: {val_bal_acc * 100:.2f}% | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best balanced accuracy ({best_bal_acc * 100:.2f}%). Checkpoint saved to {checkpoint_path}")

    print("\nTraining complete.")
    print(f"Best validation balanced accuracy: {best_bal_acc * 100:.2f}%")
    print(f"Saved model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

