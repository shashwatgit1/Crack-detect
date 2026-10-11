from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Support both Dataset and dataset directory casing
IMAGE_DIR = ROOT / "Dataset" / "campus" / "images"
if not IMAGE_DIR.exists():
    IMAGE_DIR = ROOT / "dataset" / "campus" / "images"

MASK_DIR = ROOT / "Dataset" / "campus" / "annotations"
if not MASK_DIR.exists():
    MASK_DIR = ROOT / "dataset" / "campus" / "annotations"

MODEL_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 256
EPOCHS = 10
BATCH_SIZE = 4


class SegmentationDataset(Dataset):
    """Dataset for concrete crack pixel-level segmentation."""

    def __init__(self, split="train", val_ratio=0.2, random_state=42):
        all_pairs = []

        if IMAGE_DIR.exists() and MASK_DIR.exists():
            for image_path in sorted(IMAGE_DIR.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                mask_path = MASK_DIR / f"{image_path.stem}.png"
                if mask_path.exists():
                    all_pairs.append((image_path, mask_path))

        if not all_pairs:
            raise RuntimeError(
                f"No image/mask pairs found in {IMAGE_DIR} and {MASK_DIR}. "
                "Ensure images exist in campus/images/ and matching masks in campus/annotations/."
            )

        # Deterministic train/validation split
        rng = random.Random(random_state)
        indices = list(range(len(all_pairs)))
        rng.shuffle(indices)

        val_count = max(1, int(len(all_pairs) * val_ratio)) if len(all_pairs) > 3 else 0
        train_indices = indices[val_count:]
        val_indices = indices[:val_count]

        if split == "train":
            self.pairs = [all_pairs[i] for i in train_indices]
        elif split in {"val", "validation", "test"}:
            self.pairs = [all_pairs[i] for i in val_indices]
        elif split == "all":
            self.pairs = all_pairs
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        with Image.open(image_path) as im:
            image = im.convert("RGB").resize((SIZE, SIZE))

        with Image.open(mask_path) as im:
            mask = im.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32))
        image = image / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())

        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net architecture for binary crack segmentation."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


def compute_dice_and_iou(preds_bin, targets_bin):
    """Compute Dice and IoU coefficients for evaluation."""
    intersection = (preds_bin & targets_bin).sum().item()
    union = (preds_bin | targets_bin).sum().item()
    pred_sum = preds_bin.sum().item()
    target_sum = targets_bin.sum().item()

    dice = (2.0 * intersection) / max(pred_sum + target_sum, 1)
    iou = intersection / max(union, 1)
    return dice, iou


def dice_loss(logits, targets, smooth=1.0):
    """Differentiable soft Dice loss."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def evaluate(model, loader, bce_criterion):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            loss = bce_criterion(logits, masks) + dice_loss(logits, masks)
            total_loss += loss.item()

            preds_bin = torch.sigmoid(logits) > 0.5
            targets_bin = masks > 0.5
            dice, iou = compute_dice_and_iou(preds_bin, targets_bin)
            total_dice += dice
            total_iou += iou
            count += 1

    return total_loss / max(count, 1), total_dice / max(count, 1), total_iou / max(count, 1)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "segmentation.pth"

    train_dataset = SegmentationDataset(split="train")
    val_dataset = SegmentationDataset(split="val")

    print(f"Device: {DEVICE}")
    print(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = UNet().to(DEVICE)
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_dice = 0.0

    print("\n--- Training U-Net Segmentation Model ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d}", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce_criterion(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, bce_criterion)

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best model saved to {checkpoint_path} (Val Dice: {best_val_dice:.4f})")

    print(f"\nSegmentation training complete. Best Val Dice: {best_val_dice:.4f}")
    print(f"Model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
