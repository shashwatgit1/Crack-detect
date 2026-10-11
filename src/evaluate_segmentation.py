import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train_segmentation import (
    SegmentationDataset,
    UNet,
    DEVICE
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net crack segmentation on held-out test data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "all"], help="Split to evaluate")
    args = parser.parse_args()

    checkpoint = ROOT / "models" / "segmentation.pth"

    if not checkpoint.exists():
        raise FileNotFoundError(
            "Segmentation model not found. Run "
            "python -m src.train_segmentation first."
        )

    dataset = SegmentationDataset(split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    intersection = 0
    union = 0
    predicted_pixels = 0
    actual_pixels = 0

    saved_samples = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            logits = model(images.to(DEVICE))
            probs = torch.sigmoid(logits)
            preds_bin = probs > 0.5
            targets_bin = masks.to(DEVICE) > 0.5

            inter = (preds_bin & targets_bin).sum().item()
            u = (preds_bin | targets_bin).sum().item()
            pred_count = preds_bin.sum().item()
            target_count = targets_bin.sum().item()

            intersection += inter
            union += u
            predicted_pixels += pred_count
            actual_pixels += target_count

            if len(saved_samples) < 3:
                # Denormalize image for qualitative plotting
                img_np = images[0].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_denorm = np.clip(img_np * std + mean, 0, 1)

                mask_gt = targets_bin[0, 0].cpu().numpy().astype(np.uint8) * 255
                mask_pred = preds_bin[0, 0].cpu().numpy().astype(np.uint8) * 255

                saved_samples.append((img_denorm, mask_gt, mask_pred))

    dice = (2.0 * intersection) / max(predicted_pixels + actual_pixels, 1)
    iou = intersection / max(union, 1)

    print(f"\n--- Segmentation Evaluation ({args.split.upper()} set, {len(dataset)} samples) ---")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"Mean IoU:         {iou:.4f}")

    # Save qualitative visual comparison
    if saved_samples:
        results_dir = ROOT / "results" / "segmentation"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / "qualitative_evaluation.png"

        fig, axes = plt.subplots(len(saved_samples), 3, figsize=(10, 3 * len(saved_samples)))
        if len(saved_samples) == 1:
            axes = np.expand_dims(axes, 0)

        for i, (orig, gt, pred) in enumerate(saved_samples):
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title("Input Concrete Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt, cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("Predicted Crack Mask")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved qualitative overlay comparison to: {save_path}")


if __name__ == "__main__":
    main()
