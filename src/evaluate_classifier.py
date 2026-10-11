
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score
)
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
import torch.nn as nn

import argparse
from tqdm import tqdm

from src.dataset import CrackDataset
from src.preprocessing import inference_transform

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV3-Large Classifier on held-out test data")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches to evaluate")
    args = parser.parse_args()

    model_path = ROOT / "models" / "classifier.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            "Classifier checkpoint not found. Run "
            "python -m src.train_classifier first."
        )

    dataset = CrackDataset("test", inference_transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = mobilenet_v3_large(weights=None)
    features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(features, 2)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    actual, predicted = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating on test set")
        for batch_idx, (images, labels) in enumerate(pbar):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()

            actual.extend(labels.tolist())
            predicted.extend(predictions)

    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Non-cracked", "Cracked"],
        zero_division=0
    ))

    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(actual, predicted), 4)
    )
    print(
        "Macro F1:",
        round(f1_score(actual, predicted, average="macro", zero_division=0), 4)
    )

    output_dir = ROOT / "results" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-cracked", "Cracked"]
    )
    display.plot()
    plt.title("Crack Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("Confusion matrix saved to results/classification/")


if __name__ == "__main__":
    main()
