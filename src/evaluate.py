import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import get_data_loaders
from model import create_model


def evaluate():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, test_loader, classes = get_data_loaders()

    model = create_model()

    checkpoint = torch.load(
        "models/crack_classifier.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=classes
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            all_labels,
            all_predictions
        )
    )


if __name__ == "__main__":
    evaluate()