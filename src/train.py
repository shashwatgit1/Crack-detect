import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loaders
from model import create_model


NUM_EPOCHS = 10
LEARNING_RATE = 0.001


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, val_loader, test_loader, classes = get_data_loaders()

    print("Classes:", classes)

    model = create_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    os.makedirs("models", exist_ok=True)

    for epoch in range(NUM_EPOCHS):

        # -------------------------
        # Training
        # -------------------------
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        # -------------------------
        # Validation
        # -------------------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes
                },
                "models/crack_classifier.pth"
            )

            print("Best model saved.")

    print("\nTraining completed.")
    print("Best validation accuracy:", best_val_accuracy)


if __name__ == "__main__":
    train()