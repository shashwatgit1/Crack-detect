import sys
import torch
from PIL import Image
from torchvision import transforms

from model import create_model


IMAGE_SIZE = 224


def predict(image_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "models/crack_classifier.pth",
        map_location=device
    )

    classes = checkpoint["classes"]

    model = create_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, 1
        )

    predicted_class = classes[predicted.item()]

    print("\nPrediction:", predicted_class)

    print(
        "Confidence:",
        f"{confidence.item() * 100:.2f}%"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python predict.py <image_path>"
        )

    else:

        predict(sys.argv[1])