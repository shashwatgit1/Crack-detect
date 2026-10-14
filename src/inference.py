
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.models import mobilenet_v3_large

from src.preprocessing import inference_transform
from src.severity import estimate_severity
from src.train_segmentation import UNet

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier():
    path = ROOT / "models" / "classifier.pth"
    if not path.exists():
        raise FileNotFoundError(
            "Classifier checkpoint missing. Run the training script first."
        )

    model = mobilenet_v3_large(weights=None)
    features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(features, 2)
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    return model


def load_segmenter():
    path = ROOT / "models" / "segmentation.pth"
    if not path.exists():
        return None

    model = UNet()
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    return model


def classify_image(image, model=None):
    if model is None:
        model = load_classifier()

    image = image.convert("RGB")
    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_class = int(probabilities.argmax().item())

    return {
        "label": "Cracked" if predicted_class == 1 else "Non-cracked",
        "crack_probability": float(probabilities[1].item()),
        "confidence": float(probabilities[predicted_class].item())
    }


def segment_image(image, model=None):
    """Return a binary crack mask as a grayscale PIL image."""
    if model is None:
        model = load_segmenter()

    if model is None:
        return None

    image = image.convert("RGB")
    resized = image.resize((256, 256))
    array = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std

    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0, 0]

    mask = (probabilities.cpu().numpy() > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask)


def create_overlay(image, mask):
    """Overlay predicted crack pixels in red on the original image."""
    if mask is None:
        return image.convert("RGB")

    original = np.asarray(image.convert("RGB"))
    mask_array = np.asarray(mask.convert("L"))

    mask_array = cv2.resize(
        mask_array,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = original.copy()
    selected = mask_array > 127

    # Blend selected pixels with red.
    red = np.zeros_like(original)
    red[:, :, 0] = 255
    overlay[selected] = (
        0.5 * original[selected] + 0.5 * red[selected]
    ).astype(np.uint8)

    return Image.fromarray(overlay)


def analyze_image(image):
    """
    Run classification and, when trained weights exist, segmentation.
    """
    classifier = load_classifier()
    result = classify_image(image, classifier)

    segmenter = load_segmenter()
    mask = None

    if result["label"] == "Cracked" and segmenter is not None:
        mask = segment_image(image, segmenter)

    if mask is not None:
        result["severity"] = estimate_severity(
            np.asarray(mask.convert("L"))
        )
        result["mask"] = mask
        result["overlay"] = create_overlay(image, mask)
    elif result["label"] == "Non-cracked":
        result["severity"] = {
            "severity": "No crack detected",
            "total_crack_area_pixels": 0,
            "crack_area_pixels": 0,
            "num_cracks": 0,
            "connected_components": 0,
            "crack_area_percentage": 0.0,
            "estimated_length_pixels": 0,
            "individual_crack_areas": [],
            "note": "Surface was classified as non-cracked. No cracks detected."
        }
        result["mask"] = None
        result["overlay"] = image.convert("RGB")
    else:
        result["severity"] = {
            "severity": "Unavailable",
            "note": (
                "Train the segmentation model using annotated campus images "
                "to obtain an image-based severity estimate."
            )
        }
        result["mask"] = None
        result["overlay"] = image.convert("RGB")

    return result
