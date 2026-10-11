
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224

# Training transforms: small variations help the model generalize.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation, testing, and inference transforms.
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_rgb_image(image_path):
    """Load an image safely and convert it to RGB."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def prepare_image(image):
    """Prepare a PIL image for classifier inference."""
    image = image.convert("RGB")
    return inference_transform(image).unsqueeze(0)
