import torch
import torch.nn as nn
from torchvision import models


def create_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 2)

    return model