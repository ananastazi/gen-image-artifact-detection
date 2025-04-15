# src/model.py
import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, use_pretrained: bool = True):
    """
    Builds a specified CNN or ViT model for transfer learning.

    Handles ResNet, EfficientNet, and ViT architectures from torchvision.

    Args:
        model_name (str): Name of the model (e.g., "resnet18", "efficientnet_b0", "vit_b_16").
        num_classes (int): Number of output classes.
        use_pretrained (bool): Whether to load pretrained weights.

    Returns:
        torch.nn.Module: The constructed model.
    """
    model = None
    print(f"Building model: {model_name} (Pretrained: {use_pretrained})")

    # --- Load the base model ---
    if model_name.startswith("resnet"):
        weights = None
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if use_pretrained else None
            model = models.resnet34(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            model = models.resnet50(weights=weights)
        else:
             raise ValueError(f"ResNet variant '{model_name}' not currently supported.")

        # --- Modify ResNet Classifier ---
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            print(f"  Replaced ResNet 'fc' layer. Input features: {num_ftrs}, Output classes: {num_classes}")
        else:
            raise TypeError(f"Could not find 'fc' layer in {model_name}.")

    elif model_name.startswith("efficientnet"):
        weights = None
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
            model = models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b1":
             weights = models.EfficientNet_B1_Weights.DEFAULT if use_pretrained else None
             model = models.efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"EfficientNet variant '{model_name}' not currently supported.")

        # --- Modify EfficientNet Classifier ---
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            if isinstance(model.classifier[-1], nn.Linear):
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                print(f"  Replaced EfficientNet classifier's last layer. Input features: {num_ftrs}, Output classes: {num_classes}")
            else:
                raise TypeError(f"EfficientNet classifier sequential doesn't end with Linear: {model_name}")
        else:
             raise TypeError(f"Could not find classifier Sequential block in {model_name}")

    elif model_name.startswith("vit"):
        weights = None
        if model_name == "vit_b_16":
             weights = models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
             model = models.vit_b_16(weights=weights)
        else:
            raise ValueError(f"ViT variant '{model_name}' not currently supported.")

        # --- Modify ViT Classifier ---
        if hasattr(model, 'heads') and hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
             num_ftrs = model.heads.head.in_features
             model.heads.head = nn.Linear(num_ftrs, num_classes)
             print(f"  Replaced ViT 'heads.head' layer. Input features: {num_ftrs}, Output classes: {num_classes}")
        elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
            print(f"  Replaced ViT 'head' layer. Input features: {num_ftrs}, Output classes: {num_classes}")
        else:
             raise TypeError(f"Could not find classifier head ('heads.head' or 'head') in {model_name}")

    else:
        raise ValueError(f"Model architecture family for '{model_name}' is not recognized.")

    return model