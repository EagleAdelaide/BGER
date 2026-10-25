"""Classification backbones. ResNet-50 default; others optional/extensible."""
from __future__ import annotations
import torch.nn as nn
import torchvision.models as tvm


def build_model(arch: str = "resnet50", num_classes: int = 200,
                pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


# ImageNet normalization constants (shared by train + eval transforms).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
