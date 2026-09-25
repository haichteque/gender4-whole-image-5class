"""Shared MobileNetV2 5-class whole-image classifier.

Input: float32 NCHW in [0, 1] (RGB / 255). ImageNet mean/std is applied inside
the graph so callers only need a simple /255 preprocess.
Output (export / SoftmaxWrapper): softmax probabilities over LABELS.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 256
NUM_CLASSES = 5
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"

with open(LABELS_PATH, encoding="utf-8") as f:
    LABELS: list[str] = json.load(f)

assert len(LABELS) == NUM_CLASSES, f"expected {NUM_CLASSES} labels, got {LABELS}"


class Gender4(nn.Module):
    """MobileNetV2 backbone + Linear(1280, 5). Forward returns logits."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.base = models.mobilenet_v2(weights=weights)
        self.base.classifier = nn.Linear(1280, NUM_CLASSES)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base((x - self.mean) / self.std)


class SoftmaxWrapper(nn.Module):
    """Export wrapper: logits → softmax probabilities named `output`."""

    def __init__(self, backbone: Gender4):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.backbone(x), dim=1)


def build_model(pretrained: bool = True) -> Gender4:
    return Gender4(pretrained=pretrained)


def load_checkpoint(path: str | Path, map_location="cpu") -> Gender4:
    model = build_model(pretrained=False)
    state = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model
