"""
src/model.py
============
ResNet-18 model builder for binary TB classification.

Why this is a separate module:
  - The model definition is used in multiple contexts:
    centralised training (notebook 07), FL client (notebook 08),
    DP training (notebook 09), and the Gradio demo (notebook 17).
  - Keeping the model in one place ensures every experiment uses
    exactly the same architecture.

Architecture Overview
---------------------
  ResNet-18 (pre-trained on ImageNet)
    │
    ├── Backbone: frozen for first N epochs (feature extraction phase)
    │            then unfrozen (fine-tuning phase)
    │
    └── Custom head:
          AdaptiveAvgPool2d → Flatten
          → Dropout(p=0.5)
          → Linear(512, 2)      ← TB negative vs TB positive
          → softmax (at inference)

This 'freeze then fine-tune' strategy is standard practice:
  - It prevents the small TB dataset from immediately overwriting
    useful ImageNet features in the backbone.
  - After a few epochs, we unfreeze and let all weights update.

Usage
-----
  from src.model import build_model, freeze_backbone, unfreeze_backbone
  model = build_model(pretrained=True, num_classes=2)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────

def build_model(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Build a ResNet-18 model modified for binary TB classification.

    Parameters
    ----------
    pretrained : bool
        If True, load ImageNet pre-trained weights.
        If False, initialise randomly (used in ablations).
    num_classes : int
        Number of output classes. Default 2 (TB negative, TB positive).
    dropout : float
        Dropout probability before the final linear layer.
        Helps prevent overfitting on the small TB dataset.

    Returns
    -------
    nn.Module
        Modified ResNet-18 ready for training.

    Example
    -------
    >>> model = build_model(pretrained=True, num_classes=2)
    >>> x = torch.randn(4, 3, 224, 224)   # batch of 4 images
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 2])
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Replace the original FC head (designed for 1000 ImageNet classes)
    # with a head for our binary classification task.
    # ResNet-18's penultimate feature dimension is 512.
    in_features = model.fc.in_features   # Should be 512

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all ResNet layers except the final classification head.

    When frozen, backbone gradients are not computed → faster training
    and less risk of overwriting useful pretrained features early on.

    Parameters
    ----------
    model : nn.Module
        ResNet-18 model from build_model().
    """
    for name, param in model.named_parameters():
        if "fc" not in name:           # fc = our custom head
            param.requires_grad = False

    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total  = sum(1 for p in model.parameters())
    print(f"Frozen {n_frozen}/{n_total} parameter tensors (backbone frozen)")


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all model parameters for full fine-tuning.

    Call this after the feature-extraction warm-up phase.

    Parameters
    ----------
    model : nn.Module
        ResNet-18 model.
    """
    for param in model.parameters():
        param.requires_grad = True

    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen all {n_trainable} parameter tensors (full fine-tuning)")


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total model parameters.

    Useful for the model card and sanity-checking.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict with keys 'trainable', 'frozen', 'total'
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    return {"trainable": trainable, "frozen": frozen, "total": total}


def get_model_device(model: nn.Module) -> str:
    """
    Return the device string ('cpu' or 'cuda') the model is on.
    """
    return next(model.parameters()).device.type
