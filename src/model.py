"""
model.py — Ordinal regression model with CORN loss for Fruit Ripeness Grading.

Architecture:
  - Backbone: MobileNetV2 (default) or ResNet18, pretrained on ImageNet
  - Head 1 (CORN ordinal): Linear(feature_dim → num_classes-1)
  - Head 2 (Damage, tomato only): Linear(feature_dim → 1) with sigmoid

Usage:
    from src.model import FruitOrdinalNet
    model = FruitOrdinalNet(fruit="banana", backbone="mobilenet_v2")
"""

import torch
import torch.nn as nn
from torchvision import models


class FruitOrdinalNet(nn.Module):
    """
    Multi-task ordinal regression network for fruit ripeness grading.

    Args:
        fruit: 'banana' (4 ordinal classes) or 'tomato' (3 ordinal + damage flag)
        backbone: 'mobilenet_v2' or 'resnet18'
        num_ordinal_classes: number of ordinal classes (4 for banana, 3 for tomato)
        pretrained: use ImageNet pretrained weights
        dropout: dropout probability before heads
    """

    def __init__(
        self,
        fruit: str = "banana",
        backbone: str = "mobilenet_v2",
        num_ordinal_classes: int = None,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fruit = fruit
        self.backbone_name = backbone

        # Default num_ordinal_classes
        if num_ordinal_classes is None:
            num_ordinal_classes = 4 if fruit in ("banana", "strawberry", "avocado") else 3
        self.num_ordinal_classes = num_ordinal_classes

        # ── Build backbone ────────────────────────────────────────────────
        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
            self.feature_dim = base.classifier[1].in_features  # 1280
            # Remove the original classifier
            base.classifier = nn.Identity()
            self.backbone = base
            # For Grad-CAM: target the last conv features
            self.gradcam_target_layer = base.features[-1]

        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.feature_dim = base.fc.in_features  # 512
            base.fc = nn.Identity()
            self.backbone = base
            self.gradcam_target_layer = base.layer4[-1]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── Shared dropout ────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)

        # ── Head 1: CORN ordinal head ─────────────────────────────────────
        # CORN requires (num_classes - 1) output units
        self.corn_head = nn.Linear(self.feature_dim, num_ordinal_classes - 1)

        # ── Head 2: Damage binary head (tomato only) ──────────────────────
        if fruit == "tomato":
            self.damage_head = nn.Linear(self.feature_dim, 1)
        else:
            self.damage_head = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 224, 224) input images

        Returns:
            corn_logits: (B, num_classes-1) for CORN loss
            damage_logits: (B, 1) for BCE loss, or None if banana
        """
        features = self.backbone(x)        # (B, feature_dim)
        features = self.dropout(features)

        corn_logits = self.corn_head(features)  # (B, K-1)

        damage_logits = None
        if self.damage_head is not None:
            damage_logits = self.damage_head(features)  # (B, 1)

        return corn_logits, damage_logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features (for Grad-CAM or analysis)."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (Phase 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_params(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(fruit: str, backbone: str = "mobilenet_v2",
                pretrained: bool = True, dropout: float = 0.3) -> FruitOrdinalNet:
    """Factory function to create a FruitOrdinalNet."""
    model = FruitOrdinalNet(
        fruit=fruit,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )
    return model


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for fruit in ["banana", "tomato"]:
        print(f"\n{'='*50}")
        print(f"  {fruit.upper()} model")
        print(f"{'='*50}")
        model = build_model(fruit, backbone="mobilenet_v2")
        x = torch.randn(2, 3, 224, 224)
        corn_logits, damage_logits = model(x)
        print(f"  Input shape:        {x.shape}")
        print(f"  CORN logits shape:  {corn_logits.shape}")
        if damage_logits is not None:
            print(f"  Damage logits shape: {damage_logits.shape}")
        params = model.count_params()
        print(f"  Total params:       {params['total']:,}")
        print(f"  Trainable params:   {params['trainable']:,}")
        print(f"  Grad-CAM target:    {type(model.gradcam_target_layer).__name__}")
