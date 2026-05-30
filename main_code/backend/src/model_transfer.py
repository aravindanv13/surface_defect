"""
model_transfer.py
-----------------
Transfer learning models for surface defect detection.

Supported backbones:
    - ResNet18  (default)
    - ResNet50
    - VGG16

Strategy:
    1. Load ImageNet-pretrained weights.
    2. Freeze all backbone parameters.
    3. Replace the final classifier layer with a custom head.
    4. (Optional) Unfreeze the backbone for fine-tuning after warmup.

Input  : (B, 3, 224, 224)
Output : (B, num_classes) — raw logits
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
)




class ClassifierHead(nn.Module):
    

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_units: int = 256,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────
# Transfer learning wrapper
# ─────────────────────────────────────────────

BackboneName = Literal["resnet18", "resnet50", "vgg16"]


class TransferModel(nn.Module):
    """
    Generic transfer-learning model wrapping torchvision pretrained
    backbones with a custom classifier head.

    Args:
        backbone_name  : One of "resnet18", "resnet50", "vgg16".
        num_classes    : Number of defect categories.
        freeze_backbone: If True, backbone weights are frozen initially.
        hidden_units   : Units in the intermediate FC layer of the head.
        dropout_p      : Dropout probability in the head.
    """

    def __init__(
        self,
        backbone_name: BackboneName = "resnet18",
        num_classes: int = 4,
        freeze_backbone: bool = True,
        hidden_units: int = 256,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # ── Load pretrained backbone ─────────────────────────────────────
        backbone, in_features = self._build_backbone(backbone_name)
        self.backbone = backbone

        # ── Freeze / unfreeze ────────────────────────────────────────────
        if freeze_backbone:
            self.freeze_backbone()

        # ── Custom classifier head ───────────────────────────────────────
        self.head = ClassifierHead(
            in_features=in_features,
            num_classes=num_classes,
            hidden_units=hidden_units,
            dropout_p=dropout_p,
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_backbone(name: BackboneName) -> Tuple[nn.Module, int]:
        """
        Constructs a pretrained backbone with its final classification
        layer removed, returning (backbone, feature_dim).
        """
        if name == "resnet18":
            base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            in_features = base.fc.in_features
            base.fc = nn.Identity()          # remove original classifier

        elif name == "resnet50":
            base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            in_features = base.fc.in_features
            base.fc = nn.Identity()

        elif name == "vgg16":
            base = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            in_features = base.classifier[6].in_features  # 4096
            # Replace the final linear; keep AdaptiveAvgPool inside features
            base.classifier[6] = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported backbone '{name}'. "
                "Choose from: resnet18, resnet50, vgg16."
            )

        return base, in_features

    # ── Freeze / unfreeze API ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freezes all parameters in the backbone (backbone only, not head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[TransferModel] Backbone '{self.backbone_name}' frozen.")

    def unfreeze_backbone(self, layers_to_unfreeze: Optional[int] = None) -> None:
        """
        Unfreezes backbone parameters for fine-tuning.

        Args:
            layers_to_unfreeze: If None, unfreeze all. If an int N,
                                unfreeze the last N named parameter groups.
        """
        params = list(self.backbone.named_parameters())

        if layers_to_unfreeze is None:
            for _, param in params:
                param.requires_grad = True
            print(f"[TransferModel] All backbone layers unfrozen.")
        else:
            # Unfreeze from the end
            for _, param in params[-layers_to_unfreeze:]:
                param.requires_grad = True
            n_frozen = len(params) - layers_to_unfreeze
            print(
                f"[TransferModel] Last {layers_to_unfreeze} param groups unfrozen; "
                f"{n_frozen} remain frozen."
            )

    def get_trainable_params(self):
        """Returns iterator over trainable parameters (for optimizer)."""
        return filter(lambda p: p.requires_grad, self.parameters())

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Logits tensor (B, num_classes).
        """
        features = self.backbone(x)       # (B, in_features)
        logits = self.head(features)      # (B, num_classes)
        return logits

    # ── Info helpers ─────────────────────────────────────────────────────

    def parameter_summary(self) -> None:
        """Prints a summary of total vs trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n[TransferModel] Parameter summary:")
        print(f"  Backbone       : {self.backbone_name}")
        print(f"  Total params   : {total:,}")
        print(f"  Trainable      : {trainable:,}")
        print(f"  Frozen         : {frozen:,}")


# ─────────────────────────────────────────────
# Convenience factories
# ─────────────────────────────────────────────

def build_resnet18(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    dropout_p: float = 0.5,
) -> TransferModel:
    """Returns a ResNet18-based transfer learning model."""
    return TransferModel(
        backbone_name="resnet18",
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout_p=dropout_p,
    )


def build_resnet50(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    dropout_p: float = 0.5,
) -> TransferModel:
    """Returns a ResNet50-based transfer learning model."""
    return TransferModel(
        backbone_name="resnet50",
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout_p=dropout_p,
    )


def build_vgg16(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    dropout_p: float = 0.5,
) -> TransferModel:
    """Returns a VGG16-based transfer learning model."""
    return TransferModel(
        backbone_name="vgg16",
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout_p=dropout_p,
    )


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    num_classes = 4
    dummy_input = torch.randn(4, 3, 224, 224)

    for backbone in ["resnet18", "resnet50", "vgg16"]:
        print(f"\n{'─'*50}")
        print(f"Testing backbone: {backbone}")
        model = TransferModel(
            backbone_name=backbone,
            num_classes=num_classes,
            freeze_backbone=True,
        )
        model.parameter_summary()

        output = model(dummy_input)
        assert output.shape == (4, num_classes), \
            f"Shape mismatch: {output.shape}"
        print(f"  Output shape: {output.shape}  ✓")

    print("\nAll transfer model smoke tests passed ✓")
