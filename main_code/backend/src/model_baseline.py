"""
model_baseline.py
-----------------
Custom CNN baseline model for surface defect detection.

Architecture:
    Block 1 : Conv(3→32)  → BN → ReLU → Conv(32→32)  → BN → ReLU → MaxPool → Dropout
    Block 2 : Conv(32→64) → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool → Dropout
    Block 3 : Conv(64→128)→ BN → ReLU → Conv(128→128)→ BN → ReLU → MaxPool → Dropout
    Head    : AdaptiveAvgPool → Flatten → FC(512) → BN → ReLU → Dropout → FC(num_classes)

Input  : (B, 3, 224, 224)
Output : (B, num_classes)  — raw logits
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Building block: Conv → BN → ReLU
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Fused Conv2d + BatchNorm2d + ReLU building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,           # BN absorbs the bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────
# Convolutional stage (double-conv + pool)
# ─────────────────────────────────────────────

class ConvStage(nn.Module):
    """
    Two ConvBNReLU layers followed by 2×2 MaxPool and spatial Dropout.
    Doubles the number of feature-map channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.stage = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


# ─────────────────────────────────────────────
# Baseline CNN
# ─────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """
    Custom 3-stage CNN for multi-class surface defect classification.

    Args:
        num_classes    : Number of output classes.
        dropout_conv   : Dropout probability in conv blocks.
        dropout_fc     : Dropout probability in the classifier head.
        hidden_fc_units: Number of units in the intermediate FC layer.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_conv: float = 0.2,
        dropout_fc: float = 0.5,
        hidden_fc_units: int = 512,
    ):
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────
        self.features = nn.Sequential(
            ConvStage(3,   32,  dropout_p=dropout_conv),   # 224 → 112
            ConvStage(32,  64,  dropout_p=dropout_conv),   # 112 → 56
            ConvStage(64,  128, dropout_p=dropout_conv),   # 56  → 28
        )

        # Global average pool → 128-d vector regardless of input spatial size
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))    # 28 → 4×4 = 16 per channel

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_fc_units),
            nn.BatchNorm1d(hidden_fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(hidden_fc_units, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for Conv layers, Xavier for Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Logits tensor (B, num_classes).
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the raw feature maps after the last conv stage
        (before global pool). Used by Grad-CAM.
        """
        return self.features(x)


# ─────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────

def build_baseline_cnn(
    num_classes: int = 4,
    dropout_conv: float = 0.2,
    dropout_fc: float = 0.5,
) -> BaselineCNN:
    """
    Factory function that returns a freshly initialised BaselineCNN.

    Args:
        num_classes : Number of output classes.
        dropout_conv: Spatial dropout after each conv stage.
        dropout_fc  : Dropout before the final linear layer.

    Returns:
        BaselineCNN instance.
    """
    model = BaselineCNN(
        num_classes=num_classes,
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc,
    )
    return model


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    num_classes = 4
    model = build_baseline_cnn(num_classes=num_classes)

    # Print architecture
    print(model)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Forward pass smoke test
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput  shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, num_classes), "Output shape mismatch!"
    print("Baseline CNN smoke test passed ✓")
