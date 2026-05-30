"""
gradcam.py
----------
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

Supports:
    - BaselineCNN  : hooks the last ConvStage of `model.features`
    - TransferModel (ResNet18/50): hooks `model.backbone.layer4`
    - TransferModel (VGG16)      : hooks `model.backbone.features[-1]`

Usage:
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(image_tensor, class_idx)   # (H, W) numpy
    overlay = cam.overlay(original_pil_image, heatmap)
    overlay.save("outputs/gradcam/example.png")

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization," ICCV 2017.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.data_loader import get_val_transforms
from src.utils import get_logger

logger = get_logger()


# ─────────────────────────────────────────────
# Grad-CAM core class
# ─────────────────────────────────────────────

class GradCAM:
    """
    Implements Grad-CAM for a given model and target layer.

    Registers forward and backward hooks to capture:
        - Activations  : feature maps at the target layer (forward pass).
        - Gradients    : gradients flowing back to the target layer.

    Then computes the weighted activation map (CAM).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model       : Trained PyTorch model (must be in eval mode).
            target_layer: The specific layer to hook (e.g. model.features[-1]).
        """
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ── Hooks ──────────────────────────────────────────────────────────

    def _save_activations(self, module, input, output) -> None:
        """Forward hook: captures the layer's output tensor."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        """Backward hook: captures gradients w.r.t. the layer's output."""
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Cleans up registered hooks (call when done)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── CAM generation ─────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_idx   : Target class index. If None, uses argmax (predicted).
            device      : Device to run inference on.

        Returns:
            Normalised heatmap as a numpy array of shape (H, W), values in [0, 1].
        """
        if device is None:
            device = next(self.model.parameters()).device

        input_tensor = input_tensor.to(device)
        self.model.eval()

        # ── Forward pass ──────────────────────────────────────────────
        logits = self.model(input_tensor)          # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # ── Backward pass ─────────────────────────────────────────────
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # ── Compute CAM ───────────────────────────────────────────────
        # Gradients: (1, C, H, W)  → global average pool → (C,)
        weights = self._gradients.mean(dim=[2, 3])[0]    # (C,)

        # Activations: (1, C, H, W) → (C, H, W)
        activations = self._activations[0]               # (C, H, W)

        # Weighted sum over channels: (H, W)
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)                                # ReLU removes negatives

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    # ── Overlay helpers ─────────────────────────────────────────────────

    @staticmethod
    def overlay(
        image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays the heatmap on the original image using OpenCV.

        Args:
            image    : Original PIL image or (H, W, 3) numpy array.
            heatmap  : Normalised heatmap (H, W) in [0, 1].
            alpha    : Blend weight for heatmap (0 = image only, 1 = heatmap only).
            colormap : OpenCV colormap constant.

        Returns:
            Blended image as (H, W, 3) uint8 numpy array.
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image.copy()

        # Resize heatmap to match image
        h, w = image_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), colormap
        )
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        blended = (
            (1 - alpha) * image_np.astype(float)
            + alpha * heatmap_rgb.astype(float)
        ).clip(0, 255).astype(np.uint8)

        return blended


# ─────────────────────────────────────────────
# Target-layer resolution
# ─────────────────────────────────────────────

def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Resolves the appropriate target layer for Grad-CAM based on model type.

    Supported:
        BaselineCNN    → model.features[-1] (last ConvStage)
        TransferModel  → depends on backbone_name:
            resnet18/50 → model.backbone.layer4
            vgg16       → model.backbone.features[-1]

    Args:
        model: Trained model instance.

    Returns:
        The target nn.Module layer.

    Raises:
        ValueError if the model architecture is unrecognised.
    """
    # ── BaselineCNN ───────────────────────────────────────────────────
    model_class = type(model).__name__
    if model_class == "BaselineCNN":
        return model.features[-1]   # last ConvStage (contains conv layers)

    # ── TransferModel ─────────────────────────────────────────────────
    if model_class == "TransferModel":
        backbone_name = model.backbone_name
        if backbone_name in ("resnet18", "resnet50"):
            return model.backbone.layer4
        elif backbone_name == "vgg16":
            # Last max-pool before classifier; use last conv block
            return model.backbone.features[-1]
        else:
            raise ValueError(f"Unknown backbone for Grad-CAM: {backbone_name}")

    raise ValueError(
        f"Unrecognised model class: {model_class}. "
        "Please manually specify target_layer."
    )


# ─────────────────────────────────────────────
# Batch visualization
# ─────────────────────────────────────────────

def visualize_gradcam_batch(
    model: nn.Module,
    image_paths: List[str],
    class_names: List[str],
    device: torch.device,
    output_dir: str = "outputs/gradcam/",
    img_size: int = 224,
    class_idx: Optional[int] = None,
) -> None:
    """
    Generates and saves Grad-CAM overlays for a list of images.

    Args:
        model       : Trained model (eval mode).
        image_paths : List of image file paths.
        class_names : List of class name strings.
        device      : Inference device.
        output_dir  : Directory to save heatmap PNGs.
        img_size    : Input image size.
        class_idx   : If None, uses predicted class per image.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    target_layer = get_target_layer(model)
    cam = GradCAM(model, target_layer)
    transform = get_val_transforms(img_size)

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Load original + preprocessed tensor
        original = Image.open(img_path).convert("RGB")
        tensor = transform(original).unsqueeze(0)   # (1, 3, H, W)

        # Generate CAM
        heatmap, predicted_idx = cam.generate(tensor, class_idx=class_idx, device=device)
        predicted_class = class_names[predicted_idx]

        # Create overlay
        overlay = GradCAM.overlay(original, heatmap)

        # ── Build 3-panel figure ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"Grad-CAM  |  File: {img_path.name}  |  Predicted: {predicted_class}",
            fontsize=11, fontweight="bold",
        )

        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Heatmap with colorbar
        im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        # Save
        out_name = f"gradcam_{img_path.stem}.png"
        out_path = Path(output_dir) / out_name
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Grad-CAM saved → {out_path}  (pred: {predicted_class})")

    cam.remove_hooks()
    logger.info(f"[Grad-CAM] All overlays saved to {output_dir}")


# ─────────────────────────────────────────────
# Single-image convenience function
# ─────────────────────────────────────────────

def run_gradcam(
    model: nn.Module,
    image_path: str,
    class_names: List[str],
    device: torch.device,
    output_dir: str = "outputs/gradcam/",
    img_size: int = 224,
) -> str:
    """
    Convenience wrapper for a single image.

    Returns:
        Path to the saved overlay image.
    """
    visualize_gradcam_batch(
        model=model,
        image_paths=[image_path],
        class_names=class_names,
        device=device,
        output_dir=output_dir,
        img_size=img_size,
    )
    stem = Path(image_path).stem
    return str(Path(output_dir) / f"gradcam_{stem}.png")
