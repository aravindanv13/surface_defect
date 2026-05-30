"""
utils.py
--------
Shared utility functions used across the project:
    - Device detection
    - Checkpoint save/load
    - Metric tracking (AverageMeter)
    - Learning-rate scheduler factory
    - Reproducibility seed
    - Logging helper
"""

import os
import random
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str = "defect_detector", level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

def get_device(use_gpu: bool = True) -> torch.device:
    """
    Returns the best available device.

    Args:
        use_gpu: If False, forces CPU even when CUDA/MPS is available.

    Returns:
        torch.device
    """
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():        # Apple Silicon
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Sets random seed for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


# ─────────────────────────────────────────────
# AverageMeter — running mean tracker
# ─────────────────────────────────────────────

class AverageMeter:
    """
    Tracks the running mean and latest value of a scalar metric.

    Usage:
        meter = AverageMeter("loss")
        meter.update(loss.item(), n=batch_size)
        print(meter.avg)
    """

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Args:
            val: Scalar value (e.g. loss for a single batch).
            n  : Number of samples this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ─────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    save_path: str,
    is_best: bool = False,
) -> None:
    """
    Saves a training checkpoint.

    Args:
        model    : PyTorch model.
        optimizer: Optimizer (state dict included).
        epoch    : Current epoch number.
        metrics  : Dict of metric values to store (e.g. val_acc).
        save_path: Where to write the .pth file.
        is_best  : If True, also writes a separate best_model.pth copy.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved → {save_path}")

    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Best model updated → {best_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, int, Dict[str, Any]]:
    """
    Loads a model checkpoint.

    Args:
        model          : Model instance (architecture must match checkpoint).
        checkpoint_path: Path to the .pth checkpoint file.
        optimizer      : If provided, its state dict is restored too.
        device         : Device to map tensors to.

    Returns:
        (model, epoch, metrics)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    logger.info(
        f"Checkpoint loaded ← {checkpoint_path}  "
        f"(epoch {epoch}, val_acc={metrics.get('val_acc', 'N/A')})"
    )
    return model, epoch, metrics


# ─────────────────────────────────────────────
# Learning-rate scheduler factory
# ─────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 30,
    steps_per_epoch: Optional[int] = None,
    warmup_epochs: int = 5,
):
    """
    Returns a learning-rate scheduler.

    Supported types:
        "cosine"       : CosineAnnealingLR
        "step"         : StepLR (drops by 0.1 every 10 epochs)
        "plateau"      : ReduceLROnPlateau (based on val metric)
        "onecycle"     : OneCycleLR (requires steps_per_epoch)

    Args:
        optimizer      : PyTorch optimizer.
        scheduler_type : One of the above strings.
        epochs         : Total training epochs.
        steps_per_epoch: Required for "onecycle".
        warmup_epochs  : Warm-up steps (unused by most schedulers here).

    Returns:
        PyTorch LR scheduler.
    """
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        max_lr = optimizer.param_groups[0]["lr"] * 10
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    return scheduler


# ─────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────

class TrainingHistory:
    """
    Accumulates epoch-level metrics for later plotting.

    Usage:
        hist = TrainingHistory()
        hist.update(train_loss=0.4, val_loss=0.5, train_acc=0.8, val_acc=0.78)
        hist.save("outputs/history.json")
    """

    def __init__(self):
        self.data: Dict[str, list] = {}

    def update(self, **kwargs: float) -> None:
        """Appends scalar values to their respective lists."""
        for key, val in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(float(val))

    def save(self, path: str) -> None:
        """Saves history to a JSON file."""
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"Training history saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TrainingHistory":
        """Loads history from a JSON file."""
        import json
        hist = cls()
        with open(path) as f:
            hist.data = json.load(f)
        return hist


# ─────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────

class Timer:
    """Simple context-manager / manual timer."""

    def __init__(self):
        self._start: float = 0.0

    def start(self) -> None:
        self._start = time.time()

    def elapsed(self) -> float:
        """Returns seconds since start()."""
        return time.time() - self._start

    def elapsed_str(self) -> str:
        """Returns elapsed time as a human-readable string."""
        secs = self.elapsed()
        mins, secs = divmod(int(secs), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        logger.info(f"Elapsed: {self.elapsed_str()}")


# ─────────────────────────────────────────────
# Label smoothing loss (optional drop-in)
# ─────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing regularization.

    Args:
        smoothing: Smoothing factor ε ∈ [0, 1). 0 = standard CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()
