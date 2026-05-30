"""
train.py
--------
Training loop for both the Baseline CNN and Transfer Learning models.

Features:
    - Configurable epochs, batch size, learning rate
    - Warm-up then optional backbone unfreeze (transfer model)
    - Cosine / Step / Plateau / OneCycle LR schedulers
    - Best-model checkpoint saving (based on val accuracy)
    - TensorBoard logging
    - Training / validation history saved as JSON

Usage (programmatic — prefer main.py for CLI):
    from src.train import train_model
    history = train_model(model, train_loader, val_loader, config)
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import (
    AverageMeter,
    Timer,
    TrainingHistory,
    build_scheduler,
    get_logger,
    save_checkpoint,
)

logger = get_logger()


# ─────────────────────────────────────────────
# Single epoch helpers
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,               # step-level schedulers (e.g. OneCycleLR)
    grad_clip: float = 0.0,
) -> Tuple[float, float]:
    """
    Runs one complete training epoch.

    Args:
        model     : Model in train() mode.
        loader    : Training DataLoader.
        criterion : Loss function.
        optimizer : Optimizer.
        device    : Target device.
        scheduler : If not None and has step() per batch (OneCycleLR), called here.
        grad_clip : If > 0, clips gradient norms.

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")

    pbar = tqdm(loader, desc="  [Train]", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Step per-batch schedulers
        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        # Metrics
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels).sum().item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(correct / batch_size, batch_size)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs one complete validation epoch (no gradient computation).

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")

    pbar = tqdm(loader, desc="  [Val]  ", leave=False, dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        correct = preds.eq(labels).sum().item()
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(correct / batch_size, batch_size)

    return loss_meter.avg, acc_meter.avg


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    model_name: str = "model",
) -> TrainingHistory:
    """
    Full training loop with:
        - Configurable epochs, LR, scheduler
        - Best-model checkpointing
        - TensorBoard logging
        - Optional backbone unfreeze at a specified epoch

    Args:
        model       : PyTorch model to train.
        train_loader: Training DataLoader.
        val_loader  : Validation DataLoader.
        config      : Dict with keys:
                        epochs          (int)
                        lr              (float)
                        weight_decay    (float)
                        scheduler       (str)
                        grad_clip       (float)
                        unfreeze_epoch  (int or None)
                        model_dir       (str)
                        log_dir         (str)
                        history_path    (str)
        device      : torch.device.
        model_name  : Prefix used for checkpoint filenames.

    Returns:
        TrainingHistory object with loss/accuracy curves.
    """
    epochs: int = config.get("epochs", 30)
    lr: float = config.get("lr", 1e-3)
    weight_decay: float = config.get("weight_decay", 5e-4)  # increased from 1e-4
    scheduler_type: str = config.get("scheduler", "cosine")
    grad_clip: float = config.get("grad_clip", 5.0)
    unfreeze_epoch: Optional[int] = config.get("unfreeze_epoch", None)
    model_dir: str = config.get("model_dir", "models/")
    log_dir: str = config.get("log_dir", "outputs/logs/")
    history_path: str = config.get(
        "history_path", f"outputs/{model_name}_history.json"
    )

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Loss — increased label smoothing to prevent overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    # Optimizer — only over trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))

    history = TrainingHistory()
    best_val_acc: float = 0.0
    timer = Timer()
    timer.start()

    logger.info(f"\n{'═'*60}")
    logger.info(f"  Training: {model_name}")
    logger.info(f"  Epochs  : {epochs}")
    logger.info(f"  LR      : {lr}")
    logger.info(f"  Device  : {device}")
    logger.info(f"{'═'*60}")

    for epoch in range(1, epochs + 1):
        epoch_str = f"Epoch [{epoch:03d}/{epochs}]"

        # ── Optional backbone unfreeze ────────────────────────────────────
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            if hasattr(model, "unfreeze_backbone"):
                model.unfreeze_backbone()
                # Rebuild optimizer to include newly unfrozen params
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr * 0.1,          # lower LR for fine-tuning
                    weight_decay=weight_decay,
                )
                logger.info(
                    f"[Epoch {epoch}] Backbone unfrozen; LR reduced to {lr * 0.1:.2e}"
                )

        # ── Train ─────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=(scheduler if isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ) else None),
            grad_clip=grad_clip,
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # ── LR scheduler step (epoch-level schedulers) ────────────────────
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────────────
        logger.info(
            f"{epoch_str}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}"
        )

        writer.add_scalars(
            "Loss",
            {"train": train_loss, "val": val_loss},
            epoch,
        )
        writer.add_scalars(
            "Accuracy",
            {"train": train_acc, "val": val_acc},
            epoch,
        )
        writer.add_scalar("LR", current_lr, epoch)

        history.update(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=current_lr,
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_acc": val_acc, "val_loss": val_loss},
            save_path=os.path.join(model_dir, f"{model_name}_last.pth"),
            is_best=is_best,
        )

    writer.close()
    logger.info(f"\n[Training complete] Best val_acc = {best_val_acc:.4f}")
    logger.info(f"[Training complete] Total time: {timer.elapsed_str()}")

    history.save(history_path)
    return history
