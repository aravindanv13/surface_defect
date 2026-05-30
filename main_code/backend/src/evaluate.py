"""
evaluate.py
-----------
Evaluation utilities for trained models:
    - Full test-set inference
    - Accuracy, Precision, Recall, F1-score (per-class + weighted)
    - Confusion matrix (saved as PNG)
    - Training curves plot (loss & accuracy)
    - Model comparison table (Baseline vs Transfer)

All outputs are saved under reports/ or outputs/.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from src.utils import get_logger, TrainingHistory

logger = get_logger()


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs model inference over all batches in loader.

    Returns:
        all_preds  : (N,) integer predicted class indices.
        all_labels : (N,) integer true class indices.
        all_probs  : (N, C) softmax probabilities.
    """
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    for images, labels in tqdm(loader, desc="  [Inference]", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.vstack(all_probs),
    )


# ─────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Computes a comprehensive dict of evaluation metrics.

    Args:
        y_true      : Ground-truth labels.
        y_pred      : Predicted labels.
        class_names : Human-readable class names.

    Returns:
        Dict with keys: accuracy, precision, recall, f1,
        per_class (dict), classification_report (str).
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    # Per-class
    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))),
        zero_division=0,
    )

    per_class = {
        class_names[i]: {
            "precision": round(float(precision_c[i]), 4),
            "recall": round(float(recall_c[i]), 4),
            "f1": round(float(f1_c[i]), 4),
            "support": int(support_c[i]),
        }
        for i in range(len(class_names))
    }

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision_w), 4),
        "recall_weighted": round(float(recall_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "per_class": per_class,
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Saves a confusion matrix heatmap as a PNG.

    Args:
        y_true      : Ground-truth labels.
        y_pred      : Predicted labels.
        class_names : Class label strings.
        save_path   : Output file path.
        title       : Figure title.
        normalize   : If True, shows row-normalized fractions instead of counts.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        cbar_label = "Fraction"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.5),
                                    max(5, len(class_names) * 1.2)))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────

def plot_training_curves(
    history: TrainingHistory,
    save_dir: str,
    model_name: str = "model",
) -> None:
    """
    Plots and saves training vs validation loss and accuracy curves.

    Args:
        history   : TrainingHistory object.
        save_dir  : Directory to save PNG files.
        model_name: Used in filenames and titles.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    data = history.data

    epochs = list(range(1, len(data.get("train_loss", [])) + 1))

    # ── Loss ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, data["train_loss"], label="Train Loss", color="#E87C4C", lw=2)
    axes[0].plot(epochs, data["val_loss"],   label="Val Loss",   color="#4C9BE8", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    # ── Accuracy ──────────────────────────────────────────────────────────
    axes[1].plot(epochs, data["train_acc"], label="Train Acc", color="#E87C4C", lw=2)
    axes[1].plot(epochs, data["val_acc"],   label="Val Acc",   color="#4C9BE8", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved → {path}")


# ─────────────────────────────────────────────
# Model comparison report
# ─────────────────────────────────────────────

def generate_comparison_report(
    results: Dict[str, Dict],
    save_path: str = "reports/model_comparison.json",
    plot_path: str = "reports/model_comparison.png",
) -> None:
    """
    Creates a JSON + bar-chart comparison of multiple models.

    Args:
        results  : Dict mapping model_name → metrics dict (from compute_metrics).
        save_path: Where to save the JSON summary.
        plot_path: Where to save the comparison bar chart.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ── JSON summary ─────────────────────────────────────────────────────
    summary = {
        name: {
            "accuracy": m["accuracy"],
            "precision": m["precision_weighted"],
            "recall": m["recall_weighted"],
            "f1": m["f1_weighted"],
        }
        for name, m in results.items()
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Comparison JSON saved → {save_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────
    model_names = list(summary.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4C9BE8", "#E87C4C", "#50C878", "#BF5FFF"]

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        values = [summary[name][m] for m in metrics]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=name,
            color=colors[i % len(colors)],
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Comparison chart saved → {plot_path}")

    # ── Console table ─────────────────────────────────────────────────────
    col_w = 16
    header = f"{'Model':<20}" + "".join(f"{m.capitalize():>{col_w}}" for m in metrics)
    print(f"\n{'─'*len(header)}")
    print("MODEL COMPARISON")
    print(f"{'─'*len(header)}")
    print(header)
    print("─" * len(header))
    for name, vals in summary.items():
        row = f"{name:<20}" + "".join(f"{vals[m]:>{col_w}.4f}" for m in metrics)
        print(row)
    print(f"{'─'*len(header)}\n")


# ─────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    model_name: str = "model",
    reports_dir: str = "reports/",
    outputs_dir: str = "outputs/",
    history: Optional[TrainingHistory] = None,
) -> Dict:
    """
    Runs the complete evaluation pipeline for one model:
        1. Inference on test set
        2. Metric computation
        3. Confusion matrix plot
        4. Training curves (if history provided)
        5. Returns metrics dict

    Args:
        model       : Trained PyTorch model.
        test_loader : Test DataLoader.
        class_names : Class label list.
        device      : Inference device.
        model_name  : Name tag for files.
        reports_dir : Where to save confusion matrix, etc.
        outputs_dir : Where to save training curve plots.
        history     : Optional TrainingHistory for curve plot.

    Returns:
        Metrics dict from compute_metrics().
    """
    logger.info(f"\nEvaluating: {model_name}")

    # 1. Inference
    y_pred, y_true, y_probs = predict(model, test_loader, device)

    # 2. Metrics
    metrics = compute_metrics(y_true, y_pred, class_names)

    # 3. Print classification report
    logger.info(f"\n{metrics['classification_report']}")
    logger.info(
        f"[{model_name}] "
        f"Accuracy={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1_weighted']:.4f}"
    )

    # 4. Save metrics JSON
    metrics_path = os.path.join(reports_dir, f"{model_name}_metrics.json")
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        import json
        json.dump(
            {k: v for k, v in metrics.items() if k != "classification_report"},
            f, indent=2,
        )
    logger.info(f"Metrics JSON saved → {metrics_path}")

    # 5. Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(reports_dir, f"{model_name}_confusion_matrix.png"),
        title=f"Confusion Matrix — {model_name}",
    )

    # 6. Training curves
    if history is not None:
        plot_training_curves(
            history,
            save_dir=outputs_dir,
            model_name=model_name,
        )

    return metrics
