"""
preprocess.py
-------------
Dataset preparation utilities:
  - Verify folder structure
  - Compute per-channel mean/std of the training set
  - Generate a class-distribution bar chart
  - Optional: resize/convert raw images to a staging folder

Run as a standalone script:
    python src/preprocess.py --data_dir data/ --output_dir outputs/
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────
# Folder-structure verification
# ─────────────────────────────────────────────

REQUIRED_SPLITS = ["train", "val", "test"]


def verify_dataset_structure(data_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Verifies that the dataset directory contains the required splits
    (train / val / test) and that each split has at least one class folder
    with images.

    Returns:
        A nested dict: {split: {class_name: image_count}}

    Raises:
        FileNotFoundError if a required split is missing.
        ValueError if a split contains no images.
    """
    data_path = Path(data_dir)
    summary: Dict[str, Dict[str, int]] = {}

    for split in REQUIRED_SPLITS:
        split_path = data_path / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Required split folder not found: {split_path}"
            )

        class_counts: Dict[str, int] = {}
        for class_folder in sorted(split_path.iterdir()):
            if not class_folder.is_dir():
                continue
            n_images = sum(
                1 for f in class_folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            )
            class_counts[class_folder.name] = n_images

        if not class_counts:
            raise ValueError(f"No class sub-folders found in: {split_path}")

        summary[split] = class_counts

    print("\n[Verify] Dataset structure OK")
    for split, counts in summary.items():
        total = sum(counts.values())
        print(f"  {split:6s}  -> {total} images across {len(counts)} classes")
        for cls, n in counts.items():
            print(f"            {cls:15s}: {n}")
    return summary


# ─────────────────────────────────────────────
# Per-channel mean/std computation
# ─────────────────────────────────────────────

def compute_dataset_stats(
    train_dir: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[List[float], List[float]]:
    """
    Computes per-channel mean and standard deviation of the training set
    by making a single pass with no normalization.

    Args:
        train_dir  : Path to the training split folder.
        img_size   : Image resize dimension.
        batch_size : Batch size for the computation pass.
        num_workers: DataLoader workers.

    Returns:
        (mean, std) each as a list of 3 floats (R, G, B).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),         # -> [0, 1] float32
    ])

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0

    print("\n[Stats] Computing per-channel mean and std …")
    for images, _ in tqdm(loader, desc="  scanning batches"):
        # images: (B, C, H, W)
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels += batch_pixels

    mean = (channel_sum / n_pixels).tolist()
    std = ((channel_sq_sum / n_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()

    print(f"  Mean : {[round(v, 4) for v in mean]}")
    print(f"  Std  : {[round(v, 4) for v in std]}")
    return mean, std


# ─────────────────────────────────────────────
# Class distribution visualisation
# ─────────────────────────────────────────────

def plot_class_distribution(
    summary: Dict[str, Dict[str, int]],
    output_dir: str,
) -> None:
    """
    Saves a grouped bar chart showing per-class image counts for each split.

    Args:
        summary   : Output of verify_dataset_structure().
        output_dir: Directory to save the PNG figure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = list(summary.keys())
    # Gather union of all class names
    all_classes = sorted({cls for counts in summary.values() for cls in counts})
    n_classes = len(all_classes)

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 2), 6))
    colors = ["#4C9BE8", "#E87C4C", "#50C878"]

    for i, split in enumerate(splits):
        counts = [summary[split].get(cls, 0) for cls in all_classes]
        bars = ax.bar(x + i * width, counts, width, label=split, color=colors[i % len(colors)])
        # Add value labels on top of bars
        for bar, val in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution per Dataset Split", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_classes, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    save_path = output_path / "class_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[Plot] Class distribution saved -> {save_path}")


# ─────────────────────────────────────────────
# Optional: resize & copy raw images to staging
# ─────────────────────────────────────────────

def resize_and_stage(
    source_dir: str,
    dest_dir: str,
    img_size: int = 224,
) -> None:
    """
    Copies images from source_dir to dest_dir, resizing each to img_size×img_size.
    Preserves class/split sub-folder structure.

    Args:
        source_dir: Root directory of the raw dataset.
        dest_dir  : Target directory for processed images.
        img_size  : Target image dimension (square).
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    processed = 0

    print(f"\n[Stage] Resizing images from {source_path} -> {dest_path}")
    for img_file in tqdm(list(source_path.rglob("*")), desc="  processing"):
        if img_file.suffix.lower() not in image_exts:
            continue
        relative = img_file.relative_to(source_path)
        target_file = dest_path / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_file).convert("RGB")
        img = img.resize((img_size, img_size), Image.LANCZOS)
        img.save(target_file, quality=95)
        processed += 1

    print(f"[Stage] {processed} images resized and saved to {dest_path}")


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing and verification utilities."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/",
        help="Root directory of the structured dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/",
        help="Directory to save plots."
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Image resize dimension."
    )
    parser.add_argument(
        "--compute_stats", action="store_true",
        help="Compute per-channel mean/std of the training set."
    )
    parser.add_argument(
        "--stage_dir", type=str, default=None,
        help="If provided, resize raw images and copy to this directory."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Verify structure and get class counts
    summary = verify_dataset_structure(args.data_dir)

    # 2. Plot class distribution
    plot_class_distribution(summary, args.output_dir)

    # 3. Optionally compute training stats
    if args.compute_stats:
        train_dir = str(Path(args.data_dir) / "train")
        compute_dataset_stats(train_dir, img_size=args.img_size)

    # 4. Optionally resize & stage images
    if args.stage_dir:
        resize_and_stage(args.data_dir, args.stage_dir, img_size=args.img_size)


if __name__ == "__main__":
    main()
