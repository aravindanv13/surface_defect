"""
data_loader.py
--------------
Handles dataset loading, splitting, and DataLoader creation for the
Surface Defect Detection project.

Folder structure expected:
    data/
        train/
            crack/
            scratch/
            dent/
            no_defect/
        val/
            ...
        test/
            ...
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# ─────────────────────────────────────────────
# Class label mapping (consistent across files)
# ─────────────────────────────────────────────
CLASS_NAMES: List[str] = ["crack", "dent", "no_defect", "scratch"]
NUM_CLASSES: int = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Returns augmented transforms for training data.
    Augmentations include:
        - Random horizontal/vertical flip
        - Random rotation (±15°)
        - Color jitter (brightness, contrast, saturation)
        - Resize + CenterCrop for consistent sizing
        - Normalization (ImageNet stats)
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Returns deterministic transforms for validation/test data.
    No augmentation — only resize, crop, and normalize.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─────────────────────────────────────────────
# Custom Dataset (for inference on raw folders)
# ─────────────────────────────────────────────

class DefectDataset(Dataset):
    """
    Custom dataset that wraps torchvision.datasets.ImageFolder
    with additional metadata support.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir   : Path to folder with class subfolders.
            transform  : torchvision transforms to apply.
            class_names: Optional explicit class list (for label consistency).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Use ImageFolder internally for simplicity
        self._inner = datasets.ImageFolder(
            root=str(self.root_dir),
            transform=self.transform,
        )

        # If explicit class names provided, verify they match
        if class_names is not None:
            assert set(class_names) == set(self._inner.classes), (
                f"Provided class_names {class_names} do not match "
                f"found classes {self._inner.classes}"
            )
        self.classes = self._inner.classes
        self.class_to_idx = self._inner.class_to_idx

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._inner[idx]

    @property
    def targets(self) -> List[int]:
        """Returns list of integer labels for the entire dataset."""
        return self._inner.targets


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds train, validation, and test DataLoaders from a structured
    directory.

    Args:
        data_dir   : Root data directory (must contain train/, val/, test/).
        batch_size : Number of samples per batch.
        img_size   : Target image size (square).
        num_workers: Parallel data loading workers.
        pin_memory : Whether to pin memory for faster GPU transfer.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    data_path = Path(data_dir)

    train_dataset = DefectDataset(
        root_dir=data_path / "train",
        transform=get_train_transforms(img_size),
    )
    val_dataset = DefectDataset(
        root_dir=data_path / "val",
        transform=get_val_transforms(img_size),
    )
    test_dataset = DefectDataset(
        root_dir=data_path / "test",
        transform=get_val_transforms(img_size),
    )

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,          # avoid partial batches during training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[DataLoader] Classes       : {class_names}")
    print(f"[DataLoader] Train samples : {len(train_dataset)}")
    print(f"[DataLoader] Val samples   : {len(val_dataset)}")
    print(f"[DataLoader] Test samples  : {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────
# Utility: single image loader for inference
# ─────────────────────────────────────────────

def load_single_image(
    image_path: str,
    img_size: int = 224,
) -> torch.Tensor:
    """
    Loads and preprocesses a single image for model inference.

    Args:
        image_path: Path to the image file.
        img_size  : Target resize dimension.

    Returns:
        Tensor of shape (1, 3, img_size, img_size).
    """
    transform = get_val_transforms(img_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor
