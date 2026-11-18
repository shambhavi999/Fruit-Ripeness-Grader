"""
data.py — Dataset scanning, loading, and augmentation for Fruit Ripeness Grading.

Supports:
  - Banana: 4-class ordinal (unripe=0 < ripe=1 < overripe=2 < spoiled=3)
  - Tomato: 3-class ordinal (unripe=0 < ripe=1 < overripe=2) + binary damage flag

Usage (standalone — prints class-count table):
    python src/data.py --data_root dataset/
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ── Ordinal class definitions ────────────────────────────────────────────────

BANANA_CLASSES = ["unripe", "ripe", "overripe", "spoiled"]  # ordinal 0→3
TOMATO_ORDINAL_CLASSES = ["unripe", "ripe", "overripe"]     # ordinal 0→2
TOMATO_ALL_CLASSES = ["unripe", "ripe", "overripe", "damaged"]
STRAWBERRY_CLASSES = ["unripe", "partially_ripe", "ripe", "rotten"]  # ordinal 0→3
AVOCADO_CLASSES = ["unripe", "partially_ripe", "ripe", "rotten"]     # ordinal 0→3

BANANA_CLASS_TO_IDX = {c: i for i, c in enumerate(BANANA_CLASSES)}
TOMATO_ORDINAL_TO_IDX = {c: i for i, c in enumerate(TOMATO_ORDINAL_CLASSES)}
STRAWBERRY_CLASS_TO_IDX = {c: i for i, c in enumerate(STRAWBERRY_CLASSES)}
AVOCADO_CLASS_TO_IDX = {c: i for i, c in enumerate(AVOCADO_CLASSES)}

FRUIT_CLASSES = {
    "banana": BANANA_CLASSES,
    "tomato": TOMATO_ORDINAL_CLASSES,
    "strawberry": STRAWBERRY_CLASSES,
    "avocado": AVOCADO_CLASSES,
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Augmentation pipeline for training images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """Deterministic transform for val/test images with high-fidelity antialiasing."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset classes ──────────────────────────────────────────────────────────

class OrdinalFruitDataset(Dataset):
    """
    Generic ordinal fruit dataset for single-task ordinal regression
    (Banana, Strawberry, Avocado).
    Returns (image_tensor, ordinal_label) where label ∈ {0, 1, 2, ..., K-1}.
    """

    def __init__(self, root: str, fruit: str = "banana", split: str = "train", transform=None):
        self.root = Path(root) / fruit / split
        self.fruit = fruit
        self.transform = transform
        self.samples = []  # (path, label)

        if fruit == "banana":
            class_map = BANANA_CLASS_TO_IDX
        elif fruit == "strawberry":
            class_map = STRAWBERRY_CLASS_TO_IDX
        elif fruit == "avocado":
            class_map = AVOCADO_CLASS_TO_IDX
        else:
            raise ValueError(f"Unknown ordinal fruit: {fruit}")

        for class_name, label in class_map.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class BananaDataset(OrdinalFruitDataset):
    """Backward compatibility alias for Banana."""
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__(root, fruit="banana", split=split, transform=transform)


class TomatoDataset(Dataset):
    """
    Tomato multi-task dataset.
    Returns (image_tensor, ordinal_label, damaged_flag).

    ordinal_label: 0=unripe, 1=ripe, 2=overripe, -1 if damaged (mask in loss).
    damaged_flag:  1 if damaged, 0 otherwise.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root) / "tomato" / split
        self.transform = transform
        self.samples = []  # (path, ordinal_label, damaged_flag)

        for class_name in TOMATO_ALL_CLASSES:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            if class_name == "damaged":
                ordinal_label = -1  # will be masked out of ordinal loss
                damaged_flag = 1
            else:
                ordinal_label = TOMATO_ORDINAL_TO_IDX[class_name]
                damaged_flag = 0

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                    self.samples.append((str(img_path), ordinal_label, damaged_flag))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ordinal_label, damaged_flag = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, ordinal_label, damaged_flag


# ── Dataset scanning / EDA ───────────────────────────────────────────────────

def scan_dataset(data_root: str) -> dict:
    """
    Walk the dataset directory and count images per fruit/split/class.
    Returns a nested dict: {fruit: {split: {class: count}}}.
    """
    data_root = Path(data_root)
    result = {}

    for fruit in ["banana", "tomato", "strawberry", "avocado"]:
        fruit_dir = data_root / fruit
        if not fruit_dir.exists():
            continue
        result[fruit] = {}
        for split in ["train", "val", "test"]:
            split_dir = fruit_dir / split
            if not split_dir.exists():
                continue
            result[fruit][split] = {}
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = sum(
                        1 for f in class_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
                    )
                    result[fruit][split][class_dir.name] = count

    return result


def print_dataset_table(stats: dict):
    """Pretty-print the dataset statistics."""
    print("\n" + "=" * 70)
    print("  DATASET SCAN — Per-Class Image Counts")
    print("=" * 70)

    for fruit, splits in stats.items():
        print(f"\n{'─' * 70}")
        print(f"  {fruit.upper()}")
        print(f"{'─' * 70}")
        print(f"  {'Split':<8} {'Class':<14} {'Count':>8} {'% of Split':>12}")
        print(f"  {'─'*8} {'─'*14} {'─'*8} {'─'*12}")

        for split, classes in splits.items():
            total = sum(classes.values())
            for cls_name, count in classes.items():
                pct = 100 * count / total if total > 0 else 0
                print(f"  {split:<8} {cls_name:<14} {count:>8} {pct:>11.1f}%")
            print(f"  {split:<8} {'TOTAL':<14} {total:>8} {'100.0%':>12}")
            print()

        # Overall fruit total
        overall = sum(c for s in splits.values() for c in s.values())
        print(f"  {'ALL':<8} {'GRAND TOTAL':<14} {overall:>8}")


def compute_class_weights(stats: dict, fruit: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced sampling.
    """
    if fruit == "banana":
        classes = BANANA_CLASSES
    elif fruit == "strawberry":
        classes = STRAWBERRY_CLASSES
    elif fruit == "avocado":
        classes = AVOCADO_CLASSES
    else:
        classes = TOMATO_ALL_CLASSES

    train_counts = stats[fruit].get("train", {})
    counts = [train_counts.get(c, 0) for c in classes]
    total = sum(counts)
    weights = [total / (len(counts) * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def get_sampler_weights(dataset, fruit: str, class_weights: torch.Tensor):
    """
    Build per-sample weights for WeightedRandomSampler to handle class imbalance.
    """
    sample_weights = []
    for sample in dataset.samples:
        if fruit != "tomato":
            label = sample[1]
            sample_weights.append(class_weights[label].item())
        else:  # tomato
            ordinal_label, damaged_flag = sample[1], sample[2]
            if damaged_flag == 1:
                idx = 3  # damaged is index 3 in TOMATO_ALL_CLASSES
            else:
                idx = ordinal_label
            sample_weights.append(class_weights[idx].item())

    return torch.tensor(sample_weights, dtype=torch.float64)


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    fruit: str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    img_size: int = 224,
    smoke_test: bool = False,
    smoke_fraction: float = 0.05,
) -> dict:
    """
    Build train/val/test DataLoaders for the specified fruit.
    """
    # Scan for class weights
    stats = scan_dataset(data_root)

    if fruit not in stats:
        raise ValueError(f"No data found for '{fruit}' in {data_root}")

    class_weights = compute_class_weights(stats, fruit)

    train_tf = get_train_transforms(img_size)
    eval_tf = get_eval_transforms(img_size)

    if fruit == "tomato":
        train_ds = TomatoDataset(data_root, split="train", transform=train_tf)
        val_ds   = TomatoDataset(data_root, split="val",   transform=eval_tf)
        test_ds  = TomatoDataset(data_root, split="test",  transform=eval_tf)
    else:
        train_ds = OrdinalFruitDataset(data_root, fruit=fruit, split="train", transform=train_tf)
        val_ds   = OrdinalFruitDataset(data_root, fruit=fruit, split="val",   transform=eval_tf)
        test_ds  = OrdinalFruitDataset(data_root, fruit=fruit, split="test",  transform=eval_tf)

    # Smoke test: subsample
    if smoke_test:
        rng = np.random.RandomState(seed)
        for ds in [train_ds, val_ds, test_ds]:
            n = max(1, int(len(ds.samples) * smoke_fraction))
            indices = rng.choice(len(ds.samples), size=n, replace=False)
            ds.samples = [ds.samples[i] for i in sorted(indices)]

    # Weighted sampler for training
    sampler_weights = get_sampler_weights(train_ds, fruit, class_weights)
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin, generator=g,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        ),
        "class_weights": class_weights,
        "stats": stats,
    }

    return loaders


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scan dataset and print statistics")
    parser.add_argument("--data_root", type=str, default="dataset/",
                        help="Path to dataset/ directory")
    args = parser.parse_args()

    stats = scan_dataset(args.data_root)

    if not stats:
        print("\n[ERROR] No dataset found! Please place the Kaggle dataset as:")
        print("  dataset/banana/{train,val,test}/{unripe,ripe,overripe,spoiled}/")
        print("  dataset/tomato/{train,val,test}/{unripe,ripe,overripe,damaged}/")
        sys.exit(1)

    print_dataset_table(stats)

    # Print class weights
    for fruit in stats:
        weights = compute_class_weights(stats, fruit)
        classes = BANANA_CLASSES if fruit == "banana" else TOMATO_ALL_CLASSES
        print(f"\n  Class weights for {fruit} (inverse-frequency):")
        for cls, w in zip(classes, weights):
            print(f"    {cls:<12} → {w:.3f}")


if __name__ == "__main__":
    main()
