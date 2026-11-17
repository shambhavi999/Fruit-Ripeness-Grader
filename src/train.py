"""
train.py — Training loop for Fruit Ripeness Ordinal Regression.

Features:
  - CORN loss for ordinal regression
  - Multi-task loss for tomato (ordinal + damage)
  - 2-phase training: freeze backbone → unfreeze fine-tune
  - Early stopping on val QWK
  - WeightedRandomSampler for class imbalance
  - Checkpoint saving (best val QWK)
  - Training history CSV logging

Usage:
    python src/train.py --fruit banana --epochs 30
    python src/train.py --fruit banana --smoke_test
"""

import os
import sys
import json
import csv
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import cohen_kappa_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_dataloaders, print_dataset_table
from src.model import build_model


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(all_preds: list, all_labels: list) -> dict:
    """
    Compute ordinal regression metrics.

    Args:
        all_preds: list of predicted ordinal labels
        all_labels: list of true ordinal labels

    Returns:
        dict with accuracy, mae, qwk
    """
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    accuracy = np.mean(preds == labels)
    mae = np.mean(np.abs(preds - labels))
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    return {"accuracy": float(accuracy), "mae": float(mae), "qwk": float(qwk)}


def train_one_epoch(model, loader, optimizer, device, fruit, num_ordinal_classes,
                    lambda_damage=0.5):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if fruit != "tomato":
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            corn_logits, _ = model(images)
            loss = corn_loss(corn_logits, labels, num_classes=num_ordinal_classes)

        else:  # tomato
            images, ordinal_labels, damaged_flags = batch
            images = images.to(device)
            ordinal_labels = ordinal_labels.to(device)
            damaged_flags = damaged_flags.to(device).float()

            optimizer.zero_grad()
            corn_logits, damage_logits = model(images)

            # Ordinal loss: only on non-damaged samples
            ordinal_mask = ordinal_labels >= 0
            loss = torch.tensor(0.0, device=device)

            if ordinal_mask.sum() > 0:
                masked_logits = corn_logits[ordinal_mask]
                masked_labels = ordinal_labels[ordinal_mask]
                loss = loss + corn_loss(masked_logits, masked_labels,
                                        num_classes=num_ordinal_classes)

            # Damage loss: on ALL samples
            bce = nn.BCEWithLogitsLoss()
            damage_loss = bce(damage_logits.squeeze(-1), damaged_flags)
            loss = loss + lambda_damage * damage_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, fruit, num_ordinal_classes, lambda_damage=0.5):
    """
    Validate and return loss + metrics.
    Returns: (val_loss, metrics_dict, damage_accuracy_or_None)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []
    all_damage_preds = []
    all_damage_labels = []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        if fruit != "tomato":
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            corn_logits, _ = model(images)
            loss = corn_loss(corn_logits, labels, num_classes=num_ordinal_classes)

            preds = corn_label_from_logits(corn_logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        else:  # tomato
            images, ordinal_labels, damaged_flags = batch
            images = images.to(device)
            ordinal_labels = ordinal_labels.to(device)
            damaged_flags = damaged_flags.to(device).float()

            corn_logits, damage_logits = model(images)

            loss = torch.tensor(0.0, device=device)
            ordinal_mask = ordinal_labels >= 0
            if ordinal_mask.sum() > 0:
                loss = loss + corn_loss(corn_logits[ordinal_mask],
                                        ordinal_labels[ordinal_mask],
                                        num_classes=num_ordinal_classes)

            bce = nn.BCEWithLogitsLoss()
            damage_loss = bce(damage_logits.squeeze(-1), damaged_flags)
            loss = loss + lambda_damage * damage_loss

            # Collect ordinal predictions (non-damaged only)
            if ordinal_mask.sum() > 0:
                preds = corn_label_from_logits(corn_logits[ordinal_mask]).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(ordinal_labels[ordinal_mask].cpu().numpy())

            # Collect damage predictions (all samples)
            damage_preds = (torch.sigmoid(damage_logits.squeeze(-1)) > 0.5).long()
            all_damage_preds.extend(damage_preds.cpu().numpy())
            all_damage_labels.extend(damaged_flags.long().cpu().numpy())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_preds, all_labels) if all_labels else \
              {"accuracy": 0.0, "mae": 0.0, "qwk": 0.0}

    damage_acc = None
    if all_damage_labels:
        damage_acc = float(np.mean(
            np.array(all_damage_preds) == np.array(all_damage_labels)
        ))

    return avg_loss, metrics, damage_acc


def train(
    fruit: str,
    data_root: str = "dataset/",
    backbone: str = "mobilenet_v2",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    lr_finetune: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_epochs: int = 5,
    patience: int = 7,
    lambda_damage: float = 0.5,
    seed: int = 42,
    smoke_test: bool = False,
    save_dir: str = "models/",
    results_dir: str = "results/",
    num_workers: int = 0,
):
    """Main training function."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  TRAINING: {fruit.upper()} | Device: {device}")
    print(f"  Backbone: {backbone} | Epochs: {epochs} | BS: {batch_size}")
    if smoke_test:
        print(f"  *** SMOKE TEST MODE (5% data, {min(epochs, 2)} epochs) ***")
        epochs = min(epochs, 2)
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        fruit=fruit,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        smoke_test=smoke_test,
    )
    print_dataset_table(loaders["stats"])

    num_ordinal_classes = 4 if fruit in ("banana", "strawberry", "avocado") else 3

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(fruit, backbone=backbone, pretrained=True)
    model.to(device)
    params = model.count_params()
    print(f"\n  Model params: {params['total']:,} total, {params['trainable']:,} trainable")

    # Phase 1: freeze backbone
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    best_qwk = -1.0
    patience_counter = 0
    history = []

    result_dir = Path(results_dir) / fruit
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Phase 2: unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1:
            print(f"\n  >>> Phase 2: Unfreezing backbone, reducing LR to {lr_finetune}")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr_finetune, weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs,
            )

        print(f"\n  Epoch {epoch}/{epochs} (LR={optimizer.param_groups[0]['lr']:.6f})")

        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, device, fruit,
            num_ordinal_classes, lambda_damage,
        )

        val_loss, val_metrics, damage_acc = validate(
            model, loaders["val"], device, fruit,
            num_ordinal_classes, lambda_damage,
        )

        scheduler.step()

        # Log
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_mae": round(val_metrics["mae"], 4),
            "val_qwk": round(val_metrics["qwk"], 4),
        }
        if damage_acc is not None:
            row["val_damage_acc"] = round(damage_acc, 4)
        history.append(row)

        status = (
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  mae={val_metrics['mae']:.4f}  "
            f"qwk={val_metrics['qwk']:.4f}"
        )
        if damage_acc is not None:
            status += f"  damage_acc={damage_acc:.4f}"

        # Early stopping
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            patience_counter = 0
            ckpt_path = save_path / f"{fruit}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "fruit": fruit,
                "backbone": backbone,
                "num_ordinal_classes": num_ordinal_classes,
            }, ckpt_path)
            status += "  ★ BEST"
        else:
            patience_counter += 1
            status += f"  (patience {patience_counter}/{patience})"

        print(status)

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (best QWK: {best_qwk:.4f})")
            break

    # ── Save history ──────────────────────────────────────────────────────
    history_path = result_dir / "training_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"\n  History saved: {history_path}")

    # ── Save loss curves ──────────────────────────────────────────────────
    _plot_training_curves(history, result_dir, fruit)

    print(f"\n  Best checkpoint: {save_path / f'{fruit}_best.pth'}")
    print(f"  Best val QWK: {best_qwk:.4f}")

    return model, history


def _plot_training_curves(history: list, result_dir: Path, fruit: str):
    """Plot and save training/validation loss and metric curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_list = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    val_maes = [h["val_mae"] for h in history]
    val_qwks = [h["val_qwk"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(epochs_list, train_losses, label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs_list, val_losses, label="Val Loss", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{fruit.capitalize()} — Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy & QWK
    axes[1].plot(epochs_list, val_accs, label="Accuracy", marker="o", markersize=3)
    axes[1].plot(epochs_list, val_qwks, label="QWK", marker="s", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{fruit.capitalize()} — Accuracy & QWK")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # MAE
    axes[2].plot(epochs_list, val_maes, label="MAE", marker="o", markersize=3,
                 color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE (class units)")
    axes[2].set_title(f"{fruit.capitalize()} — Mean Absolute Error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(result_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(result_dir / "metric_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved: {result_dir / 'loss_curves.png'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train fruit ripeness model")
    parser.add_argument("--fruit", type=str, required=True,
                        choices=["banana", "tomato", "strawberry", "avocado"])
    parser.add_argument("--data_root", type=str, default="dataset/")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lambda_damage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--save_dir", type=str, default="models/")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
