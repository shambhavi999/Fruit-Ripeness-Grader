"""
evaluate.py — Evaluation on the held-out test set for Fruit Ripeness models.

Computes: Accuracy, MAE, QWK, confusion matrix, per-class metrics.
Saves: confusion_matrix.png, metrics.json

Usage:
    python src/evaluate.py --fruit banana
    python src/evaluate.py --fruit tomato
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score
)
from coral_pytorch.dataset import corn_label_from_logits
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import (
    get_dataloaders, BANANA_CLASSES, TOMATO_ORDINAL_CLASSES, TOMATO_ALL_CLASSES,
    STRAWBERRY_CLASSES, AVOCADO_CLASSES,
)
from src.model import build_model


def evaluate_model(
    fruit: str,
    data_root: str = "dataset/",
    backbone: str = "mobilenet_v2",
    checkpoint_path: str = None,
    results_dir: str = "results/",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Evaluate a trained model on the test set.

    Returns:
        dict of all metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = f"models/{fruit}_best.pth"

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {fruit.upper()} on TEST SET")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────
    num_ordinal_classes = 4 if fruit in ("banana", "strawberry", "avocado") else 3
    model = build_model(fruit, backbone=backbone, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Val metrics at save time: {ckpt.get('val_metrics', 'N/A')}")

    # ── Load test data ────────────────────────────────────────────────────
    loaders = get_dataloaders(
        fruit=fruit, data_root=data_root, batch_size=batch_size,
        num_workers=num_workers, seed=seed,
    )
    test_loader = loaders["test"]

    # ── Predict ───────────────────────────────────────────────────────────
    all_preds = []
    all_labels = []
    all_damage_preds = []
    all_damage_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Testing"):
            if fruit != "tomato":
                images, labels = batch
                images = images.to(device)
                corn_logits, _ = model(images)
                preds = corn_label_from_logits(corn_logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
            else:
                images, ordinal_labels, damaged_flags = batch
                images = images.to(device)
                corn_logits, damage_logits = model(images)

                ordinal_mask = ordinal_labels >= 0
                if ordinal_mask.sum() > 0:
                    preds = corn_label_from_logits(
                        corn_logits[ordinal_mask]
                    ).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(ordinal_labels[ordinal_mask].numpy())

                damage_preds = (
                    torch.sigmoid(damage_logits.squeeze(-1)) > 0.5
                ).long().cpu().numpy()
                all_damage_preds.extend(damage_preds)
                all_damage_labels.extend(damaged_flags.numpy())

    # ── Compute metrics ───────────────────────────────────────────────────
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(np.mean(all_preds == all_labels))
    mae = float(np.mean(np.abs(all_preds - all_labels)))
    qwk = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))

    if fruit == "banana":
        class_names = BANANA_CLASSES
    elif fruit == "strawberry":
        class_names = STRAWBERRY_CLASSES
    elif fruit == "avocado":
        class_names = AVOCADO_CLASSES
    else:
        class_names = TOMATO_ORDINAL_CLASSES

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True,
    )

    metrics = {
        "fruit": fruit,
        "backbone": backbone,
        "accuracy": round(accuracy, 4),
        "mae": round(mae, 4),
        "qwk": round(qwk, 4),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }

    # Damage metrics for tomato
    if all_damage_labels:
        all_damage_preds = np.array(all_damage_preds)
        all_damage_labels = np.array(all_damage_labels)
        damage_acc = float(np.mean(all_damage_preds == all_damage_labels))
        damage_tp = int(np.sum((all_damage_preds == 1) & (all_damage_labels == 1)))
        damage_fp = int(np.sum((all_damage_preds == 1) & (all_damage_labels == 0)))
        damage_fn = int(np.sum((all_damage_preds == 0) & (all_damage_labels == 1)))
        damage_precision = damage_tp / max(damage_tp + damage_fp, 1)
        damage_recall = damage_tp / max(damage_tp + damage_fn, 1)
        metrics["damage"] = {
            "accuracy": round(damage_acc, 4),
            "precision": round(damage_precision, 4),
            "recall": round(damage_recall, 4),
        }

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  TEST RESULTS — {fruit.upper()}")
    print(f"{'─'*50}")
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  MAE:            {mae:.4f}")
    print(f"  QWK:            {qwk:.4f}")
    if "damage" in metrics:
        print(f"  Damage Acc:     {metrics['damage']['accuracy']:.4f}")
        print(f"  Damage Prec:    {metrics['damage']['precision']:.4f}")
        print(f"  Damage Recall:  {metrics['damage']['recall']:.4f}")
    print(f"\n  Classification Report (ordinal classes):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ── Save results ──────────────────────────────────────────────────────
    result_dir = Path(results_dir) / fruit
    result_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{fruit.capitalize()} — Confusion Matrix\n"
                 f"Acc={accuracy:.3f}  MAE={mae:.3f}  QWK={qwk:.3f}",
                 fontsize=13)
    plt.tight_layout()
    cm_path = result_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Confusion matrix saved: {cm_path}")

    # metrics.json
    json_path = result_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {json_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate fruit ripeness model")
    parser.add_argument("--fruit", type=str, required=True,
                        choices=["banana", "tomato", "strawberry", "avocado"])
    parser.add_argument("--data_root", type=str, default="dataset/")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    evaluate_model(
        fruit=args.fruit,
        data_root=args.data_root,
        backbone=args.backbone,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
