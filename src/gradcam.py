"""
gradcam.py — Grad-CAM visualization for Fruit Ripeness models.

Generates heatmap overlays showing which image regions drive the prediction.
Works with MobileNetV2 and ResNet18 backbones.

Usage:
    python src/gradcam.py --fruit banana
    python src/gradcam.py --fruit tomato
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from coral_pytorch.dataset import corn_label_from_logits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import (
    get_eval_transforms, BANANA_CLASSES, TOMATO_ORDINAL_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD,
)
from src.model import build_model


class GradCAM:
    """
    Grad-CAM implementation for FruitOrdinalNet.

    Hooks into the target conv layer, computes gradients w.r.t. the predicted
    ordinal class, and produces a class-activation heatmap.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Auto-detect target layer if not provided
        if target_layer is None:
            target_layer = model.gradcam_target_layer
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image
            target_class: ordinal class index to explain; if None, uses predicted

        Returns:
            cam: (H, W) heatmap normalized to [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        corn_logits, _ = self.model(input_tensor)

        # Get predicted class if not specified
        if target_class is None:
            target_class = corn_label_from_logits(corn_logits).item()

        # For CORN: target the conditional logit corresponding to this class
        # We backprop through the sum of relevant logits for this rank
        # For rank k, the relevant logits are indices 0..k-1 (should be positive)
        # and index k (should be negative if exists)
        num_tasks = corn_logits.shape[1]
        if target_class == 0:
            # Class 0: first logit should be negative (P(Y>0) is low)
            target_logit = -corn_logits[0, 0]
        elif target_class >= num_tasks:
            # Highest class: all logits should be positive
            target_logit = corn_logits[0, :].sum()
        else:
            # Intermediate: logits before target are positive, target is negative
            target_logit = corn_logits[0, :target_class].sum() - corn_logits[0, target_class]

        target_logit.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]       # (C, h, w)
        activations = self.activations[0]   # (C, h, w)

        # Global average pool gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (h, w)

        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()

        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam


def overlay_heatmap(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_image: (H, W, 3) RGB image, uint8
        cam: (H, W) heatmap normalized to [0, 1]
        alpha: blending factor
        colormap: OpenCV colormap

    Returns:
        overlay: (H, W, 3) RGB overlaid image, uint8
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.float32(heatmap) * alpha + np.float32(original_image) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to uint8 RGB numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def generate_gradcam_examples(
    fruit: str,
    data_root: str = "dataset/",
    backbone: str = "mobilenet_v2",
    checkpoint_path: str = None,
    results_dir: str = "results/",
    num_examples: int = 5,
    seed: int = 42,
):
    """Generate Grad-CAM overlays for example test images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = f"models/{fruit}_best.pth"

    print(f"\n{'='*60}")
    print(f"  GRAD-CAM: {fruit.upper()}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # Load model
    model = build_model(fruit, backbone=backbone, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    transform = get_eval_transforms()

    if fruit == "banana":
        classes = BANANA_CLASSES
    else:
        classes = TOMATO_ORDINAL_CLASSES

    output_dir = Path(results_dir) / fruit / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = Path(data_root) / fruit / "test" / cls_name
        if not cls_dir.exists():
            print(f"  [SKIP] {cls_dir} not found")
            continue

        img_files = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        if not img_files:
            continue

        # Sample examples
        n = min(num_examples, len(img_files))
        indices = rng.choice(len(img_files), size=n, replace=False)

        for i, idx in enumerate(indices):
            img_path = img_files[idx]
            original = Image.open(img_path).convert("RGB")
            original_np = np.array(original)

            # Preprocess
            input_tensor = transform(original).unsqueeze(0).to(device)

            # Generate Grad-CAM
            with torch.enable_grad():
                cam = gradcam.generate(input_tensor, target_class=cls_idx)

            # Get prediction
            with torch.no_grad():
                corn_logits, damage_logits = model(input_tensor)
                pred_label = corn_label_from_logits(corn_logits).item()
                pred_name = classes[pred_label] if pred_label < len(classes) else "?"

            # Create overlay
            overlay = overlay_heatmap(original_np, cam, alpha=0.4)

            # Plot side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(original_np)
            axes[0].set_title(f"Original — True: {cls_name}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(overlay)
            correct = "✓" if pred_label == cls_idx else "✗"
            axes[1].set_title(
                f"Grad-CAM — Pred: {pred_name} {correct}", fontsize=11
            )
            axes[1].axis("off")

            plt.suptitle(
                f"{fruit.capitalize()} Grad-CAM: {cls_name}",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()

            save_name = f"{cls_name}_{i+1}.png"
            fig.savefig(output_dir / save_name, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {output_dir / save_name}")

    print(f"\n  All Grad-CAM examples saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples")
    parser.add_argument("--fruit", type=str, required=True,
                        choices=["banana", "tomato"])
    parser.add_argument("--data_root", type=str, default="dataset/")
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    generate_gradcam_examples(
        fruit=args.fruit,
        data_root=args.data_root,
        backbone=args.backbone,
        checkpoint_path=args.checkpoint,
        results_dir=args.results_dir,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
