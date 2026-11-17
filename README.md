# Fruit Ripeness Grading & Packhouse QC via Ordinal Regression

A deep-learning computer-vision project that grades the ripeness of **bananas** and
**tomatoes** from photographs, using **ordinal regression** (CORN loss) instead of
standard multi-class classification, with a polished **Streamlit** web app.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Datasets & Class Structure](#datasets--class-structure)
3. [Why Ordinal Regression?](#why-ordinal-regression)
4. [The Tomato "Damaged" Multi-Task Decision](#the-tomato-damaged-multi-task-decision)
5. [Commercial Packhouse Grading Standards (USDA / UNECE)](#commercial-packhouse-grading-standards-usda--unece)
6. [Architecture](#architecture)
7. [Quick Start & Training](#quick-start--training)
8. [Results & Benchmark Metrics](#results--benchmark-metrics)
9. [Streamlit QC System & Conveyor Batch Mode](#streamlit-qc-system--conveyor-batch-mode)
10. [Grad-CAM Explainability](#grad-cam-explainability)
11. [Limitations & Future Work](#limitations--future-work)
12. [References](#references)

---

## Problem Statement

Fruit ripeness classification is a critical computer-vision task in automated post-harvest agriculture, yet standard multi-class classifiers (softmax + cross-entropy) treat all misclassifications identically. In reality, **predicting "unripe" for a spoiled or rotten fruit is a catastrophic error compared to predicting "ripe"** — it has severe economic, logistical, and food safety consequences.

This project frames ripeness grading as **ordinal regression**: a supervised learning problem where classes preserve a strict monotonic biological progression:
$$\text{Unripe} \prec \text{Partially Ripe} \prec \text{Ripe} \prec \text{Overripe / Rotten}$$

---

## Datasets & Class Structure

The project evaluates four major commercial fruits across two peer-reviewed open datasets:

1. **FreshCheck Fruit Ripeness Dataset** (Kaggle): 20,704 images across Banana and Tomato.
2. **Comprehensive Ripening Stages of Strawberries and Avocados** (Mendeley Data, Kamat et al., 2024): 760 benchmark images partitioned into 80/10/10 stratified train/val/test splits.

| Fruit | Ordinal Ripeness Classes | Special Quality Head | Total Images | Primary Source |
|:---|:---|:---:|:---:|:---|
| **Banana** | `unripe` $\rightarrow$ `ripe` $\rightarrow$ `overripe` $\rightarrow$ `spoiled` | None | ~13,478 | FreshCheck (Kaggle) |
| **Tomato** | `unripe` $\rightarrow$ `ripe` $\rightarrow$ `overripe` | Binary `damaged` | ~7,224 | FreshCheck (Kaggle) |
| **Strawberry** | `unripe` $\rightarrow$ `partially_ripe` $\rightarrow$ `ripe` $\rightarrow$ `rotten` | None | 182 | Mendeley Data (2024) |
| **Avocado** | `unripe` $\rightarrow$ `partially_ripe` $\rightarrow$ `ripe` $\rightarrow$ `rotten` | None | 578 | Mendeley Data (2024) |

---

## Why Ordinal Regression?

| Approach | Loss | Error awareness |
|----------|------|-----------------|
| Standard classification | Cross-entropy | All errors equally bad |
| **Ordinal regression (CORN)** | Conditional chain-rule binary CE | Nearby errors penalized less |

We use **CORN** (**C**onditional **O**rdinal **R**egression for **N**eural Networks)
from the `coral-pytorch` library. CORN decomposes the K-class ordinal problem into
K-1 conditional binary tasks using the chain rule of probability:

$$P(Y = k) = P(Y > 0) \cdot P(Y > 1 \mid Y > 0) \cdots P(Y \leq k \mid Y > k-1)$$

This guarantees monotonicity in the cumulative distribution and is strictly more
general than the older CORAL approach (no weight-sharing constraint).

**Metrics:** We evaluate with ordinal-appropriate metrics:
- **MAE** (Mean Absolute Error in class units) — penalizes distant errors more
- **QWK** (Quadratic Weighted Kappa) — standard ordinal agreement metric
- **Accuracy** — for reference

---

## The Tomato "Damaged" Design Decision

> **Key architectural choice — defended in the viva.**

The four tomato folder labels are `unripe`, `ripe`, `overripe`, and `damaged`.
While the first three form a natural ordinal chain (unripe < ripe < overripe),
**"damaged" is NOT a ripeness stage** — it is a quality/condition label. A damaged
tomato can be unripe, ripe, or overripe.

Placing `damaged` at rank 3 in an ordinal chain (unripe < ripe < overripe < damaged)
would be **semantically incorrect** — it would imply that a damaged tomato is
"more ripe" than an overripe one, which is meaningless.

### Solution: Multi-Task Head

```
                        ┌─────────────────────────────────┐
                        │  MobileNetV2 Backbone (frozen→   │
                        │  then fine-tuned)                │
                        └────────────┬────────────────────┘
                                     │ features (1280-d)
                              ┌──────┴──────┐
                              │   Dropout    │
                              └──────┬──────┘
                        ┌────────────┴────────────┐
                        │                         │
                 ┌──────▼──────┐          ┌───────▼──────┐
                 │  CORN Head  │          │ Damage Head  │
                 │  (3 classes)│          │ (binary)     │
                 │  K-1 = 2   │          │ sigmoid      │
                 │  outputs    │          │ 1 output     │
                 └─────────────┘          └──────────────┘
                 unripe/ripe/             damaged:
                 overripe                 yes / no
```

**Training rule:**
- Images from `{unripe, ripe, overripe}` → contribute to **both** heads
  (ordinal loss + damage label = 0)
- Images from `{damaged}` → contribute **only** to the damage head
  (damage label = 1; masked out of ordinal loss)

**Total loss:** `L = λ₁ · corn_loss + λ₂ · bce_loss` (default λ₁=1.0, λ₂=0.5)

---

## Architecture

| Component | Detail |
|-----------|--------|
| Backbone | MobileNetV2 (pretrained ImageNet) |
| Feature dim | 1280 |
| CORN head | Linear(1280 → K-1) |
| Damage head (tomato) | Linear(1280 → 1) + sigmoid |
| Dropout | 0.3 |
| Optimizer | AdamW (lr=1e-3 → 1e-4) |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience=7 on val QWK |

**Two-phase training:**
1. **Phase 1 (epochs 1–5):** Freeze backbone, train heads only → fast convergence
2. **Phase 2 (epoch 6+):** Unfreeze backbone, lower LR 10×, full fine-tuning

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & place dataset

Download from [Kaggle](https://www.kaggle.com/datasets/orofinadedamola/freshcheck-fruit-ripeness-classification-dataset)
and extract into `dataset/`:

```
dataset/
├── banana/{train,val,test}/{unripe,ripe,overripe,spoiled}/
└── tomato/{train,val,test}/{unripe,ripe,overripe,damaged}/
```

### 3. Scan dataset (verify class distribution)

```bash
python src/data.py --data_root dataset/
```

### 4. Train models

```bash
# Smoke test first (2 epochs, 5% data)
python src/train.py --fruit banana --smoke_test
python src/train.py --fruit tomato --smoke_test

# Full training
python src/train.py --fruit banana --epochs 30
python src/train.py --fruit tomato --epochs 30
```

### 5. Evaluate on test set

```bash
python src/evaluate.py --fruit banana
python src/evaluate.py --fruit tomato
```

### 6. Generate Grad-CAM visualizations

```bash
python src/gradcam.py --fruit banana
python src/gradcam.py --fruit tomato
```

### 7. Launch Streamlit app

```bash
streamlit run app.py
```

---

## Results

All evaluation metrics were computed on held-out **test sets** (never seen during training or validation).

### Banana (4-Class Ordinal: Unripe < Ripe < Overripe < Spoiled)
- **Test set size:** 1,351 images
- **Model Checkpoint:** `models/banana_best.pth`

| Metric | Test Result | Description |
|:---|:---:|:---|
| **Accuracy** | **97.93%** | 1,323 / 1,351 images classified with 100% exact ordinal stage |
| **Mean Absolute Error (MAE)** | **0.0385** | Almost zero distance error across the 4-level ordinal scale |
| **Quadratic Weighted Kappa (QWK)** | **0.9649** | Near-perfect ordinal agreement (>0.90 is gold-standard) |

#### Per-Class Performance:
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Unripe** | 97.31% | 99.09% | **0.9819** | 219 |
| **Ripe** | 96.63% | 99.75% | **0.9816** | 402 |
| **Overripe** | 97.79% | 98.52% | **0.9815** | 270 |
| **Spoiled** | 99.55% | 95.43% | **0.9745** | 460 |

---

### Tomato (Multi-Task: 3-Class Ordinal Ripeness + Binary Damage Detection)
- **Test set size:** 724 images (618 for ordinal ripeness + 106 damaged)
- **Model Checkpoint:** `models/tomato_best.pth`

| Metric | Test Result | Description |
|:---|:---:|:---|
| **Ordinal Ripeness Accuracy** | **97.73%** | 604 / 618 correct ripeness stage |
| **Ordinal MAE** | **0.0227** | Average ordinal distance error across 3 stages |
| **Ordinal QWK** | **0.9823** | Exceptional ordinal ranking consistency |
| **Damage Detection Accuracy** | **96.82%** | 699 / 724 correct quality classifications |
| **Damage Precision** | **89.52%** | Low false alarm rate on quality checks |
| **Damage Recall** | **88.68%** | High sensitivity to defects and bruises |

#### Per-Class Ordinal Performance:
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Unripe** | 97.74% | 98.30% | **0.9802** | 176 |
| **Ripe** | 97.69% | 95.91% | **0.9679** | 220 |
| **Overripe** | 97.78% | 99.10% | **0.9843** | 222 |

---

### Avocado (4-Class Ordinal: Unripe < Partially Ripe < Ripe < Rotten)
- **Test set size:** 61 images
- **Model Checkpoint:** `models/avocado_best.pth`

| Metric | Test Result | Description |
|:---|:---:|:---|
| **Accuracy** | **86.89%** | 53 / 61 images classified with exact ordinal stage (Validation: 92.86%) |
| **Mean Absolute Error (MAE)** | **0.1311** | Minimal ordinal distance error |
| **Quadratic Weighted Kappa (QWK)** | **0.9610** | Gold-standard agreement (>0.95) capturing subtle green-to-black skin ripening |

#### Per-Class Performance:
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Unripe** | 76.19% | 94.12% | **0.8421** | 17 |
| **Partially Ripe** | 87.50% | 58.33% | **0.7000** | 12 |
| **Ripe** | 75.00% | 100.00% | **0.8571** | 6 |
| **Rotten** | 100.00% | 92.31% | **0.9600** | 26 |

---

### Strawberry (4-Class Ordinal: Unripe < Partially Ripe < Ripe < Rotten)
- **Test set size:** 21 images
- **Model Checkpoint:** `models/strawberry_best.pth`

| Metric | Test Result | Description |
|:---|:---:|:---|
| **Accuracy** | **76.19%** | 16 / 21 exact matches on high-resolution camera benchmarks |
| **Mean Absolute Error (MAE)** | **0.2857** | Low distance error across delicate visual transition boundaries |
| **Quadratic Weighted Kappa (QWK)** | **0.7500** | Substantial agreement on 4-stage strawberry maturation |

#### Per-Class Performance:
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Unripe** | 100.00% | 100.00% | **1.0000** | 5 |
| **Partially Ripe** | 72.73% | 100.00% | **0.8421** | 8 |
| **Ripe** | 60.00% | 60.00% | **0.6000** | 5 |
| **Rotten** | 0.00% | 0.00% | **0.0000** | 3 |

Visual artifacts generated:
- Confusion matrices: `results/{banana,tomato,strawberry,avocado}/confusion_matrix.png`
- Training & validation curves: `results/{banana,tomato,strawberry,avocado}/loss_curves.png`
- Detailed JSON metrics: `results/{banana,tomato,strawberry,avocado}/metrics.json`
- Grad-CAM heatmap overlays: `results/{banana,tomato,strawberry,avocado}/gradcam/`

---

## Streamlit QC System & Conveyor Batch Mode

The Streamlit web application (`app.py`) provides an industrial dual-mode diagnostic suite:

### 1. Single Specimen Diagnostic Mode
- **Dual Input Channels:** Always-visible drag-and-drop file uploader alongside pre-loaded benchmark test specimens.
- **Continuous Ripeness Index (CRI):** Calculates the exact expected rank $E[Y] = \sum k \cdot P(Y=k)$ on a continuous scale $[0.0, 3.0]$.
- **Commercial Packhouse Grading Standard:** Maps predictions into official **USDA / UNECE Commercial Tiers** (Class 1 / Extra, Class 2, Processing Grade, Cull / Reject) with market price indices and supply chain allocation protocols.
- **Tomato Multi-Task Defect Flag:** Isolated binary damage detection head preventing diseased/cracked tomatoes from entering the table retail chain.
- **Explainability Studio:** Live Grad-CAM activation heatmap overlay with continuous transparency blend slider ($0.0 \rightarrow 1.0$) and selectable scientific colormaps (Jet, Inferno, Viridis, Turbo).

### 2. Bulk Crate / Conveyor Batch QC Mode
- **Multi-Image Industrial Processing:** Allows packhouse inspectors to upload batches or simulate a 16-unit carton in real time.
- **Batch Freshness Telemetry:** Computes aggregate lot acceptance rate, defect percentage, and commercial distribution.
- **Lot Acceptance Verdict:** Automated decision threshold (`LOT ACCEPTED` vs. `FLAGGED FOR RE-SORTING`).
- **Traceable CSV Audit Report:** One-click generation of `Fruit_QC_Audit_Report.csv` with timestamps, specimen IDs, CRI scores, confidence levels, and commercial tiers.

---

## Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which spatial
regions of the input image most influenced the model's prediction. For ripeness
grading, we expect the model to focus on:
- **Banana:** skin color (green→yellow→brown), dark spots, texture
- **Tomato:** color uniformity, cracks, soft spots, green shoulders

Example overlays are saved in `results/{fruit}/gradcam/`.

---

## Limitations & Future Work

1. **Class overlap:** Ripe/overripe boundaries are subjective and may differ between annotators
2. **Lighting sensitivity:** Models trained on studio lighting may struggle with diverse real-world conditions
3. **Single-fruit assumption:** Each image must contain one fruit; no multi-fruit detection
4. **Tomato damage types:** "Damaged" is a broad label; sub-categories (cracked, bruised, moldy) could improve utility
5. **Future:** Mobile deployment (ONNX/TFLite), multi-fruit auto-detection, active learning for ambiguous samples

---

## References

1. Shi, W., Raschka, S. (2021). "CORN — Conditional Ordinal Regression for Neural Networks." *Pattern Recognition*, 122, 108263.
2. Cao, W., Mirjalili, V., Raschka, S. (2020). "Rank Consistent Ordinal Regression for Neural Networks (CORAL)." *Pattern Recognition Letters*, 140, 325–331.
3. Selvaraju, R.R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV 2017*.
4. FreshCheck Dataset: https://www.kaggle.com/datasets/orofinadedamola/freshcheck-fruit-ripeness-classification-dataset
5. `coral-pytorch`: https://github.com/Raschka-research-group/coral-pytorch

---

*Deep Learning for Computer Vision — University Course Project*
