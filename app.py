"""
app.py — Fruit Ripeness Grading & Industrial Packhouse Quality Control (QC) System.

Features:
  - Supported Fruits: Banana, Tomato, Strawberry, Avocado
  - Dual Operational Modes:
      1. Single Specimen Diagnostic
      2. Bulk Crate / Conveyor Batch QC Mode (with CSV Audit Report Export)
  - Commercial Packhouse Grading Standard (USDA / UNECE Commercial Tiers)
  - Ordinal regression with CORN loss
  - Continuous Ripeness Index (CRI: Expected Rank E[Y])
  - Visual discrete ordinal scale gauge
  - Dynamic Shelf-Life & Post-Harvest Storage Guidance
  - Tomato Multi-Task Quality / Damage Defect Detection
  - Interactive Grad-CAM Studio (Live transparency slider & colormap selector)
  - Clean, professional styling (no emojis in titles, headers, or selectors)

Usage:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageOps
from coral_pytorch.dataset import corn_label_from_logits

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import (
    get_eval_transforms, BANANA_CLASSES, TOMATO_ORDINAL_CLASSES,
    STRAWBERRY_CLASSES, AVOCADO_CLASSES,
)
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fruit Ripeness Grader & Packhouse QC System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling & Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 12px;
        color: #9E9E9E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tier-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4CAF50, #FFC107, #FF9800, #F44336);
    }
</style>
""", unsafe_allow_html=True)


# ── Domain Specifications & Tiers ─────────────────────────────────────────────

FRUIT_SPECS = {
    "banana": {
        "name": "Banana",
        "classes": BANANA_CLASSES,
        "num_classes": 4,
        "max_rank": 3.0,
        "colors": ["#4CAF50", "#FFC107", "#FF9800", "#F44336"],
    },
    "tomato": {
        "name": "Tomato",
        "classes": TOMATO_ORDINAL_CLASSES,
        "num_classes": 3,
        "max_rank": 2.0,
        "colors": ["#4CAF50", "#E91E63", "#FF5722"],
    },
    "strawberry": {
        "name": "Strawberry",
        "classes": STRAWBERRY_CLASSES,
        "num_classes": 4,
        "max_rank": 3.0,
        "colors": ["#4CAF50", "#FF9800", "#E91E63", "#795548"],
    },
    "avocado": {
        "name": "Avocado",
        "classes": AVOCADO_CLASSES,
        "num_classes": 4,
        "max_rank": 3.0,
        "colors": ["#4CAF50", "#8BC34A", "#33691E", "#3E2723"],
    },
}

SHELF_LIFE_SPECS = {
    "banana": {
        "unripe": {"days": "5 – 7 days", "action": "Store at room temperature out of direct sunlight. Allows natural ethylene gas to develop sugars."},
        "ripe": {"days": "2 – 3 days", "action": "Optimal sweetness and texture. Eat fresh or refrigerate to slow peel spot development."},
        "overripe": {"days": "1 – 2 days", "action": "Sugar concentration elevated. Best suited for smoothies, banana bread, or baking."},
        "spoiled": {"days": "0 days (Past consumption)", "action": "Discard safely due to internal fermentation and potential microbial growth."},
    },
    "tomato": {
        "unripe": {"days": "6 – 8 days", "action": "Store stem-side down at room temperature. Do not refrigerate (suppresses volatile aroma enzymes)."},
        "ripe": {"days": "3 – 5 days", "action": "Peak flavor, acidity, and lycopene levels. Store at cool room temperature."},
        "overripe": {"days": "1 – 2 days", "action": "Softened flesh. Best cooked into fresh pasta sauce, soups, or roasted."},
    },
    "strawberry": {
        "unripe": {"days": "3 – 5 days", "action": "Store dry in single layer at 10°C–12°C. Low sugar content; allow partial color development."},
        "partially_ripe": {"days": "2 – 3 days", "action": "Keep in ventilated containers. Move into cold retail storage (2°C–4°C)."},
        "ripe": {"days": "1 – 2 days", "action": "Refrigerate at 1°C–3°C with 90% humidity. Consume fresh immediately."},
        "rotten": {"days": "0 days", "action": "Discard immediately. Highly vulnerable to Botrytis cinerea (grey mold) spreading to adjacent fruit."},
    },
    "avocado": {
        "unripe": {"days": "5 – 7 days", "action": "Firm hard export texture. Store at 6°C–8°C for prolonged shipping or ambient room temp to ripen."},
        "partially_ripe": {"days": "2 – 3 days", "action": "Breaking skin color with slight give. Retail display window."},
        "ripe": {"days": "1 – 2 days", "action": "Dark peel with soft yield to gentle pressure. Optimal for fresh slicing and guacamole."},
        "rotten": {"days": "0 days", "action": "Discard. Flesh blackened, stringy, and rancid from oil oxidation."},
    },
}

COLORMAPS = {
    "Jet": cv2.COLORMAP_JET,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Turbo": cv2.COLORMAP_TURBO,
}


def get_commercial_packhouse_tier(fruit: str, pred_name: str, cri: float, is_damaged: bool = False) -> dict:
    """
    Map predicted ripeness and defects to USDA / UNECE commercial packhouse trade standards.
    """
    if fruit == "banana":
        if pred_name == "spoiled":
            return {
                "tier": "CULL / REJECT (Non-Marketable)",
                "color": "#F44336",
                "badge": "BIO-WASTE REJECT",
                "allocation": "Divert to composting / anaerobic biogas digestion.",
                "price_index": "0% (Economic Loss)",
                "action": "Immediate segregation to prevent ethylene contamination of crate.",
                "marketable": False,
            }
        elif pred_name == "overripe":
            return {
                "tier": "PROCESSING GRADE (Industrial Value)",
                "color": "#FF9800",
                "badge": "INDUSTRIAL VALUE",
                "allocation": "Route to commercial bakeries, smoothies, baby food puree.",
                "price_index": "40% of baseline",
                "action": "Immediate cold-chain transfer to food processing facilities.",
                "marketable": True,
            }
        elif pred_name == "ripe":
            return {
                "tier": "CLASS 2 (Table Consumption / Immediate Retail)",
                "color": "#FFC107",
                "badge": "DIRECT CONSUMER RETAIL",
                "allocation": "Immediate display in grocery retail produce aisle.",
                "price_index": "85% of baseline",
                "action": "Target sale window: 48 – 72 hours.",
                "marketable": True,
            }
        else:  # unripe
            return {
                "tier": "CLASS 1 / EXTRA (Premium Export Grade)",
                "color": "#4CAF50",
                "badge": "PREMIUM EXPORT TIER",
                "allocation": "Wholesale export & controlled ethylene ripening chambers.",
                "price_index": "100% (Premium Value)",
                "action": "Prime long-haul transport quality. Store under controlled humidity.",
                "marketable": True,
            }

    elif fruit == "tomato":
        if is_damaged:
            return {
                "tier": "CULL / REJECT (Defective / Unmarketable)",
                "color": "#F44336",
                "badge": "PHYSICAL DEFECT REJECT",
                "allocation": "Discard due to structural puncture, bruising, or fungal rot.",
                "price_index": "0% (Economic Loss)",
                "action": "Segregate immediately to prevent mold propagation across crate.",
                "marketable": False,
            }
        elif pred_name == "overripe":
            return {
                "tier": "PROCESSING GRADE (Industrial Cannery)",
                "color": "#FF9800",
                "badge": "CANNERY PROCESSING",
                "allocation": "Route to industrial paste, marinara, soup, or puree processing.",
                "price_index": "45% of baseline",
                "action": "Wash and process within 24 hours.",
                "marketable": True,
            }
        elif pred_name == "ripe":
            return {
                "tier": "CLASS 1 (Fresh Market Standard)",
                "color": "#E91E63",
                "badge": "FRESH TABLE STANDARD",
                "allocation": "Supermarket produce section for salad & fresh table use.",
                "price_index": "100% (Full Retail)",
                "action": "Store at 12°C–15°C; do not chill below 10°C.",
                "marketable": True,
            }
        else:  # unripe
            return {
                "tier": "CLASS 1 (Green Shipping Maturity)",
                "color": "#4CAF50",
                "badge": "SHIPPING MATURITY",
                "allocation": "Long-distance distributor shipping. Ripens evenly at depot.",
                "price_index": "95% of baseline",
                "action": "Transport in ventilated cartons; ethylene gas triggers uniform color.",
                "marketable": True,
            }

    elif fruit == "strawberry":
        if pred_name == "rotten":
            return {
                "tier": "CULL / REJECT (Fungal Decay)",
                "color": "#F44336",
                "badge": "MOLD REJECT",
                "allocation": "Discard; Botrytis / fungal risk.",
                "price_index": "0%",
                "action": "Remove to avoid mold spread in punnets.",
                "marketable": False,
            }
        elif pred_name == "unripe":
            return {
                "tier": "PROCESSING GRADE (Acidic / Confectionery)",
                "color": "#4CAF50",
                "badge": "JAM / EXTRACTION",
                "allocation": "Jam, confectionery pectin, or fruit tart acidifier.",
                "price_index": "50%",
                "action": "Process or blend with high-sugar purees.",
                "marketable": True,
            }
        elif pred_name == "partially_ripe":
            return {
                "tier": "CLASS 2 (Local Distribution)",
                "color": "#FFC107",
                "badge": "REGIONAL MARKET",
                "allocation": "Fast-moving short-distance retail distribution.",
                "price_index": "80%",
                "action": "Refrigerate at 2°C–4°C.",
                "marketable": True,
            }
        else:  # ripe
            return {
                "tier": "CLASS 1 / EXTRA (Premium Fresh Punnet)",
                "color": "#E91E63",
                "badge": "PREMIUM FRESH PUNNET",
                "allocation": "Top-tier supermarket fresh punnet display.",
                "price_index": "100% (Premium)",
                "action": "Keep chilled; peak sweetness and aroma.",
                "marketable": True,
            }

    elif fruit == "avocado":
        if pred_name == "rotten":
            return {
                "tier": "CULL / REJECT (Rancid / Rotten)",
                "color": "#F44336",
                "badge": "NON-MARKETABLE",
                "allocation": "Divert to composting / bio-refinery.",
                "price_index": "0%",
                "action": "Discard safely.",
                "marketable": False,
            }
        elif pred_name == "unripe":
            return {
                "tier": "CLASS 1 / HARD (Long Export Tier)",
                "color": "#4CAF50",
                "badge": "HARD EXPORT TIER",
                "allocation": "International refrigerated container export.",
                "price_index": "100%",
                "action": "Store at 5°C–7°C; long shelf-life buffer.",
                "marketable": True,
            }
        elif pred_name == "partially_ripe":
            return {
                "tier": "CLASS 1 (Breaking / Near Retail)",
                "color": "#8BC34A",
                "badge": "BREAKING STAGE",
                "allocation": "Supermarket shelf staging for 2–3 days ripening.",
                "price_index": "90%",
                "action": "Hold at ambient temperature to soften.",
                "marketable": True,
            }
        else:  # ripe
            return {
                "tier": "READY-TO-EAT (Peak Flavor)",
                "color": "#33691E",
                "badge": "READY-TO-EAT RETAIL",
                "allocation": "Restaurant / grocery 'Ripe & Ready' premium display.",
                "price_index": "100% (High Margin)",
                "action": "Consume or refrigerate immediately to hold texture.",
                "marketable": True,
            }

    # Generic fallback
    return {
        "tier": "COMMERCIAL STANDARD",
        "color": "#4CAF50",
        "badge": "STANDARD",
        "allocation": "General commercial distribution.",
        "price_index": "100%",
        "action": "Standard post-harvest handling.",
        "marketable": True,
    }


# ── Model Loading (Cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_model(fruit: str, backbone: str = "mobilenet_v2"):
    """Load a trained model from checkpoint with immediate warmup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("models") / f"{fruit}_best.pth"

    if not checkpoint_path.exists():
        return None, None, device

    model = build_model(fruit, backbone=backbone, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Pre-warm model execution engine
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            _ = model(dummy)
    except Exception:
        pass

    gradcam = GradCAM(model)
    return model, gradcam, device


def compute_ordinal_probabilities(corn_logits: torch.Tensor, num_classes: int):
    """
    Decompose CORN conditional logits into exact class probabilities P(Y=k)
    and expected ordinal rank E[Y] (Continuous Ripeness Index).
    """
    cond_probs = torch.sigmoid(corn_logits[0]).cpu().numpy()
    cum_gt = np.cumprod(cond_probs)

    probs = np.zeros(num_classes)
    probs[0] = 1.0 - cum_gt[0]
    for k in range(1, num_classes - 1):
        probs[k] = cum_gt[k - 1] - cum_gt[k]
    probs[num_classes - 1] = cum_gt[-1]
    probs = np.clip(probs, 0.0, 1.0)
    if probs.sum() > 0:
        probs = probs / probs.sum()

    expected_rank = float(sum(k * p for k, p in enumerate(probs)))
    return probs, expected_rank


# ── Sidebar Controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Operational Mode")
    app_mode = st.radio(
        "Select Operating Mode",
        options=["Single Specimen Diagnostic", "Bulk Crate / Conveyor Batch QC"],
        index=0,
        help="Switch between detailed single-image inspection and bulk packhouse crate auditing."
    )

    st.markdown("---")
    st.markdown("### Crop Selection")
    fruit = st.selectbox(
        "Select Fruit Type",
        options=["banana", "tomato", "strawberry", "avocado"],
        format_func=lambda x: FRUIT_SPECS[x]["name"],
    )

    st.markdown("---")
    st.markdown("### Explainability Studio")
    gradcam_alpha = st.slider(
        "Grad-CAM Heatmap Blend",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Adjust the overlay transparency of the Grad-CAM activation map.",
    )

    selected_colormap = st.selectbox(
        "Heatmap Colormap",
        options=["Jet", "Inferno", "Viridis", "Turbo"],
        index=0,
    )

    st.markdown("---")
    with st.expander("Model Specifications"):
        st.markdown(f"""
        - **Architecture:** MobileNetV2 + CORN Head
        - **Input Resolution:** 224 × 224 px
        - **Loss Function:** Conditional Ordinal Regression (CORN)
        - **Active Crop:** {FRUIT_SPECS[fruit]['name']}
        - **Number of Classes:** {FRUIT_SPECS[fruit]['num_classes']}
        - **Inference Device:** {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
        """)


# ── Main Header ──────────────────────────────────────────────────────────────

st.title("Fruit Ripeness Grader")

# Load model
model, gradcam, device = load_model(fruit)

if model is None:
    st.warning(f"Checkpoint `models/{fruit}_best.pth` is currently training or pending. Please select an active trained model.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE SPECIMEN DIAGNOSTIC
# ═════════════════════════════════════════════════════════════════════════════

if app_mode == "Single Specimen Diagnostic":
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Input Image")

        # 1. Primary Direct File Uploader (Always visible!)
        uploaded_file = st.file_uploader(
            "Upload a fruit photo (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"uploader_{fruit}",
            help="Drag and drop or browse an image file from your computer.",
        )

        # 2. Benchmark Sample Selector
        sample_dir = Path("sample_images")
        sample_images = {}
        if sample_dir.exists():
            for f in sorted(sample_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png') and fruit in f.stem.lower():
                    clean_name = f.stem.replace(f"{fruit}_", "").replace("_sample", "").replace("_", " ").capitalize()
                    sample_images[f"{clean_name} Sample"] = str(f)

        sample_options = ["None (Upload Mode)"] + list(sample_images.keys())
        selected_sample = st.selectbox(
            "Or choose a benchmark test sample",
            options=sample_options,
            index=0 if uploaded_file else (1 if len(sample_options) > 1 else 0),
            help="Quickly test pre-loaded benchmark specimens from the held-out test set.",
        )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image = ImageOps.exif_transpose(image)
            st.caption(f"Displaying uploaded image: `{uploaded_file.name}`")
        elif selected_sample != "None (Upload Mode)" and selected_sample in sample_images:
            image = Image.open(sample_images[selected_sample]).convert("RGB")
            image = ImageOps.exif_transpose(image)
            st.caption(f"Displaying benchmark test specimen: {selected_sample}")

        if image is not None:
            st.image(image, caption="Specimen for Analysis", use_container_width=True)

    with col_results:
        st.subheader("Analysis & Diagnostic Output")

        if image is not None:
            # Preprocessing & Inference
            transform = get_eval_transforms()
            input_tensor = transform(image).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                corn_logits, damage_logits = model(input_tensor)
                pred_label = corn_label_from_logits(corn_logits).item()
            latency_ms = (time.perf_counter() - t0) * 1000

            spec = FRUIT_SPECS[fruit]
            classes = spec["classes"]
            num_classes = spec["num_classes"]
            max_rank = spec["max_rank"]
            colors = spec["colors"]

            probs, expected_rank = compute_ordinal_probabilities(corn_logits, num_classes)
            pred_name = classes[pred_label]
            confidence = probs[pred_label]

            # Damage check for tomato
            is_damaged = False
            damage_prob = 0.0
            if fruit == "tomato" and damage_logits is not None:
                damage_prob = torch.sigmoid(damage_logits).item()
                is_damaged = damage_prob > 0.5

            # Commercial Grade Standard Tier
            tier_info = get_commercial_packhouse_tier(fruit, pred_name, expected_rank, is_damaged)

            # Key Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Ripeness Stage</div>
                    <div class="metric-value">{pred_name.replace('_', ' ').upper()}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Confidence</div>
                    <div class="metric-value">{confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #2196F3;">
                    <div class="metric-label">Continuous Ripeness</div>
                    <div class="metric-value">{expected_rank:.2f} / {max_rank:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Continuous Ripeness Meter
            normalized_cri = expected_rank / max_rank
            st.markdown(f"**Continuous Ripeness Position:** `{expected_rank:.2f}` on monotonic scale `[0.00 – {max_rank:.1f}]`")
            st.progress(min(max(normalized_cri, 0.0), 1.0))

            # Discrete Ordinal Scale Visualizer
            gauge_cols = st.columns(num_classes)
            for i, cls in enumerate(classes):
                with gauge_cols[i]:
                    is_pred = i == pred_label
                    color = colors[i]
                    cls_display = cls.replace('_', ' ').capitalize()
                    if is_pred:
                        st.markdown(
                            f'<div style="background-color:{color};color:white;'
                            f'padding:8px 4px;border-radius:6px;text-align:center;'
                            f'font-weight:700;border:2px solid #FFFFFF;">'
                            f'ACTIVE: {cls_display.upper()}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background-color:{color}22;color:{color};'
                            f'padding:8px 4px;border-radius:6px;text-align:center;font-size:13px;">'
                            f'{cls_display} ({probs[i]:.0%})</div>',
                            unsafe_allow_html=True,
                        )

            # ── Commercial Packhouse Grading Standard (USDA / UNECE) ─────────
            st.markdown("---")
            st.markdown("### Commercial Packhouse Grading Standard (USDA / UNECE)")
            
            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                st.markdown(
                    f'<div style="background-color:{tier_info["color"]};color:white;'
                    f'padding:8px 14px;border-radius:6px;font-weight:700;font-size:15px;display:inline-block;margin-bottom:10px;">'
                    f'{tier_info["tier"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Market Price Index:** `{tier_info['price_index']}`")
                st.markdown(f"**Operational Protocol:** {tier_info['action']}")
            with p_col2:
                st.markdown(f"**Supply Chain Allocation:**\n\n{tier_info['allocation']}")

            # Tomato Damage Head Result
            if fruit == "tomato" and damage_logits is not None:
                st.markdown("---")
                if is_damaged:
                    st.error(f"Quality Flag: Physical Damage Detected ({damage_prob:.1%} confidence)\n\n"
                             f"Cracks, punctures, or surface lesions observed. Divert from fresh table cartons.")
                else:
                    st.success(f"Quality Flag: Structurally Sound ({(1-damage_prob):.1%} sound confidence)\n\n"
                               f"No significant physical abrasions or cracks identified.")

            # Shelf-Life & Handling Advice
            st.markdown("---")
            st.markdown("### Shelf-Life & Storage Guidance")
            shelf_spec = SHELF_LIFE_SPECS.get(fruit, {}).get(pred_name, {"days": "N/A", "action": "N/A"})

            c_col1, c_col2 = st.columns([1, 2])
            with c_col1:
                st.markdown(f"**Estimated Shelf-Life:**\n### {shelf_spec['days']}")
            with c_col2:
                st.markdown(f"**Storage Recommendation:**\n{shelf_spec['action']}")

            # Grad-CAM Visualization
            st.markdown("---")
            st.markdown("### Explainable AI (Grad-CAM Activation Map)")
            st.caption(f"Gradient-weighted Class Activation Mapping targeting the final conv layer. Blending alpha: {gradcam_alpha:.2f}")

            with torch.enable_grad():
                cam = gradcam.generate(input_tensor, target_class=pred_label)

            original_np = np.array(image)
            overlay = overlay_heatmap(
                original_np,
                cam,
                alpha=gradcam_alpha,
                colormap=COLORMAPS[selected_colormap],
            )

            cam_col1, cam_col2 = st.columns(2)
            with cam_col1:
                st.image(original_np, caption="Original Specimen", use_container_width=True)
            with cam_col2:
                st.image(overlay, caption=f"Grad-CAM Heatmap ({selected_colormap})", use_container_width=True)

        else:
            st.info("Upload an image or select a benchmark sample on the left to begin grading.")


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2: BULK CRATE / CONVEYOR BATCH QC MODE
# ═════════════════════════════════════════════════════════════════════════════

else:
    st.subheader("Bulk Crate / Conveyor Batch QC Inspection")
    st.markdown("Simulate high-throughput commercial sorting lines. Process multiple specimens simultaneously, evaluate crate lot acceptance thresholds, and export official quality control inspection audit reports.")

    batch_col1, batch_col2 = st.columns([1, 2], gap="large")

    with batch_col1:
        st.markdown("#### Batch Input Configuration")

        uploaded_batch = st.file_uploader(
            "Upload Batch Images (select multiple files)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"batch_uploader_{fruit}",
            help="Select several fruit images representing an incoming pallet or carton."
        )

        use_simulated_crate = st.button("Load Simulated Packhouse Carton (16 Units)")
        if use_simulated_crate:
            st.session_state["active_batch_mode"] = "simulated"
        elif uploaded_batch:
            st.session_state["active_batch_mode"] = "uploaded"

    batch_items = []
    if uploaded_batch and st.session_state.get("active_batch_mode") == "uploaded":
        for f in uploaded_batch:
            try:
                img = Image.open(f).convert("RGB")
                img = ImageOps.exif_transpose(img)
                batch_items.append((f.name, img))
            except Exception:
                pass
    elif st.session_state.get("active_batch_mode") == "simulated":
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            for p in sorted(sample_dir.glob("*.jpg")):
                try:
                    img = Image.open(p).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    batch_items.append((p.name, img))
                except Exception:
                    pass

    with batch_col2:
        if batch_items:
            st.markdown(f"#### Processing Lot: `{len(batch_items)} Specimens`")
            progress_bar = st.progress(0.0)

            transform = get_eval_transforms()
            spec = FRUIT_SPECS[fruit]
            classes = spec["classes"]
            num_classes = spec["num_classes"]
            max_rank = spec["max_rank"]

            results_list = []
            start_time = time.perf_counter()

            # Vectorized fast batch inference
            tensors = [transform(img) for _, img in batch_items]
            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                batch_corn_logits, batch_damage_logits = model(batch_tensor)

            for idx, (fname, img) in enumerate(batch_items):
                corn_logits = batch_corn_logits[idx:idx+1]
                pred_idx = corn_label_from_logits(corn_logits).item()

                probs, cri = compute_ordinal_probabilities(corn_logits, num_classes)
                pred_stage = classes[pred_idx]
                conf = probs[pred_idx]

                is_dmg = False
                if fruit == "tomato" and batch_damage_logits is not None:
                    is_dmg = torch.sigmoid(batch_damage_logits[idx:idx+1]).item() > 0.5

                tier_info = get_commercial_packhouse_tier(fruit, pred_stage, cri, is_dmg)

                results_list.append({
                    "Specimen ID": fname,
                    "Fruit": fruit.capitalize(),
                    "Predicted Stage": pred_stage.replace('_', ' ').capitalize(),
                    "Continuous Index": round(cri, 2),
                    "Confidence": f"{conf:.1%}",
                    "Commercial Tier": tier_info["tier"].split('(')[0].strip(),
                    "Price Index": tier_info["price_index"],
                    "Quality Status": "Damaged/Defect" if is_dmg else ("Spoiled" if pred_stage in ["spoiled", "rotten"] else "Sound"),
                    "Marketable": tier_info["marketable"],
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            progress_bar.progress(1.0)
            total_batch_time = (time.perf_counter() - start_time) * 1000
            df_results = pd.DataFrame(results_list)

            # Summary metrics
            total_count = len(df_results)
            marketable_count = sum(df_results["Marketable"])
            acceptance_pct = (marketable_count / total_count) * 100 if total_count > 0 else 0
            reject_pct = 100.0 - acceptance_pct

            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Total Inspected", f"{total_count} units")
            with s_col2:
                st.metric("Lot Acceptance Rate", f"{acceptance_pct:.1f}%")
            with s_col3:
                st.metric("Batch Latency", f"{total_batch_time:.0f} ms", f"{total_count / (total_batch_time/1000):.1f} FPS")

            # Lot Acceptance Verdict
            if acceptance_pct >= 85.0:
                st.success(f"LOT VERDICT: ACCEPTED FOR COMMERCIAL DISTRIBUTION ({acceptance_pct:.1f}% marketable units >= 85% packhouse threshold)")
            else:
                st.error(f"LOT VERDICT: FLAGGED FOR MANUAL RE-SORTING ({reject_pct:.1f}% defect/reject ratio exceeds 15% tolerance)")

            # Grade Breakdown Table
            st.markdown("#### Commercial Grade Distribution")
            grade_dist = df_results["Commercial Tier"].value_counts().reset_index()
            grade_dist.columns = ["Commercial Tier", "Unit Count"]
            grade_dist["Percentage"] = (grade_dist["Unit Count"] / total_count * 100).apply(lambda x: f"{x:.1f}%")
            st.dataframe(grade_dist, use_container_width=True)

            # Detailed Specimens Table
            st.markdown("#### Individual Specimen Audit Trail")
            st.dataframe(df_results[["Specimen ID", "Predicted Stage", "Continuous Index", "Confidence", "Commercial Tier", "Quality Status"]], use_container_width=True)

            # Export CSV Audit Report
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Official QC Inspection Audit Report (.CSV)",
                data=csv_data,
                file_name=f"Fruit_QC_Audit_Report_{fruit}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Export full traceable CSV record for packhouse inventory and regulatory compliance."
            )

        else:
            st.info("Upload multiple images or click 'Load Simulated Packhouse Carton' to execute batch quality control analysis.")
