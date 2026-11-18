"""
app.py — Fruit Ripeness Grading & Industrial Packhouse Quality Control (QC) System.
Bento Box Layout & Minimalist Modern UI.

Features:
  - Bento Box Grid Interface (tidy blocks, softly rounded corners, minimal whitespace, high legibility)
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
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
torch.set_num_threads(max(1, os.cpu_count() or 4))
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
    page_title="Fruit Ripeness Grading via Ordinal Regression",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bento Box Styling & Custom CSS ───────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Syne:wght@700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall Canvas */
    .stApp {
        background-color: #F8FAF8;
        color: #111827;
    }

    /* Container bounds & padding */
    /* Container bounds & padding (Fixed: generous top space so navbar is never cut off) */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
    }

    /* Floating Bento Navigation Bar (Inspired by Nanoplazz) */
    .bento-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E5E9E5;
        border-radius: 9999px;
        padding: 10px 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    .bento-nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 800;
        font-size: 15px;
        color: #111827;
        letter-spacing: -0.3px;
    }
    .fruitscan-brand-logo {
        font-family: 'Syne', 'Outfit', 'Plus Jakarta Sans', -apple-system, sans-serif;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #0F172A;
        display: inline-flex;
        align-items: center;
        line-height: 1;
        user-select: none;
    }
    .fruitscan-brand-accent {
        background: linear-gradient(135deg, #16A34A 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    .fruitscan-brand-dot {
        color: #16A34A;
        -webkit-text-fill-color: #16A34A;
        font-size: 24px;
        font-weight: 900;
        line-height: 0;
        margin-left: 1px;
    }
    /* Sidebar Brand Header */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.5rem !important;
    }
    .sidebar-brand-header {
        padding: 4px 0 16px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #E5E9E5;
    }
    .fruitscan-sidebar-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .fruitscan-sidebar-logo .fruitscan-brand-logo {
        font-size: 26px !important;
        letter-spacing: -0.9px !important;
    }
    .fruitscan-sidebar-tagline {
        font-size: 11.5px;
        font-weight: 600;
        color: #6B7280;
        line-height: 1.35;
        letter-spacing: -0.2px;
    }

    @media (max-width: 1200px) {
        .bento-brand-sub, .bento-brand-sep {
            display: none !important;
        }
    }
    .bento-nav-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #16A34A;
        box-shadow: 0 0 8px rgba(22, 163, 74, 0.5);
        display: inline-block;
    }
    .bento-nav-links {
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 13px;
        font-weight: 600;
        color: #4B5563;
    }
    .bento-nav-pill-active {
        background: #DCFCE7;
        color: #15803D;
        padding: 5px 14px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #BBF7D0;
    }
    .bento-nav-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #F3F4F6;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        color: #374151;
    }

    /* Bento Hero */
    .bento-hero {
        text-align: center;
        padding: 14px 0 22px 0;
        margin-bottom: 18px;
    }
    .bento-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #DCFCE7;
        color: #15803D;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        border: 1px solid #BBF7D0;
    }
    .bento-hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        letter-spacing: -1px;
        line-height: 1.2;
        margin-bottom: 10px;
    }
    .bento-hero-title span {
        color: #16A34A;
        font-style: italic;
    }
    .bento-hero-subtitle {
        font-size: 15px;
        color: #6B7280;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Signature Bento Stats Grid (Nanoplazz Style: Mint, White, Sage, Bordered White) */
    .bento-top-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .bento-top-mint {
        background: #D1FAE5;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #A7F3D0;
        color: #065F46;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bento-top-white {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #E5E9E5;
        color: #1F2937;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bento-top-sage {
        background: #ECFDF5;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #D1FAE5;
        color: #047857;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bento-top-val {
        font-size: 26px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .bento-top-lbl {
        font-size: 11.5px;
        font-weight: 700;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Bento Container Card */
    .bento-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #E5E9E5;
        box-shadow: 0 4px 20px -2px rgba(16, 185, 129, 0.04), 0 2px 8px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .bento-card:hover {
        box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.08), 0 4px 12px -2px rgba(0, 0, 0, 0.03);
    }
    .bento-card-title {
        font-size: 18px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }
    .bento-card-desc {
        font-size: 13px;
        color: #6B7280;
        margin-bottom: 18px;
    }

    /* Bento Metric Tiles (Inspired by Nanoplazz clean bento style) */
    .bento-metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .bento-metric-mint {
        background: #D1FAE5;
        border-radius: 18px;
        padding: 20px 18px;
        border: 1px solid #A7F3D0;
        color: #065F46;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bento-metric-white {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 20px 18px;
        border: 1px solid #E5E9E5;
        color: #1F2937;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bento-metric-sage {
        background: #ECFDF5;
        border-radius: 18px;
        padding: 20px 18px;
        border: 1px solid #D1FAE5;
        color: #047857;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .bento-metric-val {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .bento-metric-lbl {
        font-size: 12px;
        font-weight: 700;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .glyph-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        border: 2px solid currentColor;
        display: inline-block;
    }

    /* Pill Badges */
    .bento-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    /* Buttons & Interactive Elements */
    .stButton > button {
        border-radius: 14px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        border: 1px solid #16A34A !important;
        background-color: #16A34A !important;
        color: #FFFFFF !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        line-height: 1.4 !important;
    }
    .stButton > button:hover {
        background-color: #15803D !important;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.3) !important;
    }
    .stDownloadButton > button {
        border-radius: 9999px !important;
        padding: 10px 28px !important;
        font-weight: 700 !important;
        background: #111827 !important;
        color: #FFFFFF !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #1F2937 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
    }

    /* Rounded File Uploader */
    div[data-testid="stFileUploader"] {
        border-radius: 16px;
        background: #F9FAF9;
        padding: 14px;
        border: 1px dashed #A7F3D0;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #10B981, #FBBF24, #F97316, #EF4444);
        border-radius: 9999px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E9E5;
    }
</style>
""", unsafe_allow_html=True)


# ── Domain Specifications & Tiers ─────────────────────────────────────────────


BENCHMARK_20_SPECIMENS = {
    "[Banana] 1. Unripe (Firm Green Peel)": ("sample_images/banana_unripe_sample.jpg", "banana", "Unripe"),
    "[Banana] 2. Ripe (Optimal Yellow Table)": ("sample_images/banana_ripe_sample.jpg", "banana", "Ripe"),
    "[Banana] 3. Ripe Cluster (Harvest Bundle)": ("sample_images/banana_ripe_bundle_sample.jpg", "banana", "Ripe"),
    "[Banana] 4. Overripe (High Sugar Spotting)": ("sample_images/banana_overripe_sample.jpg", "banana", "Overripe"),
    "[Banana] 5. Spoiled (Fermented Bio-Waste)": ("sample_images/banana_spoiled_sample.jpg", "banana", "Spoiled"),

    "[Tomato] 1. Unripe (Mature Green)": ("sample_images/tomato_unripe_sample.jpg", "tomato", "Unripe"),
    "[Tomato] 2. Ripe (Fresh Table Standard)": ("sample_images/tomato_ripe_sample.jpg", "tomato", "Ripe"),
    "[Tomato] 3. Ripe (Vine Harvest Cluster)": ("sample_images/tomato_vine_ripe_sample.jpg", "tomato", "Ripe"),
    "[Tomato] 4. Overripe (Soft Cannery Processing)": ("sample_images/tomato_overripe_sample.jpg", "tomato", "Overripe"),
    "[Tomato] 5. Defect / Damaged (Surface Puncture)": ("sample_images/tomato_damaged_sample.jpg", "tomato", "Damaged"),

    "[Strawberry] 1. Unripe (Green Calyx)": ("sample_images/strawberry_unripe_sample.jpg", "strawberry", "Unripe"),
    "[Strawberry] 2. Partially Ripe (Breaking Pink)": ("sample_images/strawberry_partially_ripe_sample.jpg", "strawberry", "Partially Ripe"),
    "[Strawberry] 3. Ripe (Fresh Table Punnet)": ("sample_images/strawberry_ripe_sample.jpg", "strawberry", "Ripe"),
    "[Strawberry] 4. Sweet Ripe (Peak Sugar)": ("sample_images/strawberry_sweet_ripe_sample.jpg", "strawberry", "Ripe"),
    "[Strawberry] 5. Rotten (Botrytis Mold Decay)": ("sample_images/strawberry_rotten_sample.jpg", "strawberry", "Rotten"),

    "[Avocado] 1. Unripe (Hard Green Export)": ("sample_images/avocado_unripe_sample.jpg", "avocado", "Unripe"),
    "[Avocado] 2. Partially Ripe (Breaking Skin)": ("sample_images/avocado_partially_ripe_sample.jpg", "avocado", "Partially Ripe"),
    "[Avocado] 3. Ripe (Soft Yield Table)": ("sample_images/avocado_ripe_sample.jpg", "avocado", "Ripe"),
    "[Avocado] 4. Hass Ripe (Dark Pebbled Skin)": ("sample_images/avocado_hass_ripe_sample.jpg", "avocado", "Ripe"),
    "[Avocado] 5. Rotten (Rancid Oxidation)": ("sample_images/avocado_rotten_sample.jpg", "avocado", "Rotten"),
}

FRUIT_SPECS = {
    "banana": {
        "name": "Banana",
        "classes": BANANA_CLASSES,
        "num_classes": 4,
        "max_rank": 3.0,
        "colors": ["#10B981", "#F59E0B", "#F97316", "#EF4444"],
    },
    "tomato": {
        "name": "Tomato",
        "classes": TOMATO_ORDINAL_CLASSES,
        "num_classes": 3,
        "max_rank": 2.0,
        "colors": ["#10B981", "#EC4899", "#E11D48"],
    },
    "strawberry": {
        "name": "Strawberry",
        "classes": STRAWBERRY_CLASSES,
        "num_classes": 4,
        "max_rank": 3.0,
        "colors": ["#10B981", "#F59E0B", "#E11D48", "#854D0E"],
    },
    "avocado": {
        "name": "Avocado",
        "classes": AVOCADO_CLASSES,
        "num_classes": 4,
        "max_rank": 3.0,
        "colors": ["#10B981", "#84CC16", "#3F6212", "#3E2723"],
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
                "color": "#EF4444",
                "bg_color": "#FEE2E2",
                "badge": "BIO-WASTE REJECT",
                "allocation": "Divert to composting / anaerobic biogas digestion.",
                "price_index": "0% (Economic Loss)",
                "action": "Immediate segregation to prevent ethylene contamination of crate.",
                "marketable": False,
            }
        elif pred_name == "overripe":
            return {
                "tier": "PROCESSING GRADE (Industrial Value)",
                "color": "#D97706",
                "bg_color": "#FEF3C7",
                "badge": "INDUSTRIAL VALUE",
                "allocation": "Route to commercial bakeries, smoothies, baby food puree.",
                "price_index": "40% of baseline",
                "action": "Immediate cold-chain transfer to food processing facilities.",
                "marketable": True,
            }
        elif pred_name == "ripe":
            return {
                "tier": "CLASS 2 (Table Consumption / Retail)",
                "color": "#2563EB",
                "bg_color": "#DBEAFE",
                "badge": "DIRECT CONSUMER RETAIL",
                "allocation": "Immediate display in grocery retail produce aisle.",
                "price_index": "85% of baseline",
                "action": "Target sale window: 48 – 72 hours.",
                "marketable": True,
            }
        else:  # unripe
            return {
                "tier": "CLASS 1 / EXTRA (Premium Export Grade)",
                "color": "#16A34A",
                "bg_color": "#DCFCE7",
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
                "color": "#EF4444",
                "bg_color": "#FEE2E2",
                "badge": "PHYSICAL DEFECT REJECT",
                "allocation": "Discard due to structural puncture, bruising, or fungal rot.",
                "price_index": "0% (Economic Loss)",
                "action": "Segregate immediately to prevent mold propagation across crate.",
                "marketable": False,
            }
        elif pred_name == "overripe":
            return {
                "tier": "PROCESSING GRADE (Industrial Cannery)",
                "color": "#D97706",
                "bg_color": "#FEF3C7",
                "badge": "CANNERY PROCESSING",
                "allocation": "Route to industrial paste, marinara, soup, or puree processing.",
                "price_index": "45% of baseline",
                "action": "Wash and process within 24 hours.",
                "marketable": True,
            }
        elif pred_name == "ripe":
            return {
                "tier": "CLASS 1 (Fresh Market Standard)",
                "color": "#DB2777",
                "bg_color": "#FCE7F3",
                "badge": "FRESH TABLE STANDARD",
                "allocation": "Supermarket produce section for salad & fresh table use.",
                "price_index": "100% (Full Retail)",
                "action": "Store at 12°C–15°C; do not chill below 10°C.",
                "marketable": True,
            }
        else:  # unripe
            return {
                "tier": "CLASS 1 (Green Shipping Maturity)",
                "color": "#16A34A",
                "bg_color": "#DCFCE7",
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
                "color": "#EF4444",
                "bg_color": "#FEE2E2",
                "badge": "MOLD REJECT",
                "allocation": "Discard; Botrytis / fungal decay risk.",
                "price_index": "0%",
                "action": "Remove to avoid mold spread in punnets.",
                "marketable": False,
            }
        elif pred_name == "unripe":
            return {
                "tier": "PROCESSING GRADE (Acidic / Confectionery)",
                "color": "#16A34A",
                "bg_color": "#DCFCE7",
                "badge": "JAM / EXTRACTION",
                "allocation": "Jam, confectionery pectin, or fruit tart acidifier.",
                "price_index": "50%",
                "action": "Process or blend with high-sugar purees.",
                "marketable": True,
            }
        elif pred_name == "partially_ripe":
            return {
                "tier": "CLASS 2 (Local Distribution)",
                "color": "#D97706",
                "bg_color": "#FEF3C7",
                "badge": "REGIONAL MARKET",
                "allocation": "Fast-moving short-distance retail distribution.",
                "price_index": "80%",
                "action": "Refrigerate at 2°C–4°C.",
                "marketable": True,
            }
        else:  # ripe
            return {
                "tier": "CLASS 1 / EXTRA (Premium Fresh Punnet)",
                "color": "#E11D48",
                "bg_color": "#FFE4E6",
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
                "color": "#EF4444",
                "bg_color": "#FEE2E2",
                "badge": "NON-MARKETABLE",
                "allocation": "Divert to composting / bio-refinery.",
                "price_index": "0%",
                "action": "Discard safely.",
                "marketable": False,
            }
        elif pred_name == "unripe":
            return {
                "tier": "CLASS 1 / HARD (Long Export Tier)",
                "color": "#16A34A",
                "bg_color": "#DCFCE7",
                "badge": "HARD EXPORT TIER",
                "allocation": "International refrigerated container export.",
                "price_index": "100%",
                "action": "Store at 5°C–7°C; long shelf-life buffer.",
                "marketable": True,
            }
        elif pred_name == "partially_ripe":
            return {
                "tier": "CLASS 1 (Breaking / Near Retail)",
                "color": "#65A30D",
                "bg_color": "#ECFCCB",
                "badge": "BREAKING STAGE",
                "allocation": "Supermarket shelf staging for 2–3 days ripening.",
                "price_index": "90%",
                "action": "Hold at ambient temperature to soften.",
                "marketable": True,
            }
        else:  # ripe
            return {
                "tier": "READY-TO-EAT (Peak Flavor)",
                "color": "#15803D",
                "bg_color": "#DCFCE7",
                "badge": "READY-TO-EAT RETAIL",
                "allocation": "Restaurant / grocery 'Ripe & Ready' premium display.",
                "price_index": "100% (High Margin)",
                "action": "Consume or refrigerate immediately to hold texture.",
                "marketable": True,
            }

    # Generic fallback
    return {
        "tier": "COMMERCIAL STANDARD",
        "color": "#16A34A",
        "bg_color": "#DCFCE7",
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
    st.markdown('''
    <div class="sidebar-brand-header">
        <div class="fruitscan-sidebar-logo">
            <span class="fruitscan-brand-logo">Fruit<span class="fruitscan-brand-accent">Scan</span><span class="fruitscan-brand-dot">.</span></span>
        </div>
        <div class="fruitscan-sidebar-tagline">Fruit Ripeness Grading via Ordinal Regression</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("### Operating Mode")
    app_mode = st.radio(
        "Select Operating Mode",
        options=["Single Specimen Diagnostic", "Bulk Crate / Conveyor Batch QC"],
        index=0,
        help="Switch between detailed single-image inspection and bulk packhouse carton auditing."
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


# ── Floating Navbar & Hero Banner (Nanoplazz Bento Style) ───────────────────

nav_mode_label = "Single Diagnostic" if app_mode == "Single Specimen Diagnostic" else "Batch QC Crate"
crop_name = FRUIT_SPECS[fruit]['name']

st.markdown(f"""
<div class="bento-nav">
    <div class="bento-nav-brand">
        <span class="fruitscan-brand-logo">Fruit<span class="fruitscan-brand-accent">Scan</span><span class="fruitscan-brand-dot">.</span></span>
        <span class="bento-brand-sep" style="color:#D1D5DB; margin: 0 4px; font-weight:300;">•</span>
        <span class="bento-brand-sub" style="font-size:13px; font-weight:600; color:#4B5563;">Fruit Ripeness Grading <span style="font-weight:600; color:#16A34A;">via Ordinal Regression</span></span>
    </div>
    <div class="bento-nav-links">
        <span class="bento-nav-pill-active">{nav_mode_label}</span>
        <span style="color:#9CA3AF;">•</span>
        <span>{crop_name} Calibration</span>
        <span style="color:#9CA3AF;">•</span>
        <span>USDA / UNECE Standards</span>
    </div>
    <div class="bento-nav-badge">
        <span style="width:7px; height:7px; border-radius:50%; background:#16A34A; display:inline-block;"></span>
        <span>MobileNetV2 • Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

if app_mode == "Single Specimen Diagnostic":
    st.markdown("""
    <div class="bento-hero">
        <div class="bento-hero-badge">Deep Learning • Ordinal Neural Regression</div>
        <div class="bento-hero-title">Fruit Ripeness Grading <span>via Ordinal Regression</span></div>
        <div class="bento-hero-subtitle">A Convolutional Neural Network (CNN) architecture learning visual features—color, texture, and shape—across natural biological ripeness orders (unripe → ripe → overripe).</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="bento-hero">
        <div class="bento-hero-badge" style="background:#E0E7FF; color:#4338CA; border-color:#C7D2FE;">Batch Evaluation & Quality Control</div>
        <div class="bento-hero-title">Fruit Ripeness Grading — <span>Conveyor Batch QC</span></div>
        <div class="bento-hero-subtitle">Automated high-throughput crate inspection evaluating lot acceptance thresholds and commercial distribution tiers.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Nanoplazz-style Signature Top Bento Grid ──────────────────────────────────
st.markdown(f"""
<div class="bento-top-grid">
    <div class="bento-top-mint">
        <div class="bento-top-val"><span class="glyph-dot"></span> 99.3%</div>
        <div class="bento-top-lbl">Benchmark Accuracy (Banana)</div>
    </div>
    <div class="bento-top-white">
        <div class="bento-top-val"><span class="glyph-dot"></span> 13.9 ms</div>
        <div class="bento-top-lbl">Batch Latency (73 FPS)</div>
    </div>
    <div class="bento-top-sage">
        <div class="bento-top-val"><span class="glyph-dot"></span> 0.12 MAE</div>
        <div class="bento-top-lbl">Mean Ordinal Rank Error</div>
    </div>
    <div class="bento-top-white" style="border-left: 4px solid #16A34A;">
        <div class="bento-top-val" style="color:#16A34A; font-size:22px;">{crop_name.upper()}</div>
        <div class="bento-top-lbl">Active Calibrated Model</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Load model
model, gradcam, device = load_model(fruit)

if model is None:
    st.warning(f"Checkpoint `models/{fruit}_best.pth` is currently training or pending. Please select an active trained model.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE SPECIMEN DIAGNOSTIC (BENTO BOX LAYOUT)
# ═════════════════════════════════════════════════════════════════════════════

if app_mode == "Single Specimen Diagnostic":
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-card-title">Specimen Input & Inspection</div>
            <div class="bento-card-desc">Upload a produce photo or pick a pre-loaded benchmark test sample.</div>
        """, unsafe_allow_html=True)

        # Dual Input Selector: Upload Custom Image vs. 20 Pre-Loaded Benchmark Library
        input_mode = st.radio(
            "Produce Input Method",
            options=["Select from 20 Pre-Loaded Benchmark Specimens", "Upload Your Own Produce Photo"],
            index=0,
            horizontal=True,
            help="Pick one of the 20 pre-loaded benchmark specimens across all 4 fruits, or upload your own photo."
        )

        image = None
        active_fruit = fruit

        if input_mode == "Upload Your Own Produce Photo":
            # Feature 1: Original File Uploader (Untouched!)
            uploaded_file = st.file_uploader(
                "Upload fruit photo (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"uploader_{fruit}",
                help="Drag and drop or browse an image file from your computer.",
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                image = ImageOps.exif_transpose(image)
                st.caption(f"Uploaded produce specimen: `{uploaded_file.name}`")
        else:
            # Feature 2: 20 Pre-Loaded Benchmark Specimens Library
            col_filter1, col_filter2 = st.columns([1, 2])
            with col_filter1:
                category_filter = st.selectbox(
                    "Crop Filter",
                    options=["All Crops (20 Specimens)", "Banana (5)", "Tomato (5)", "Strawberry (5)", "Avocado (5)"],
                    index=0,
                )
            
            if "Banana" in category_filter:
                bench_keys = [k for k in BENCHMARK_20_SPECIMENS if "[Banana]" in k]
            elif "Tomato" in category_filter:
                bench_keys = [k for k in BENCHMARK_20_SPECIMENS if "[Tomato]" in k]
            elif "Strawberry" in category_filter:
                bench_keys = [k for k in BENCHMARK_20_SPECIMENS if "[Strawberry]" in k]
            elif "Avocado" in category_filter:
                bench_keys = [k for k in BENCHMARK_20_SPECIMENS if "[Avocado]" in k]
            else:
                bench_keys = list(BENCHMARK_20_SPECIMENS.keys())

            with col_filter2:
                selected_sample = st.selectbox(
                    "Choose from 20 Pre-Loaded Benchmark Images",
                    options=bench_keys,
                    index=0,
                    help="Select any of the 20 calibrated benchmark produce images across all ripeness stages."
                )

            img_path, sample_fruit, ground_truth = BENCHMARK_20_SPECIMENS[selected_sample]
            active_fruit = sample_fruit
            if Path(img_path).exists():
                image = Image.open(img_path).convert("RGB")
                image = ImageOps.exif_transpose(image)
                st.caption(f"Loaded Benchmark: **{selected_sample}** • Ground-Truth Stage: `{ground_truth}`")

        if image is not None:
            st.image(image, caption="Specimen for Analysis", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        if image is not None:
            # Dynamically select calibrated model for active specimen
            eval_fruit = active_fruit if 'active_fruit' in locals() else fruit
            eval_model, eval_gradcam, _ = load_model(eval_fruit)
            if eval_model is None:
                eval_model, eval_gradcam = model, gradcam
                eval_fruit = fruit

            # Preprocessing & Fast Vectorized Inference
            transform = get_eval_transforms()
            input_tensor = transform(image).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                corn_logits, damage_logits = eval_model(input_tensor)
                pred_label = corn_label_from_logits(corn_logits).item()
            latency_ms = (time.perf_counter() - t0) * 1000

            spec = FRUIT_SPECS[eval_fruit]
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
            if eval_fruit == "tomato" and damage_logits is not None:
                damage_prob = torch.sigmoid(damage_logits).item()
                is_damaged = damage_prob > 0.5

            # Commercial Grade Standard Tier
            tier_info = get_commercial_packhouse_tier(eval_fruit, pred_name, expected_rank, is_damaged)

            # ── BENTO TILE 1: Key Performance Metrics Trio ───────────────────
            st.markdown(f"""
            <div class="bento-metric-grid">
                <div class="bento-metric-mint">
                    <div class="bento-metric-val"><span class="glyph-dot"></span> {confidence:.1%}</div>
                    <div class="bento-metric-lbl">Model Confidence</div>
                </div>
                <div class="bento-metric-white">
                    <div class="bento-metric-val" style="color: {tier_info['color']};">{pred_name.replace('_', ' ').upper()}</div>
                    <div class="bento-metric-lbl">Ripeness Stage</div>
                </div>
                <div class="bento-metric-sage">
                    <div class="bento-metric-val">{expected_rank:.2f} <span style="font-size:16px;font-weight:600;color:#6B7280;">/ {max_rank:.1f}</span></div>
                    <div class="bento-metric-lbl">Continuous Index</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── BENTO TILE 2: Monotonic Ripeness Progression Bar ──────────────
            st.markdown("""
            <div class="bento-card">
                <div class="bento-card-title">Ripeness Progression Scale</div>
                <div class="bento-card-desc">Monotonic biological continuum mapped via Conditional Ordinal Regression.</div>
            """, unsafe_allow_html=True)

            normalized_cri = expected_rank / max_rank
            st.progress(min(max(normalized_cri, 0.0), 1.0))

            scale_pills = []
            for i, cls in enumerate(classes):
                is_pred = (i == pred_label)
                color = colors[i]
                cls_display = cls.replace('_', ' ').capitalize()
                if is_pred:
                    scale_pills.append(
                        f'<div style="flex:1; min-width:80px; background-color:{color}; color:#FFFFFF; '
                        f'padding:10px 4px; border-radius:12px; text-align:center; font-weight:700; '
                        f'font-size:12px; box-shadow:0 3px 10px {color}44; white-space:nowrap; '
                        f'overflow:hidden; text-overflow:ellipsis;" title="{cls_display}">{cls_display.upper()}</div>'
                    )
                else:
                    scale_pills.append(
                        f'<div style="flex:1; min-width:80px; background-color:#F3F4F6; color:#4B5563; '
                        f'padding:10px 4px; border-radius:12px; text-align:center; font-size:11.5px; font-weight:600; '
                        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="{cls_display} ({probs[i]:.0%})">{cls_display} ({probs[i]:.0%})</div>'
                    )

            st.markdown(f'<div style="display:flex; gap:8px; margin-top:14px; width:100%; flex-wrap:wrap;">{"".join(scale_pills)}</div></div>', unsafe_allow_html=True)

            # ── BENTO TILE 3: Commercial Packhouse Standards ──────────────────
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-card-title">Commercial Packhouse Standard (USDA / UNECE)</div>
                <div class="bento-card-desc">Trade-aligned classification and post-harvest supply chain allocation.</div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
                    <div class="bento-pill" style="background:{tier_info['bg_color']}; color:{tier_info['color']}; font-size:14px; padding:8px 18px;">
                        {tier_info['tier']}
                    </div>
                    <div style="font-weight:700; font-size:14px; color:#1F2937;">
                        Price Index: <span style="color:#16A34A; background:#DCFCE7; padding:4px 10px; border-radius:8px;">{tier_info['price_index']}</span>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; background:#F9FAF9; padding:16px; border-radius:14px; border:1px solid #E5E7EB;">
                    <div>
                        <div style="font-size:12px; font-weight:700; color:#6B7280; text-transform:uppercase; margin-bottom:4px;">Operational Protocol</div>
                        <div style="font-size:14px; color:#1F2937; line-height:1.4;">{tier_info['action']}</div>
                    </div>
                    <div>
                        <div style="font-size:12px; font-weight:700; color:#6B7280; text-transform:uppercase; margin-bottom:4px;">Supply Chain Allocation</div>
                        <div style="font-size:14px; color:#1F2937; line-height:1.4;">{tier_info['allocation']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Tomato Damage Head Result
            if eval_fruit == "tomato" and damage_logits is not None:
                if is_damaged:
                    st.markdown(f"""
                    <div class="bento-card" style="border-left: 6px solid #EF4444; background:#FEF2F2;">
                        <div class="bento-card-title" style="color:#991B1B;">Quality Flag: Physical Damage Detected ({damage_prob:.1%})</div>
                        <div style="font-size:14px; color:#7F1D1D;">Surface puncture, cracking, or lesion observed. Specimen flagged for immediate diversion from table carton.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bento-card" style="border-left: 6px solid #10B981; background:#F0FDF4;">
                        <div class="bento-card-title" style="color:#166534;">Quality Flag: Structurally Sound ({(1-damage_prob):.1%})</div>
                        <div style="font-size:14px; color:#14532D;">No surface cracks or physical defects observed. Suitable for standard packhouse carton packing.</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── BENTO TILE 4: Shelf-Life & Storage Guidance ───────────────────
            shelf_spec = SHELF_LIFE_SPECS.get(eval_fruit, {}).get(pred_name, {"days": "N/A", "action": "N/A"})
            st.markdown(f"""
            <div class="bento-card">
                <div class="bento-card-title">Shelf-Life & Storage Protocol</div>
                <div class="bento-card-desc">Optimal temperature buffering and inventory window.</div>
                <div style="display:grid; grid-template-columns: 1fr 2fr; gap:16px; align-items:center;">
                    <div style="background:#F3F4F6; padding:16px; border-radius:14px; text-align:center;">
                        <div style="font-size:12px; font-weight:700; color:#6B7280; text-transform:uppercase;">Estimated Window</div>
                        <div style="font-size:22px; font-weight:800; color:#111827; margin-top:4px;">{shelf_spec['days']}</div>
                    </div>
                    <div style="font-size:14px; color:#374151; line-height:1.5;">
                        {shelf_spec['action']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── BENTO TILE 5: Grad-CAM Explainability Studio ──────────────────
            st.markdown("""
            <div class="bento-card">
                <div class="bento-card-title">Explainable AI (Grad-CAM Studio)</div>
                <div class="bento-card-desc">Gradient-weighted Class Activation Map targeting MobileNetV2 final conv layer.</div>
            """, unsafe_allow_html=True)

            with torch.enable_grad():
                cam = eval_gradcam.generate(input_tensor, target_class=pred_label)

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

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="bento-card" style="text-align:center; padding:48px 24px;">
                <div style="font-size:32px; color:#10B981; margin-bottom:12px;">○</div>
                <div class="bento-card-title">Awaiting Specimen Input</div>
                <div class="bento-card-desc" style="max-width:380px; margin:0 auto;">
                    Upload a fruit photo or select a benchmark sample from the panel on the left to generate real-time diagnostic grading.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2: BULK CRATE / CONVEYOR BATCH QC (BENTO BOX LAYOUT)
# ═════════════════════════════════════════════════════════════════════════════

else:
    batch_col1, batch_col2 = st.columns([1, 2], gap="large")

    with batch_col1:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-card-title">Batch Configuration</div>
            <div class="bento-card-desc">Upload multiple crate photos or load a 16-unit carton simulation.</div>
        """, unsafe_allow_html=True)

        uploaded_batch = st.file_uploader(
            "Upload Batch Images (multiple files)",
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

        st.markdown("</div>", unsafe_allow_html=True)

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
            fps = total_count / (total_batch_time / 1000) if total_batch_time > 0 else 0

            # ── BENTO TILE 1: High-Speed Batch Telemetry ─────────────────────
            st.markdown(f"""
            <div class="bento-metric-grid">
                <div class="bento-metric-mint">
                    <div class="bento-metric-val"><span class="glyph-dot"></span> {acceptance_pct:.1f}%</div>
                    <div class="bento-metric-lbl">Lot Acceptance Rate</div>
                </div>
                <div class="bento-metric-white">
                    <div class="bento-metric-val">{total_count} <span style="font-size:16px;color:#6B7280;">units</span></div>
                    <div class="bento-metric-lbl">Total Inspected</div>
                </div>
                <div class="bento-metric-sage">
                    <div class="bento-metric-val">{total_batch_time:.0f} ms <span style="font-size:14px;color:#16A34A;font-weight:700;">({fps:.1f} FPS)</span></div>
                    <div class="bento-metric-lbl">Batch Latency</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── BENTO TILE 2: Lot Acceptance Verdict Banner ──────────────────
            if acceptance_pct >= 85.0:
                st.markdown(f"""
                <div class="bento-card" style="border-left:6px solid #16A34A; background:#F0FDF4; padding:20px;">
                    <div style="font-size:16px; font-weight:800; color:#166534; margin-bottom:4px;">
                        LOT VERDICT: ACCEPTED FOR COMMERCIAL DISTRIBUTION
                    </div>
                    <div style="font-size:14px; color:#14532D;">
                        {acceptance_pct:.1f}% marketable units exceeds the required 85% packhouse quality threshold.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bento-card" style="border-left:6px solid #EF4444; background:#FEF2F2; padding:20px;">
                    <div style="font-size:16px; font-weight:800; color:#991B1B; margin-bottom:4px;">
                        LOT VERDICT: FLAGGED FOR MANUAL RE-SORTING
                    </div>
                    <div style="font-size:14px; color:#7F1D1D;">
                        {reject_pct:.1f}% defect/reject ratio exceeds the permitted 15% tolerance. Carton diverted for line re-inspection.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── BENTO TILE 3: Grade Breakdown & Audit Trail ───────────────────
            st.markdown("""
            <div class="bento-card">
                <div class="bento-card-title">Commercial Grade Distribution</div>
                <div class="bento-card-desc">Aggregate breakdown of cartons by commercial grade classification.</div>
            """, unsafe_allow_html=True)

            grade_dist = df_results["Commercial Tier"].value_counts().reset_index()
            grade_dist.columns = ["Commercial Tier", "Unit Count"]
            grade_dist["Percentage"] = (grade_dist["Unit Count"] / total_count * 100).apply(lambda x: f"{x:.1f}%")
            st.dataframe(grade_dist, use_container_width=True)

            st.markdown("""
                <div style="margin-top:20px;">
                    <div class="bento-card-title">Individual Specimen Audit Trail</div>
                    <div class="bento-card-desc">Traceable per-specimen diagnostics with Continuous Ripeness Index and Quality Status.</div>
                </div>
            """, unsafe_allow_html=True)

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

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="bento-card" style="text-align:center; padding:48px 24px;">
                <div style="font-size:32px; color:#4338CA; margin-bottom:12px;">○</div>
                <div class="bento-card-title">Awaiting Batch Input</div>
                <div class="bento-card-desc" style="max-width:400px; margin:0 auto;">
                    Upload multiple fruit photos or click "Load Simulated Packhouse Carton" on the left to initiate high-throughput batch grading.
                </div>
            </div>
            """, unsafe_allow_html=True)
