from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import pairwise_distances
from torchvision import models

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = {0: "Cat", 1: "Dog"}

MODEL_INFO = {
    "ResNet-50": ("resnet50", models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "MobileNetV3-Large": ("mobilenet_v3_large", models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, "classifier"),
    "ViT-B/16": ("vit_b_16", models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, "heads"),
}


def replace_head(model, head_name):
    setattr(model, head_name, nn.Identity())
    return model


@st.cache_resource(show_spinner="Loading encoder...")
def load_model(display_name):
    key, builder, weights, head_name = MODEL_INFO[display_name]
    model = builder(weights=weights)
    model = replace_head(model, head_name)
    model.eval().to(DEVICE)
    return key, model, weights.transforms()


@st.cache_data
def load_gallery(encoder_key):
    emb = np.load(ROOT / "artifacts" / f"{encoder_key}_train_embeddings.npy")
    labels = np.load(ROOT / "artifacts" / f"{encoder_key}_train_labels.npy")
    paths = json.loads((ROOT / "artifacts" / f"{encoder_key}_train_paths.json").read_text(encoding="utf-8"))
    return emb, labels, paths


def encode_query(image, model, transform):
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        z = model(x)
        z = F.normalize(z, p=2, dim=1)
    return z.cpu().numpy()[0]


def metric_name(metric):
    return {"Euclidean (L2)": "euclidean", "Manhattan (L1)": "manhattan", "Cosine distance": "cosine"}[metric]


st.set_page_config(page_title="STAT6207 Image Retrieval", layout="wide")
st.title("STAT6207 Assignment 1 — Cat/Dog Image Retrieval + KNN")
st.caption("Pre-trained visual encoder → embedding → distance → KNN")

RESULTS_DIR = ROOT / "results"


def read_csv_safe(path: Path):
    """Read a results CSV; return None if it is missing or unreadable."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def registry_default_k(encoder_key: str, metric: str):
    """Return the K stored for this exact encoder+metric in k_registry.json, else None."""
    path = ROOT / "artifacts" / "k_registry.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8")).get(f"{encoder_key}::{metric}")
        return int(record["k"]) if record else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None


with st.sidebar:
    encoder_display = st.selectbox("Encoding model", list(MODEL_INFO.keys()))
    metric_display = st.selectbox("Distance", ["Euclidean (L2)", "Manhattan (L1)", "Cosine distance"])
    sidebar_encoder = MODEL_INFO[encoder_display][0]
    sidebar_metric = metric_name(metric_display)
    registry_k = registry_default_k(sidebar_encoder, sidebar_metric)
    if registry_k is not None and 1 <= registry_k <= 19:
        default_k = registry_k
        st.caption(f"Default K = {registry_k}, read from artifacts/k_registry.json "
                   f"(key {sidebar_encoder}::{sidebar_metric}). You can still change it manually.")
    else:
        default_k = 5
        st.caption("k_registry.json unavailable for this encoder/metric — using fallback K.")
    k = st.number_input("K", min_value=1, max_value=19, value=default_k, step=2)

uploaded = st.file_uploader("Upload a cat/dog image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to see the prediction, top-5 similar images, and top-5 dissimilar images.")
else:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Query image")
    st.image(image, width=320)

    encoder_key, model, transform = load_model(encoder_display)
    X, y, paths = load_gallery(encoder_key)
    metric = metric_name(metric_display)

    with st.spinner("Computing embedding and nearest neighbors..."):
        query = encode_query(image, model, transform)
        distances = pairwise_distances(query[None, :], X, metric=metric)[0]
        order = np.argsort(distances)
        topk = order[: int(k)]
        counts = np.bincount(y[topk], minlength=2)
        pred = int(np.argmax(counts))
        similar = order[:5]
        dissimilar = order[-5:][::-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", LABEL_NAMES[pred])
    c2.metric("K", int(k))
    c3.metric("Nearest-neighbor distance", f"{distances[order[0]]:.4f}")

    st.subheader("Top 5 most similar")
    cols = st.columns(5)
    for rank, (col, idx) in enumerate(zip(cols, similar), start=1):
        with col:
            st.image(ROOT / paths[idx], use_container_width=True)
            st.caption(f"#{rank} · {LABEL_NAMES[int(y[idx])]} · d={distances[idx]:.4f}")

    st.subheader("Top 5 most dissimilar")
    cols = st.columns(5)
    for rank, (col, idx) in enumerate(zip(cols, dissimilar), start=1):
        with col:
            st.image(ROOT / paths[idx], use_container_width=True)
            st.caption(f"#{rank} · {LABEL_NAMES[int(y[idx])]} · d={distances[idx]:.4f}")

    st.subheader("KNN neighbors used for prediction")
    neighbor_rows = []
    for rank, idx in enumerate(topk, start=1):
        neighbor_rows.append({
            "rank": rank,
            "label": LABEL_NAMES[int(y[idx])],
            "distance": float(distances[idx]),
            "image": paths[idx],
        })
    st.dataframe(neighbor_rows, use_container_width=True)

    st.warning("For a clean retrieval demonstration, upload an image that is not one of the training-gallery files.")

# ---------------------------------------------------------------------------
# Experiment Results — everything below is read directly from the real files
# in results/ and artifacts/. Nothing is hard-coded or fabricated.
# ---------------------------------------------------------------------------
st.divider()
st.header("Experiment Results")
st.caption("All values below are read from the actual files in results/ — none are typed in by hand.")

# --- Advanced results (3 encoders x 3 distance metrics) ---
st.subheader("Advanced results")
adv_csv = read_csv_safe(RESULTS_DIR / "advanced_results.csv")
if adv_csv is None:
    st.info("results/advanced_results.csv was not found or could not be read. Run advanced.py to create it.")
else:
    st.dataframe(adv_csv, use_container_width=True)

adv_png = RESULTS_DIR / "advanced_accuracy.png"
if adv_png.exists():
    st.image(str(adv_png), caption="advanced_accuracy.png")
else:
    st.info("results/advanced_accuracy.png was not found. Run advanced.py to create it.")

# --- Test evaluation on the 10 unseen test images ---
st.subheader("Test evaluation on the held-out test images")
evaluation_files = [
    "evaluation_resnet50.csv",
    "evaluation_mobilenet_v3_large.csv",
    "evaluation_vit_b_16.csv",
]
evaluation_frames = [
    frame for frame in (read_csv_safe(RESULTS_DIR / name) for name in evaluation_files)
    if frame is not None
]
if not evaluation_frames:
    st.info("No evaluation_*.csv files were found in results/.")
else:
    evaluation_table = pd.concat(evaluation_frames, ignore_index=True)
    st.dataframe(evaluation_table, use_container_width=True)
    st.caption(f"Combined {len(evaluation_table)} rows from {len(evaluation_frames)} file(s), "
               "including K and K-selection provenance.")

# --- Failed cases ---
st.subheader("Failed cases")
prediction_paths = sorted(RESULTS_DIR.glob("predictions_*.csv")) if RESULTS_DIR.exists() else []
prediction_frames = []
for path in prediction_paths:
    frame = read_csv_safe(path)
    if frame is not None and "correct" in frame.columns:
        frame = frame.copy()
        frame["source_file"] = path.name
        prediction_frames.append(frame)

if not prediction_frames:
    st.info("No readable predictions_*.csv files were found in results/.")
else:
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    failed_cases = all_predictions[all_predictions["correct"].astype(str).str.lower() == "false"]
    if failed_cases.empty:
        st.info("No failed cases found.")
    else:
        st.warning(f"{len(failed_cases)} failed case(s) found in the prediction files:")
        st.dataframe(failed_cases, use_container_width=True)

# --- Confusion matrices ---
st.subheader("Confusion matrices")
confusion_paths = sorted(RESULTS_DIR.glob("confusion_*.png")) if RESULTS_DIR.exists() else []
if not confusion_paths:
    st.info("No confusion_*.png files were found in results/.")
else:
    cm_cols = st.columns(3)
    for i, path in enumerate(confusion_paths):
        with cm_cols[i % 3]:
            st.image(str(path), caption=path.name)
