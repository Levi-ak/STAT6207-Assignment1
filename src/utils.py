from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

LABEL_NAMES = {0: "Cat", 1: "Dog"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_path(encoder: str, split: str, kind: str) -> Path:
    return project_root() / "artifacts" / f"{encoder}_{split}_{kind}"


def load_embeddings(encoder: str, split: str):
    root = project_root() / "artifacts"
    emb = np.load(root / f"{encoder}_{split}_embeddings.npy")
    labels = np.load(root / f"{encoder}_{split}_labels.npy")
    paths = json.loads((root / f"{encoder}_{split}_paths.json").read_text(encoding="utf-8"))
    return emb, labels, paths


def distances_to_gallery(query: np.ndarray, gallery: np.ndarray, metric: str) -> np.ndarray:
    if query.ndim == 1:
        query = query[None, :]
    return pairwise_distances(query, gallery, metric=metric)[0]


def knn_predict(gallery_labels: np.ndarray, distances: np.ndarray, k: int) -> tuple[int, np.ndarray]:
    order = np.argsort(distances)[:k]
    neighbor_labels = gallery_labels[order]
    counts = np.bincount(neighbor_labels, minlength=2)
    # K should normally be odd, but this deterministic tie-break uses the nearest neighbor.
    if counts[0] == counts[1]:
        pred = int(neighbor_labels[0])
    else:
        pred = int(np.argmax(counts))
    return pred, order


def ensure_dirs() -> None:
    root = project_root()
    for name in ("artifacts", "results"):
        (root / name).mkdir(exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# K selection with provenance (one record per encoder + distance metric)
# ---------------------------------------------------------------------------

DEFAULT_K_VALUES = (1, 3, 5, 7, 9, 11)
K_CV_FOLDS = 5
K_CV_SEED = 6207


def k_registry_path() -> Path:
    return project_root() / "artifacts" / "k_registry.json"


def k_registry_key(encoder: str, metric: str) -> str:
    return f"{encoder}::{metric}"


def load_k_registry() -> dict:
    path = k_registry_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def select_best_k(encoder: str, metric: str, k_values=DEFAULT_K_VALUES) -> dict:
    """Select K by 5-fold stratified CV on the *training/gallery* split only.

    The held-out test split is never loaded here, so K selection cannot see
    the 10 unseen images. The returned record is self-describing: it stores
    the encoder, metric, CV score, fold count, seed and searched K values.
    """
    X, y, _ = load_embeddings(encoder, "train")
    k_values = [int(v) for v in k_values]
    cv = StratifiedKFold(n_splits=K_CV_FOLDS, shuffle=True, random_state=K_CV_SEED)

    rows = []
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        rows.append(
            {
                "k": k,
                "mean_accuracy": float(scores.mean()),
                "std_accuracy": float(scores.std()),
            }
        )

    # Highest mean accuracy, then lower standard deviation, then smaller k.
    best = sorted(rows, key=lambda r: (-r["mean_accuracy"], r["std_accuracy"], r["k"]))[0]
    return {
        "encoder": encoder,
        "metric": metric,
        "k": int(best["k"]),
        "cv_mean_accuracy": float(best["mean_accuracy"]),
        "cv_std_accuracy": float(best["std_accuracy"]),
        "cv_folds": K_CV_FOLDS,
        "cv_seed": K_CV_SEED,
        "k_values_searched": k_values,
        "selection_split": "train",
        "held_out_test_used_for_selection": False,
        "n_train_samples": int(len(y)),
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "cv_rows": rows,
    }


def record_best_k(record: dict) -> Path:
    """Store one K record under its own encoder::metric key."""
    registry = load_k_registry()
    registry[k_registry_key(record["encoder"], record["metric"])] = record
    save_json(k_registry_path(), registry)
    return k_registry_path()


def resolve_best_k(encoder: str, metric: str, k_values=DEFAULT_K_VALUES) -> dict:
    """Return the stored K record for this exact encoder+metric.

    If none exists yet it is selected from train-only 5-fold CV and recorded,
    so a caller can never fall back to another encoder's or metric's K.
    """
    record = load_k_registry().get(k_registry_key(encoder, metric))
    if record is None:
        record = select_best_k(encoder, metric, k_values=k_values)
        record_best_k(record)
    return record
