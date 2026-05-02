"""PBC competing-risks loader and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import choose_column, dataset_dict, factorize_binary, require_file, split_indices


ALIASES: Dict[str, Sequence[str]] = {
    "time": ["time", "futime", "days"],
    "status": ["status", "event", "delta"],
    "drug": ["drug", "treatment"],
    "sex": ["sex"],
    "ascites": ["ascites"],
    "hepatomegaly": ["hepatomegaly", "hepato"],
    "spiders": ["spiders"],
    "histologic": ["histologic", "stage"],
    "serBilir": ["serBilir", "bili", "bilirubin"],
    "serChol": ["serChol", "chol", "cholesterol"],
    "albumin": ["albumin"],
    "alkaline": ["alkaline", "alk.phos", "alk_phos", "alkphos"],
    "SGOT": ["SGOT", "ast"],
    "platelets": ["platelets", "platelet"],
    "prothrombin": ["prothrombin", "protime"],
    "age": ["age"],
    "edema": ["edema"],
}


BINARY = ["drug", "sex", "ascites", "hepatomegaly", "spiders"]
CONTINUOUS = [
    "serBilir",
    "serChol",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "age",
]


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, aliases in ALIASES.items():
        col = choose_column(df.columns, aliases, name)
        out[name] = df[col]
    return out


def _map_status(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy()
    if not np.issubdtype(arr.dtype, np.number):
        cleaned = values.astype(str).str.strip().str.lower()
        mapping = {
            "censored": 0,
            "alive": 0,
            "0": 0,
            "death": 1,
            "dead": 1,
            "2": 1,
            "transplanted": 2,
            "transplant": 2,
            "1": 2,
        }
        return cleaned.map(mapping).fillna(cleaned.astype("category").cat.codes).to_numpy(dtype=int)
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=int)
    unique = set(np.unique(numeric).tolist())
    if unique.issubset({0, 1, 2}):
        # Mayo PBC convention is 0=censored, 1=transplant, 2=death.
        return np.where(numeric == 2, 1, np.where(numeric == 1, 2, 0)).astype(int)
    return numeric.astype(int)


def _mode(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return 0.0
    vals, counts = np.unique(clean, return_counts=True)
    return float(vals[np.argmax(counts)])


def _build_features(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    parts_train: List[np.ndarray] = []
    parts_test: List[np.ndarray] = []
    names: List[str] = []

    for col in BINARY:
        values = factorize_binary(df[col].to_numpy())
        fill = _mode(values[train_idx])
        values = np.where(np.isnan(values), fill, values)
        parts_train.append(values[train_idx, None].astype(float))
        parts_test.append(values[test_idx, None].astype(float))
        names.append(col)

    hist = pd.to_numeric(df["histologic"], errors="coerce").to_numpy(dtype=float)
    fill = np.nanmean(hist[train_idx])
    hist = np.where(np.isnan(hist), fill, hist)
    hist = (hist - 1.0) / 3.0
    parts_train.append(hist[train_idx, None])
    parts_test.append(hist[test_idx, None])
    names.append("histologic_scaled")

    for col in CONTINUOUS:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mean = np.nanmean(values[train_idx])
        values = np.where(np.isnan(values), mean, values)
        std = np.std(values[train_idx])
        if std < 1e-8:
            std = 1.0
        values = (values - mean) / std
        parts_train.append(values[train_idx, None])
        parts_test.append(values[test_idx, None])
        names.append(col)

    edema = pd.to_numeric(df["edema"], errors="coerce").to_numpy(dtype=float)
    fill = _mode(edema[train_idx])
    edema = np.where(np.isnan(edema), fill, edema)
    categories = [0.0, 0.5, 1.0]
    for category in categories:
        parts_train.append((edema[train_idx] == category).astype(float)[:, None])
        parts_test.append((edema[test_idx] == category).astype(float)[:, None])
        names.append(f"edema_{category:g}")

    return np.concatenate(parts_train, axis=1), np.concatenate(parts_test, axis=1), names


def load_pbc(path: str | Path, test_size: float = 0.3, seed: int = 42):
    """Load PBC data with the preprocessing described in the blueprint."""
    expected = [alias[0] for alias in ALIASES.values()]
    csv_path = require_file(path, "PBC", expected)
    raw = pd.read_csv(csv_path)
    df = _canonicalize(raw)
    train_idx, test_idx = split_indices(len(df), test_size=test_size, seed=seed)
    x_train, x_test, feature_names = _build_features(df, train_idx, test_idx)
    y = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(y)):
        raise ValueError("PBC time column contains non-numeric values after coercion")
    delta = _map_status(df["status"])
    return dataset_dict(
        x_train,
        y[train_idx],
        delta[train_idx],
        x_test,
        y[test_idx],
        delta[test_idx],
        num_causes=2,
        feature_names=feature_names,
        dataset_name="pbc",
        source_path=str(csv_path),
    )
