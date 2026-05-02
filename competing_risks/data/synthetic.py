"""DeepHit-style synthetic competing-risks loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import (
    EPS,
    dataset_dict,
    rng_from_seed,
    sample_competing_risks,
    split_indices,
    standardize_from_train,
)


DEFAULT_SYNTHETIC_PATH = Path(__file__).with_name("synthetic_comprisk.csv")


def _generate_deephit_like(n: int = 20000, seed: int = 42, censoring_rate: float = 0.45):
    rng = rng_from_seed(seed)
    p = 12
    k = 2
    x = rng.normal(size=(n, p))
    beta1 = np.array([0.7, -0.6, 0.4, 0.3, -0.2, 0.1, 0.5, -0.4, 0.2, 0.0, 0.3, -0.2])
    beta2 = np.array([-0.5, 0.8, -0.3, 0.2, 0.4, -0.6, 0.1, 0.3, -0.2, 0.5, 0.0, 0.2])
    alpha = 0.28
    intercept = np.array([-3.2, -3.7])

    def mu_fn(x_arr, t):
        t_arr = np.asarray(t, dtype=float).reshape(-1, 1)
        nonlinear1 = np.sin(x_arr[:, 0]) + 0.3 * x_arr[:, 1] * x_arr[:, 2]
        nonlinear2 = np.cos(x_arr[:, 3]) - 0.25 * x_arr[:, 4] * x_arr[:, 5]
        eta1 = intercept[0] + x_arr @ beta1 + nonlinear1 + alpha * t_arr[:, 0]
        eta2 = intercept[1] + x_arr @ beta2 + nonlinear2 + alpha * t_arr[:, 0]
        return np.stack([eta1, eta2], axis=1)

    _, y, delta, _, censor_rate = sample_competing_risks(
        x, mu_fn, rng, target_censoring=censoring_rate, t_upper=40.0
    )
    return x, y + EPS, delta, censor_rate


def _read_csv(path: Path):
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    time_col = None
    event_col = None
    for candidate in ["time", "duration", "Y", "y"]:
        if candidate in df.columns:
            time_col = candidate
            break
    for candidate in ["event", "delta", "Delta", "status"]:
        if candidate in df.columns:
            event_col = candidate
            break
    if time_col is None or event_col is None:
        raise ValueError("Synthetic CSV must include time and event/delta columns")
    feature_cols = [
        c
        for c in df.columns
        if c not in {time_col, event_col} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(feature_cols) < 12:
        raise ValueError("Synthetic CSV must include at least 12 numeric feature columns")
    feature_cols = feature_cols[:12]
    x = df[feature_cols].to_numpy(dtype=float)
    y = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) + EPS
    delta = pd.to_numeric(df[event_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    return x, y, delta, feature_cols


def load_synthetic(
    path: Optional[str | Path] = None,
    test_size: float = 0.3,
    seed: int = 42,
    generate_if_missing: bool = True,
):
    """Load ``synthetic_comprisk.csv`` or generate a deterministic fallback."""
    csv_path = Path(path).expanduser() if path is not None else DEFAULT_SYNTHETIC_PATH
    loaded = None
    if csv_path.exists():
        loaded = _read_csv(csv_path)
    if loaded is None:
        if not generate_if_missing:
            raise FileNotFoundError(f"Synthetic data file not found or empty: {csv_path}")
        x, y, delta, censor_rate = _generate_deephit_like(n=20000, seed=seed)
        feature_names = [f"x{j}" for j in range(x.shape[1])]
        source = "generated"
    else:
        x, y, delta, feature_names = loaded
        censor_rate = float(np.mean(delta == 0))
        source = str(csv_path)

    train_idx, test_idx = split_indices(len(y), test_size=test_size, seed=seed)
    x_train, x_test, mean, std = standardize_from_train(x[train_idx], x[test_idx], axis=0)
    return dataset_dict(
        x_train,
        y[train_idx],
        delta[train_idx],
        x_test,
        y[test_idx],
        delta[test_idx],
        num_causes=2,
        feature_names=list(feature_names),
        dataset_name="synthetic",
        source_path=source,
        censor_rate=censor_rate,
        standardization_mean=mean,
        standardization_std=std,
    )
