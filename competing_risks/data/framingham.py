"""Framingham competing-risks loader and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import choose_column, dataset_dict, factorize_binary, require_file, split_indices


ID_CANDIDATES = ["RANDID", "id", "patient_id", "subject_id"]
TIME_CANDIDATES = ["time", "TIME", "futime", "followup_time"]
EVENT_CANDIDATES = ["event", "delta", "status"]
DEATH_CANDIDATES = ["death", "DEATH"]
CVD_CANDIDATES = ["cvd", "CVD"]
TIME_DEATH_CANDIDATES = ["timedth", "TIMEDTH", "time_death"]
TIME_CVD_CANDIDATES = ["timecvd", "TIMECVD", "time_cvd"]

BINARY_CANDIDATES = [
    "male",
    "sex",
    "currentSmoker",
    "CURSMOKE",
    "BPMeds",
    "BPMEDS",
    "prevalentStroke",
    "PREVSTRK",
    "prevalentHyp",
    "PREVHYP",
    "diabetes",
    "DIABETES",
    "PREVCHD",
    "PREVAP",
    "PREVMI",
]

NUMERIC_CANDIDATES = [
    "age",
    "AGE",
    "totChol",
    "TOTCHOL",
    "sysBP",
    "SYSBP",
    "diaBP",
    "DIABP",
    "BMI",
    "heartRate",
    "HEARTRTE",
    "glucose",
    "GLUCOSE",
    "cigsPerDay",
    "CIGPDAY",
]

EDUC_CANDIDATES = ["educ", "EDUC"]


def _find_optional(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _first_observation(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _find_optional(df.columns, ID_CANDIDATES)
    if id_col is None:
        return df.reset_index(drop=True)
    time_col = _find_optional(df.columns, TIME_CANDIDATES) or _find_optional(
        df.columns, TIME_CVD_CANDIDATES + TIME_DEATH_CANDIDATES
    )
    if time_col is None:
        return df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    return (
        df.sort_values([id_col, time_col])
        .drop_duplicates(subset=[id_col], keep="first")
        .reset_index(drop=True)
    )


def _event_time_delta(df: pd.DataFrame):
    event_col = _find_optional(df.columns, EVENT_CANDIDATES)
    time_col = _find_optional(df.columns, TIME_CANDIDATES)
    if event_col is not None and time_col is not None:
        y = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) + 1.0
        delta = pd.to_numeric(df[event_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        return y, delta

    death_col = choose_column(df.columns, DEATH_CANDIDATES, "death event")
    cvd_col = choose_column(df.columns, CVD_CANDIDATES, "CVD event")
    death = pd.to_numeric(df[death_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    cvd = pd.to_numeric(df[cvd_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    t_death_col = _find_optional(df.columns, TIME_DEATH_CANDIDATES)
    t_cvd_col = _find_optional(df.columns, TIME_CVD_CANDIDATES)
    if t_death_col is not None and t_cvd_col is not None:
        t_death = pd.to_numeric(df[t_death_col], errors="coerce").to_numpy(dtype=float)
        t_cvd = pd.to_numeric(df[t_cvd_col], errors="coerce").to_numpy(dtype=float)
        death_time = np.where(death == 1, t_death, np.inf)
        cvd_time = np.where(cvd == 1, t_cvd, np.inf)
        y = np.minimum(death_time, cvd_time)
        delta = np.where(death_time <= cvd_time, 1, 2)
        delta = np.where(np.isfinite(y), delta, 0)
        censor_time = np.nanmax(np.vstack([t_death, t_cvd]), axis=0)
        y = np.where(np.isfinite(y), y, censor_time)
        return y + 1.0, delta.astype(int)

    if time_col is None:
        time_col = choose_column(df.columns, TIME_CANDIDATES, "follow-up time")
    y = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) + 1.0
    delta = np.where(death == 1, 1, np.where(cvd == 1, 2, 0)).astype(int)
    return y, delta


def _mode(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return 0.0
    vals, counts = np.unique(clean, return_counts=True)
    return float(vals[np.argmax(counts)])


def _build_features(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    feature_train: List[np.ndarray] = []
    feature_test: List[np.ndarray] = []
    names: List[str] = []

    binary_cols = []
    for col in BINARY_CANDIDATES:
        actual = _find_optional(df.columns, [col])
        if actual is not None and actual not in binary_cols:
            binary_cols.append(actual)
    for col in binary_cols:
        values = factorize_binary(df[col].to_numpy())
        fill = _mode(values[train_idx])
        values = np.where(np.isnan(values), fill, values)
        feature_train.append(values[train_idx, None])
        feature_test.append(values[test_idx, None])
        names.append(col)

    educ_col = _find_optional(df.columns, EDUC_CANDIDATES)
    if educ_col is not None:
        educ = pd.to_numeric(df[educ_col], errors="coerce").to_numpy(dtype=float)
        fill = _mode(educ[train_idx])
        educ = np.where(np.isnan(educ), fill, educ)
        categories = [1.0, 2.0, 3.0, 4.0]
        for category in categories:
            feature_train.append((educ[train_idx] == category).astype(float)[:, None])
            feature_test.append((educ[test_idx] == category).astype(float)[:, None])
            names.append(f"{educ_col}_{category:g}")

    numeric_cols = []
    for col in NUMERIC_CANDIDATES:
        actual = _find_optional(df.columns, [col])
        if actual is not None and actual not in numeric_cols and actual not in binary_cols:
            numeric_cols.append(actual)
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mean = np.nanmean(values[train_idx])
        values = np.where(np.isnan(values), mean, values)
        std = np.std(values[train_idx])
        if std < 1e-8:
            std = 1.0
        values = (values - mean) / std
        feature_train.append(values[train_idx, None])
        feature_test.append(values[test_idx, None])
        names.append(col)

    if not feature_train:
        raise ValueError("No usable Framingham feature columns were found")
    return np.concatenate(feature_train, axis=1), np.concatenate(feature_test, axis=1), names


def load_framingham(path: str | Path, test_size: float = 0.3, seed: int = 42):
    """Load Framingham data with first-observation and z-score preprocessing."""
    expected = ["RANDID", "time or event-time columns", "death", "CVD", "educ", "risk factors"]
    csv_path = require_file(path, "Framingham", expected)
    raw = pd.read_csv(csv_path)
    df = _first_observation(raw)
    y, delta = _event_time_delta(df)
    if np.any(~np.isfinite(y)):
        raise ValueError("Framingham follow-up time contains non-finite values")
    train_idx, test_idx = split_indices(len(df), test_size=test_size, seed=seed)
    x_train, x_test, feature_names = _build_features(df, train_idx, test_idx)
    return dataset_dict(
        x_train,
        y[train_idx],
        delta[train_idx],
        x_test,
        y[test_idx],
        delta[test_idx],
        num_causes=2,
        feature_names=feature_names,
        dataset_name="framingham",
        source_path=str(csv_path),
    )
