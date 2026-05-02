"""Simulation metrics for cases I/II/III."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def mse_by_cause(pred_cif: Any, true_cif: Any) -> Dict[str, float]:
    pred = np.asarray(pred_cif, dtype=float)
    truth = np.asarray(true_cif, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError(f"pred_cif and true_cif must share shape; got {pred.shape} vs {truth.shape}")
    per_cause = np.mean((pred - truth) ** 2, axis=(0, 2))
    result = {f"mse_cause_{i + 1}": float(value) for i, value in enumerate(per_cause)}
    result["mse"] = float(np.mean((pred - truth) ** 2))
    return result


def classification_accuracy(pred_cif: Any, pred_survival: Any, true_cif: Any) -> float:
    pred = np.asarray(pred_cif, dtype=float)
    pred_s = np.asarray(pred_survival, dtype=float)
    truth = np.asarray(true_cif, dtype=float)
    true_s = 1.0 - np.sum(truth, axis=1)
    pred_labels = np.argmax(np.concatenate([pred_s[:, None, :], pred], axis=1), axis=1)
    true_labels = np.argmax(np.concatenate([true_s[:, None, :], truth], axis=1), axis=1)
    return float(np.mean(pred_labels == true_labels))


def evaluate_simulation(
    pred_cif: Any,
    pred_survival: Any,
    true_cif: Any,
) -> Dict[str, float]:
    metrics = mse_by_cause(pred_cif, true_cif)
    metrics["accuracy"] = classification_accuracy(pred_cif, pred_survival, true_cif)
    return metrics


def compute_mse_accuracy(
    cif_pred: Any,
    X_test: Any,
    times: Any,
    true_cif_fn,
    K: int | None = None,
) -> Dict[str, float]:
    """Blueprint-compatible wrapper for simulation MSE and accuracy."""
    true_cif, _ = true_cif_fn(X_test, times)
    pred = np.asarray(cif_pred, dtype=float)
    survival = 1.0 - np.sum(pred, axis=1)
    if K is not None:
        pred = pred[:, :K, :]
        true_cif = true_cif[:, :K, :]
    return evaluate_simulation(pred, survival, true_cif)
