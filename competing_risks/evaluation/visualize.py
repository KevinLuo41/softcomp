"""Visualization helpers for CIF curves and training loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _finish_plot(fig, out_path: Optional[str | Path]):
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160, bbox_inches="tight")
    return fig


def plot_training_loss(history: dict, out_path: Optional[str | Path] = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    train = history.get("train_loss", [])
    ax.plot(np.arange(1, len(train) + 1), train, label="train")
    val = history.get("val_loss", [])
    if val:
        ax.plot(np.arange(1, len(val) + 1), val, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SoftComp training loss")
    ax.legend()
    ax.grid(alpha=0.25)
    return _finish_plot(fig, out_path)


def plot_cif_comparison(
    times: Any,
    pred_cif: Any,
    true_cif: Optional[Any] = None,
    subject_idx: int = 0,
    out_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    t = np.asarray(times, dtype=float)
    pred = np.asarray(pred_cif, dtype=float)
    if pred.ndim != 3:
        raise ValueError("pred_cif must have shape (n, K, T)")
    fig, ax = plt.subplots(figsize=(7, 4))
    k = pred.shape[1]
    for cause in range(k):
        ax.plot(t, pred[subject_idx, cause, :], label=f"pred cause {cause + 1}")
    if true_cif is not None:
        truth = np.asarray(true_cif, dtype=float)
        for cause in range(k):
            ax.plot(
                t,
                truth[subject_idx, cause, :],
                linestyle="--",
                label=f"true cause {cause + 1}",
            )
    ax.set_xlabel("Time")
    ax.set_ylabel("CIF")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"CIF comparison, subject {subject_idx}")
    ax.legend()
    ax.grid(alpha=0.25)
    return _finish_plot(fig, out_path)


def plot_cif_curves(times: Any, cif: Any, out_path: Optional[str | Path] = None, max_subjects: int = 20):
    import matplotlib.pyplot as plt

    t = np.asarray(times, dtype=float)
    arr = np.asarray(cif, dtype=float)
    if arr.ndim != 3:
        raise ValueError("cif must have shape (n, K, T)")
    fig, axes = plt.subplots(arr.shape[1], 1, figsize=(7, 3 * arr.shape[1]), squeeze=False)
    for cause in range(arr.shape[1]):
        ax = axes[cause, 0]
        for i in range(min(max_subjects, arr.shape[0])):
            ax.plot(t, arr[i, cause, :], alpha=0.35)
        ax.set_title(f"Cause {cause + 1} CIF")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
    return _finish_plot(fig, out_path)


def plot_event_time_distribution(
    y: Any,
    delta: Any,
    out_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    y_arr = np.asarray(y, dtype=float)
    d_arr = np.asarray(delta, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_arr[d_arr == 0], bins=30, alpha=0.5, label="censored")
    for cause in sorted(c for c in np.unique(d_arr) if c > 0):
        ax.hist(y_arr[d_arr == cause], bins=30, alpha=0.5, label=f"cause {cause}")
    ax.set_xlabel("Observed time")
    ax.set_ylabel("Count")
    ax.set_title("Event time distribution")
    ax.legend()
    return _finish_plot(fig, out_path)


def plot_functional_pipeline(
    grid: Any,
    x_functions: Any,
    out_path: Optional[str | Path] = None,
    max_subjects: int = 6,
):
    import matplotlib.pyplot as plt

    g = np.asarray(grid, dtype=float)
    x = np.asarray(x_functions, dtype=float)
    if x.ndim != 3:
        raise ValueError("x_functions must have shape (n, p, G)")
    fig, axes = plt.subplots(x.shape[1], 1, figsize=(7, 2.5 * x.shape[1]), squeeze=False)
    for covariate in range(x.shape[1]):
        ax = axes[covariate, 0]
        for i in range(min(max_subjects, x.shape[0])):
            ax.plot(g, x[i, covariate, :], alpha=0.65)
        ax.set_title(f"Functional covariate {covariate + 1}")
        ax.grid(alpha=0.25)
    return _finish_plot(fig, out_path)


def print_data_summary(data: dict) -> None:
    print(f"n_train={len(data['Y_train'])} n_test={len(data['Y_test'])}")
    print(f"num_causes={data.get('num_causes')} x_shape={data['X_train'].shape}")
    for split in ["train", "test"]:
        delta = np.asarray(data[f"Delta_{split}"], dtype=int)
        counts = {int(k): int(np.sum(delta == k)) for k in np.unique(delta)}
        print(f"{split}_events={counts}")


def print_example_predictions(times: Any, cif: Any, n: int = 3) -> None:
    t = np.asarray(times, dtype=float)
    arr = np.asarray(cif, dtype=float)
    for i in range(min(n, arr.shape[0])):
        final = arr[i, :, -1]
        print(f"subject={i} t0={t[0]:.4g} t_last={t[-1]:.4g} final_cif={final}")
