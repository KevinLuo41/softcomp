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
    ax.set_title("CompSoft training loss")
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
