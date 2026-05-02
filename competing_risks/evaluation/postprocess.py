"""Post-processing for predicted CIF grids."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


def isotonic_project_cif(cif: Any, t_grid: Optional[Any] = None, clip: bool = True):
    """Project each subject/cause CIF curve onto the nondecreasing cone."""
    is_torch = False
    device = None
    dtype = None
    try:
        import torch

        if isinstance(cif, torch.Tensor):
            is_torch = True
            device = cif.device
            dtype = cif.dtype
            cif_np = cif.detach().cpu().numpy()
        else:
            cif_np = np.asarray(cif, dtype=float)
    except Exception:
        cif_np = np.asarray(cif, dtype=float)

    if cif_np.ndim != 3:
        raise ValueError("cif must have shape (n, K, T)")
    n, k, t = cif_np.shape
    if t_grid is None:
        x_axis = np.arange(t, dtype=float)
    else:
        x_axis = np.asarray(t_grid, dtype=float).reshape(-1)
        if x_axis.shape[0] != t:
            raise ValueError("t_grid length must match cif.shape[-1]")
    out = np.empty_like(cif_np, dtype=float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for i in range(n):
        for cause in range(k):
            projected = iso.fit_transform(x_axis, cif_np[i, cause, :])
            out[i, cause, :] = projected
    if clip:
        out = np.clip(out, 0.0, 1.0)
    if is_torch:
        import torch

        return torch.as_tensor(out, dtype=dtype, device=device)
    return out


def count_monotone_violations(cif: Any, tol: float = 1e-8):
    """Summarize monotonicity violations for a CIF tensor of shape (n, K, T)."""
    arr = np.asarray(cif, dtype=float)
    if arr.ndim != 3:
        raise ValueError("cif must have shape (n, K, T)")
    if arr.shape[-1] < 2:
        return {
            "n_pairs": int(arr.shape[0] * arr.shape[1]),
            "n_violating_pairs": 0,
            "fraction_violating": 0.0,
            "max_drop": 0.0,
            "mean_drop": 0.0,
        }
    diffs = np.diff(arr, axis=-1)
    drops = np.maximum(-diffs, 0.0)
    violating_pair = np.any(diffs < -tol, axis=-1)
    n_pairs = int(arr.shape[0] * arr.shape[1])
    n_violating = int(np.sum(violating_pair))
    positive_drops = drops[drops > tol]
    return {
        "n_pairs": n_pairs,
        "n_violating_pairs": n_violating,
        "fraction_violating": float(n_violating / max(1, n_pairs)),
        "max_drop": float(np.max(positive_drops)) if positive_drops.size else 0.0,
        "mean_drop": float(np.mean(positive_drops)) if positive_drops.size else 0.0,
    }
