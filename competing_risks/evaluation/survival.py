"""Survival and competing-risks metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def build_evaluation_time_grid(
    Y_test: Any,
    Delta_test: Any,
    n_grid: int = 100,
    percentile_cap: float = 90.0,
) -> np.ndarray:
    y = np.asarray(Y_test, dtype=float)
    delta = np.asarray(Delta_test, dtype=int)
    event_times = y[delta > 0]
    if event_times.size == 0:
        event_times = y[np.isfinite(y)]
    event_times = event_times[event_times > 0]
    if event_times.size == 0:
        raise ValueError("Cannot build evaluation grid without positive times")
    t_min = float(np.min(event_times))
    t_max = float(np.percentile(event_times, percentile_cap))
    if t_max <= t_min:
        t_max = float(np.max(event_times))
    grid = np.linspace(t_min, t_max, n_grid)
    return np.unique(grid)


@dataclass
class KaplanMeierCensoring:
    """Kaplan-Meier estimate of censoring survival G(t)."""

    times_: Optional[np.ndarray] = None
    survival_: Optional[np.ndarray] = None

    def fit(self, y: Any, delta: Any) -> "KaplanMeierCensoring":
        times = np.asarray(y, dtype=float)
        censor_event = (np.asarray(delta, dtype=int) == 0).astype(int)
        unique_times = np.unique(times)
        surv_values = []
        survival = 1.0
        for t in unique_times:
            at_risk = np.sum(times >= t)
            events = np.sum((times == t) & (censor_event == 1))
            if at_risk > 0:
                survival *= 1.0 - events / at_risk
            surv_values.append(max(survival, 1e-6))
        self.times_ = unique_times
        self.survival_ = np.asarray(surv_values, dtype=float)
        return self

    def predict(self, t: Any, left_limit: bool = False) -> np.ndarray:
        if self.times_ is None or self.survival_ is None:
            raise RuntimeError("KaplanMeierCensoring.fit must be called before predict")
        values = np.asarray(t, dtype=float)
        side = "left" if left_limit else "right"
        idx = np.searchsorted(self.times_, values, side=side) - 1
        out = np.ones_like(values, dtype=float)
        valid = idx >= 0
        out[valid] = self.survival_[idx[valid]]
        return np.maximum(out, 1e-6)


def _interp_risk(cif_for_cause: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, times, row) for row in cif_for_cause], dtype=float)


def brier_score_ipcw(
    cif_for_cause: Any,
    y: Any,
    delta: Any,
    train_y: Any,
    train_delta: Any,
    times: Any,
    cause: int,
) -> np.ndarray:
    """Cause-specific IPCW Brier score over an evaluation grid."""
    pred = np.asarray(cif_for_cause, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    d_arr = np.asarray(delta, dtype=int)
    grid = np.asarray(times, dtype=float)
    if pred.shape != (len(y_arr), len(grid)):
        raise ValueError("cif_for_cause must have shape (n, T)")
    km = KaplanMeierCensoring().fit(train_y, train_delta)
    scores = np.empty(len(grid), dtype=float)
    for j, t in enumerate(grid):
        f = pred[:, j]
        observed_event_by_t = (y_arr <= t) & (d_arr > 0)
        event_cause = observed_event_by_t & (d_arr == cause)
        other_event = observed_event_by_t & (d_arr != cause)
        still_at_risk = y_arr > t
        weights = np.zeros_like(y_arr, dtype=float)
        target = np.zeros_like(y_arr, dtype=float)
        if np.any(event_cause):
            weights[event_cause] = 1.0 / km.predict(y_arr[event_cause], left_limit=True)
            target[event_cause] = 1.0
        if np.any(other_event):
            weights[other_event] = 1.0 / km.predict(y_arr[other_event], left_limit=True)
        if np.any(still_at_risk):
            weights[still_at_risk] = 1.0 / km.predict(np.full(np.sum(still_at_risk), t))
        scores[j] = float(np.mean(weights * (target - f) ** 2))
    return scores


def integrated_brier_score(
    cif_for_cause: Any,
    y: Any,
    delta: Any,
    train_y: Any,
    train_delta: Any,
    times: Any,
    cause: int,
) -> float:
    grid = np.asarray(times, dtype=float)
    bs = brier_score_ipcw(cif_for_cause, y, delta, train_y, train_delta, grid, cause)
    if len(grid) == 1:
        return float(bs[0])
    span = float(grid[-1] - grid[0])
    if span <= 0:
        return float(np.mean(bs))
    return float(np.trapz(bs, grid) / span)


def concordance_td(
    cif_for_cause: Any,
    y: Any,
    delta: Any,
    times: Any,
    cause: int,
) -> float:
    """Antolini-style time-dependent concordance for one cause."""
    pred = np.asarray(cif_for_cause, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    d_arr = np.asarray(delta, dtype=int)
    grid = np.asarray(times, dtype=float)
    event_idx = np.where(d_arr == cause)[0]
    concordant = 0.0
    comparable = 0
    for i in event_idx:
        mask = y_arr > y_arr[i]
        if not np.any(mask):
            continue
        risk_at_t = _interp_risk(pred, grid, float(y_arr[i]))
        diff = risk_at_t[i] - risk_at_t[mask]
        concordant += float(np.sum(diff > 0))
        concordant += 0.5 * float(np.sum(np.isclose(diff, 0.0)))
        comparable += int(np.sum(mask))
    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def evaluate_survival(
    pred_cif: Any,
    y: Any,
    delta: Any,
    times: Any,
    train_y: Any,
    train_delta: Any,
    num_causes: Optional[int] = None,
) -> Dict[str, float]:
    cif = np.asarray(pred_cif, dtype=float)
    if cif.ndim != 3:
        raise ValueError("pred_cif must have shape (n, K, T)")
    k = int(num_causes or cif.shape[1])
    metrics: Dict[str, float] = {}
    for cause in range(1, k + 1):
        c_pred = cif[:, cause - 1, :]
        metrics[f"ctd_cause_{cause}"] = concordance_td(c_pred, y, delta, times, cause)
        metrics[f"ibs_cause_{cause}"] = integrated_brier_score(
            c_pred, y, delta, train_y, train_delta, times, cause
        )
    ctd_values = [v for key, v in metrics.items() if key.startswith("ctd_") and np.isfinite(v)]
    ibs_values = [v for key, v in metrics.items() if key.startswith("ibs_") and np.isfinite(v)]
    metrics["ctd_mean"] = float(np.mean(ctd_values)) if ctd_values else float("nan")
    metrics["ibs_mean"] = float(np.mean(ibs_values)) if ibs_values else float("nan")
    return metrics
