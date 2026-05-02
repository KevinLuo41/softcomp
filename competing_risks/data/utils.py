"""Shared data-generation and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-8


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def mu_to_cif(mu: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Convert K event logits to cause CIFs and survival via stable softmax."""
    mu_arr = np.asarray(mu, dtype=float)
    zeros = np.zeros(mu_arr.shape[:-1] + (1,), dtype=float)
    logits = np.concatenate([zeros, mu_arr], axis=-1)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs[..., 1:], probs[..., 0]


def evaluate_cif_grid(
    mu_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: Any,
    t_grid: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a true CIF function on a grid."""
    x_arr = np.asarray(x, dtype=float)
    times = np.asarray(t_grid, dtype=float).reshape(-1)
    n = x_arr.shape[0]
    first_mu = mu_fn(x_arr[:1], np.asarray([times[0]], dtype=float))
    k = first_mu.shape[1]
    cif = np.empty((n, k, len(times)), dtype=float)
    survival = np.empty((n, len(times)), dtype=float)
    for j, tj in enumerate(times):
        t = np.full(n, tj, dtype=float)
        fj, sj = mu_to_cif(mu_fn(x_arr, t))
        cif[:, :, j] = fj
        survival[:, j] = sj
    return cif, survival


def solve_inverse_cdf(
    mu_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: Any,
    u: Any,
    t_upper: float = 20.0,
    max_iter: int = 80,
) -> np.ndarray:
    """Solve sum_k F_k(t | x) = u by vectorized bisection."""
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    lo = np.zeros_like(u_arr, dtype=float)
    hi = np.full_like(u_arr, float(t_upper), dtype=float)
    for _ in range(30):
        cif_hi, _ = mu_to_cif(mu_fn(x_arr, hi))
        needs_more = np.sum(cif_hi, axis=1) < u_arr
        if not np.any(needs_more):
            break
        hi[needs_more] *= 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        cif_mid, _ = mu_to_cif(mu_fn(x_arr, mid))
        total_event = np.sum(cif_mid, axis=1)
        hi = np.where(total_event >= u_arr, mid, hi)
        lo = np.where(total_event < u_arr, mid, lo)
    return (lo + hi) / 2.0


def assign_causes(cif_at_time: Any, rng: np.random.Generator) -> np.ndarray:
    """Draw event causes using cause-specific CIF proportions at event time."""
    cif = np.asarray(cif_at_time, dtype=float)
    totals = np.sum(cif, axis=1, keepdims=True)
    probs = np.divide(cif, np.maximum(totals, EPS))
    cdf = np.cumsum(probs, axis=1)
    draws = rng.uniform(size=cif.shape[0])
    causes = np.empty(cif.shape[0], dtype=int)
    for i in range(cif.shape[0]):
        causes[i] = int(np.searchsorted(cdf[i], draws[i], side="right") + 1)
    return causes


def tune_exponential_censor_rate(event_times: Any, target_censoring: float) -> float:
    """Find an exponential censoring rate that matches a target censoring rate."""
    target = float(np.clip(target_censoring, 0.0, 0.999))
    if target <= 0:
        return 0.0
    t = np.asarray(event_times, dtype=float)
    lo, hi = 0.0, 1.0 / max(float(np.mean(t)), EPS)

    def rate_to_censoring(rate: float) -> float:
        return float(np.mean(1.0 - np.exp(-rate * t)))

    while rate_to_censoring(hi) < target:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if rate_to_censoring(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def apply_exponential_censoring(
    event_times: Any,
    causes: Any,
    target_censoring: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    event = np.asarray(event_times, dtype=float)
    cause = np.asarray(causes, dtype=int)
    rate = tune_exponential_censor_rate(event, target_censoring)
    if rate == 0:
        censor_time = np.full_like(event, np.inf)
    else:
        censor_time = rng.exponential(scale=1.0 / rate, size=event.shape[0])
    y = np.minimum(event, censor_time)
    delta = np.where(event <= censor_time, cause, 0).astype(int)
    return np.maximum(y, EPS), delta, censor_time, rate


def sample_competing_risks(
    x: Any,
    mu_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
    target_censoring: float = 0.5,
    t_upper: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample event time, observed time, cause, and censoring from a DGP."""
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.shape[0]
    u = rng.uniform(low=EPS, high=1.0 - EPS, size=n)
    event_times = solve_inverse_cdf(mu_fn, x_arr, u, t_upper=t_upper)
    cif_at_event, _ = mu_to_cif(mu_fn(x_arr, event_times))
    causes = assign_causes(cif_at_event, rng)
    y, delta, _, censor_rate = apply_exponential_censoring(
        event_times, causes, target_censoring, rng
    )
    return event_times, y, delta, causes, censor_rate


def assert_monotone(cif: Any, tol: float = 1e-8) -> None:
    arr = np.asarray(cif, dtype=float)
    if arr.shape[-1] < 2:
        return
    if np.any(np.diff(arr, axis=-1) < -tol):
        raise ValueError("CIF is not monotone along the time axis")


def assert_monotone_cif(
    cif_fn: Callable[[np.ndarray, np.ndarray], Any],
    x_sample: Any,
    t_grid: Any,
    atol: float = 1e-6,
) -> None:
    """Blueprint-compatible true-CIF monotonicity check for a DGP function."""
    x_arr = np.asarray(x_sample, dtype=float)
    times = np.asarray(t_grid, dtype=float)
    values = []
    for t in times:
        t_vec = np.full(x_arr.shape[0], float(t))
        out = cif_fn(x_arr, t_vec)
        cif = out[0] if isinstance(out, tuple) else out
        values.append(np.asarray(cif, dtype=float)[..., None])
    assert_monotone(np.concatenate(values, axis=-1), tol=atol)


def assign_causes_and_censor(
    x: Any,
    t_event: Any,
    cif_fn: Callable[[np.ndarray, np.ndarray], Any],
    censor_rate: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw causes at event times and apply exponential censoring."""
    rng = rng_from_seed(seed)
    x_arr = np.asarray(x, dtype=float)
    event_times = np.asarray(t_event, dtype=float)
    out = cif_fn(x_arr, event_times)
    cif_at_time = out[0] if isinstance(out, tuple) else out
    causes = assign_causes(cif_at_time, rng)
    y, delta, _, _ = apply_exponential_censoring(event_times, causes, censor_rate, rng)
    return y, delta, causes


def generate_test_observations(
    n_test: int,
    p: int,
    cif_fn: Callable[[np.ndarray, np.ndarray], Any],
    censor_rate: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate generic Gaussian test observations from a CIF function."""
    rng = rng_from_seed(seed)
    x = rng.normal(size=(n_test, p))

    def mu_adapter(x_batch: np.ndarray, t_batch: np.ndarray) -> np.ndarray:
        out = cif_fn(x_batch, t_batch)
        cif = out[0] if isinstance(out, tuple) else out
        survival = np.maximum(1.0 - np.sum(cif, axis=1, keepdims=True), EPS)
        return np.log(np.maximum(cif, EPS) / survival)

    _, y, delta, _, _ = sample_competing_risks(
        x, mu_adapter, rng, target_censoring=censor_rate, t_upper=30.0
    )
    return x, y, delta


def split_indices(n: int, test_size: float = 0.3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng_from_seed(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def standardize_from_train(
    train: Any,
    test: Any,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_arr = np.asarray(train, dtype=float)
    test_arr = np.asarray(test, dtype=float)
    mean = np.mean(train_arr, axis=axis, keepdims=True)
    std = np.std(train_arr, axis=axis, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (train_arr - mean) / std, (test_arr - mean) / std, mean, std


def require_file(path: Any, dataset_name: str, columns: Optional[Sequence[str]] = None) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        message = f"{dataset_name} data file not found: {p}"
        if columns:
            message += "\nExpected columns include: " + ", ".join(columns)
        raise FileNotFoundError(message)
    return p


def choose_column(columns: Iterable[str], candidates: Sequence[str], label: str) -> str:
    available = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in available:
            return available[candidate.lower()]
    raise ValueError(f"Missing required {label} column. Tried: {', '.join(candidates)}")


def factorize_binary(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)
    cleaned = np.array([str(v).strip().lower() for v in arr], dtype=object)
    mapping = {
        "yes": 1.0,
        "y": 1.0,
        "true": 1.0,
        "1": 1.0,
        "male": 1.0,
        "m": 1.0,
        "d-penicil": 1.0,
        "d-penicillamine": 1.0,
        "no": 0.0,
        "n": 0.0,
        "false": 0.0,
        "0": 0.0,
        "female": 0.0,
        "f": 0.0,
        "placebo": 0.0,
    }
    out = np.empty(cleaned.shape[0], dtype=float)
    unknown = []
    for i, value in enumerate(cleaned):
        if value in mapping:
            out[i] = mapping[value]
        else:
            unknown.append(value)
            out[i] = np.nan
    if unknown:
        unique = {v: j for j, v in enumerate(sorted(set(unknown)))}
        for i, value in enumerate(cleaned):
            if np.isnan(out[i]):
                out[i] = float(unique[value])
    return out


def dataset_dict(
    X_train: Any,
    Y_train: Any,
    Delta_train: Any,
    X_test: Any,
    Y_test: Any,
    Delta_test: Any,
    num_causes: int,
    feature_names: Optional[List[str]] = None,
    **metadata: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "X_train": np.asarray(X_train, dtype=float),
        "Y_train": np.asarray(Y_train, dtype=float),
        "Delta_train": np.asarray(Delta_train, dtype=int),
        "X_test": np.asarray(X_test, dtype=float),
        "Y_test": np.asarray(Y_test, dtype=float),
        "Delta_test": np.asarray(Delta_test, dtype=int),
        "num_causes": int(num_causes),
        "K": int(num_causes),
        "P": int(np.asarray(X_train).shape[1]) if np.asarray(X_train).ndim >= 2 else 1,
        "events": tuple(range(1, int(num_causes) + 1)),
        "feature_names": feature_names,
    }
    result.update(metadata)
    return result
