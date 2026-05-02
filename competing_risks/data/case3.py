"""Case III simulation: functional covariates."""

from __future__ import annotations

import numpy as np

from .utils import (
    EPS,
    dataset_dict,
    evaluate_cif_grid,
    rng_from_seed,
    sample_competing_risks,
    standardize_from_train,
)


def _fourier_basis(grid: np.ndarray, n_pairs: int = 5) -> np.ndarray:
    pieces = []
    for ell in range(1, n_pairs + 1):
        pieces.append(np.sin(2.0 * np.pi * ell * grid) / ell)
        pieces.append(np.cos(2.0 * np.pi * ell * grid) / ell)
    return np.stack(pieces, axis=0)


def _generate_functional_x(
    rng: np.random.Generator,
    n: int,
    p: int,
    grid: np.ndarray,
    n_pairs: int = 5,
) -> np.ndarray:
    basis = _fourier_basis(grid, n_pairs=n_pairs)
    n_basis = basis.shape[0]
    scales = np.repeat(1.0 / np.arange(1, n_pairs + 1), 2)
    coeff = rng.normal(size=(n, p, n_basis)) * scales.reshape(1, 1, -1)
    x = np.einsum("npb,bg->npg", coeff, basis)
    x += rng.normal(scale=0.05, size=x.shape)
    return x


def generate_case3(
    n_train: int = 5000,
    n_test: int = 1000,
    seed: int = 0,
    censoring_rate: float = 0.5,
):
    rng = rng_from_seed(seed)
    p = 3
    k = 2
    g = 50
    n_pairs = 5
    alpha = 0.4
    effect_scale = 2.0
    intercept = np.array([-3.0, -4.5], dtype=float)
    grid = np.linspace(0.0, 1.0, g)
    x_all = _generate_functional_x(rng, n_train + n_test, p, grid, n_pairs=n_pairs)
    x_train_raw, x_test_raw = x_all[:n_train], x_all[n_train:]
    x_train, x_test, mean, std = standardize_from_train(
        x_train_raw.reshape(n_train, -1),
        x_test_raw.reshape(n_test, -1),
        axis=0,
    )
    x_all = np.concatenate([x_train, x_test], axis=0).reshape(n_train + n_test, p, g)
    basis = _fourier_basis(grid, n_pairs=n_pairs)
    scales = effect_scale * np.repeat(1.0 / np.arange(1, n_pairs + 1), 2)
    beta_coeff = rng.normal(size=(k, p, basis.shape[0])) * scales.reshape(1, 1, -1)
    beta_func = np.einsum("kpb,bg->kpg", beta_coeff, basis)

    def mu_fn(x, t):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 2:
            x_arr = x_arr.reshape(x_arr.shape[0], p, g)
        t_arr = np.asarray(t, dtype=float).reshape(-1, 1)
        effects = np.einsum("npg,kpg->nk", x_arr, beta_func) / g
        return intercept.reshape(1, -1) + effects + alpha * t_arr

    _, y, delta, causes, censor_rate = sample_competing_risks(
        x_all, mu_fn, rng, target_censoring=censoring_rate, t_upper=30.0
    )
    x_train, x_test = x_all[:n_train], x_all[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    d_train, d_test = delta[:n_train], delta[n_train:]

    def true_cif_fn(x, t_grid):
        return evaluate_cif_grid(mu_fn, x, t_grid)

    return dataset_dict(
        x_train,
        y_train,
        d_train,
        x_test,
        y_test,
        d_test,
        num_causes=k,
        feature_names=[f"x{j}_grid{m}" for j in range(p) for m in range(g)],
        true_cif_fn=true_cif_fn,
        true_params={
            "intercept": intercept,
            "beta_func": beta_func,
            "alpha": alpha,
            "grid": grid,
            "standardization_mean": mean,
            "standardization_std": np.maximum(std, EPS),
        },
        latent_causes=causes,
        censor_rate=censor_rate,
        dataset_name="case3",
        functional_grid=grid,
    )
