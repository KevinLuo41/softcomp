"""Case I simulation: linear logits."""

from __future__ import annotations

import numpy as np

from .utils import dataset_dict, evaluate_cif_grid, rng_from_seed, sample_competing_risks


def generate_case1(
    n_train: int = 5000,
    n_test: int = 1000,
    seed: int = 0,
    censoring_rate: float = 0.5,
):
    rng = rng_from_seed(seed)
    p = 5
    k = 3
    alpha = 0.4
    intercept = np.array([-4.0, -4.75, -5.5], dtype=float)
    beta = rng.normal(loc=0.0, scale=0.6, size=(k, p))
    x_all = rng.normal(size=(n_train + n_test, p))

    def mu_fn(x, t):
        x_arr = np.asarray(x, dtype=float)
        t_arr = np.asarray(t, dtype=float).reshape(-1, 1)
        return intercept.reshape(1, -1) + x_arr @ beta.T + alpha * t_arr

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
        feature_names=[f"x{j}" for j in range(p)],
        true_cif_fn=true_cif_fn,
        true_params={"intercept": intercept, "beta": beta, "alpha": alpha},
        latent_causes=causes,
        censor_rate=censor_rate,
        dataset_name="case1",
    )
