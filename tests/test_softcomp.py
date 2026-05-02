import numpy as np
import torch

from competing_risks.baseline_models import DeepHit
from competing_risks.softcomp_model import SoftCompEnsemble, SoftCompNet, FunctionalSoftCompNet
from competing_risks.data.case1 import generate_case1
from competing_risks.evaluation.checkpoints import load_model_checkpoint, save_model_checkpoint
from competing_risks.evaluation.postprocess import count_monotone_violations, isotonic_project_cif
from competing_risks.evaluation.survival import (
    build_evaluation_time_grid,
    compute_ctd,
    compute_ibs,
    evaluate_survival,
)


def test_softcomp_probabilities_sum_to_one():
    model = SoftCompNet(input_dim=3, num_causes=2, hidden_dim=8, num_blocks=1)
    x = torch.randn(5, 3)
    t = torch.ones(5)
    cif, survival = model.predict_cif(x, t)
    assert cif.shape == (5, 2)
    assert survival.shape == (5,)
    total = cif.sum(dim=1) + survival
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_functional_model_forward_shape():
    model = FunctionalSoftCompNet(
        num_covariates=3,
        n_grid=10,
        embed_dim=4,
        num_causes=2,
        hidden_dim=8,
        num_blocks=1,
    )
    x = torch.randn(7, 3, 10)
    t = torch.linspace(0.1, 1.0, 7)
    logits = model(x, t)
    assert logits.shape == (7, 2)


def test_fit_smoke_on_case1():
    data = generate_case1(n_train=64, n_test=16, seed=123)
    model = SoftCompNet(input_dim=data["X_train"].shape[1], num_causes=3, hidden_dim=8)
    history = model.fit(
        data["X_train"],
        data["Y_train"],
        data["Delta_train"],
        epochs=1,
        batch_size=32,
        n_aug=1,
        verbose=False,
    )
    assert len(history["train_loss"]) == 1
    t_grid = build_evaluation_time_grid(data["Y_test"], data["Delta_test"], n_grid=5)
    cif, survival = model.predict_cif_grid(data["X_test"], t_grid)
    assert cif.shape == (16, 3, len(t_grid))
    assert survival.shape == (16, len(t_grid))


def test_isotonic_projection_is_monotone():
    cif = np.array([[[0.1, 0.3, 0.2, 0.5], [0.0, 0.2, 0.1, 0.4]]])
    before = count_monotone_violations(cif)
    projected = isotonic_project_cif(cif)
    after = count_monotone_violations(projected)
    assert before["n_violating_pairs"] == 2
    assert after["n_violating_pairs"] == 0
    assert np.all(np.diff(projected, axis=-1) >= -1e-10)


def test_survival_metrics_are_finite_for_small_arrays():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    d_train = np.array([1, 0, 2, 0])
    y = np.array([1.0, 2.0, 3.0])
    d = np.array([1, 2, 0])
    times = np.array([1.0, 2.0, 3.0])
    cif = np.array(
        [
            [[0.2, 0.3, 0.4], [0.1, 0.1, 0.2]],
            [[0.1, 0.2, 0.3], [0.2, 0.4, 0.5]],
            [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]],
        ]
    )
    metrics = evaluate_survival(cif, y, d, times, y_train, d_train, num_causes=2)
    assert np.isfinite(metrics["ibs_mean"])
    assert np.isfinite(compute_ibs(y, d, y_train, d_train, cif, times, event_k=1))
    ctd = compute_ctd(y, d, cif, times, event_k=1)
    assert np.isnan(ctd) or 0.0 <= ctd <= 1.0


def test_model_checkpoint_round_trip(tmp_path):
    model = SoftCompNet(input_dim=2, num_causes=2, hidden_dim=4)
    path = tmp_path / "model.pt"
    save_model_checkpoint(model, path, metadata={"hello": "world"})
    loaded = SoftCompNet(input_dim=2, num_causes=2, hidden_dim=4)
    metadata = load_model_checkpoint(loaded, path)
    assert metadata["hello"] == "world"
    for left, right in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(left, right)


def test_softcomp_ensemble_and_checkpoint_constructor():
    model = SoftCompNet(input_dim=2, num_causes=2, hidden_dim=4)
    loaded = SoftCompNet.load_from_checkpoint(model.checkpoint_dict())
    ensemble = SoftCompEnsemble([model, loaded])
    x = torch.randn(4, 2)
    t = torch.ones(4)
    cif, survival = ensemble.predict_cif(x, t)
    assert cif.shape == (4, 2)
    assert survival.shape == (4,)
    assert torch.all(cif >= 0)


def test_baseline_predict_cif_api():
    data = generate_case1(n_train=64, n_test=8, seed=321)
    model = DeepHit(num_causes=data["num_causes"]).fit(
        data["X_train"], data["Y_train"], data["Delta_train"]
    )
    times = build_evaluation_time_grid(data["Y_test"], data["Delta_test"], n_grid=4)
    cif = model.predict_cif(data["X_test"], times)
    assert tuple(cif.shape) == (8, data["num_causes"], len(times))
