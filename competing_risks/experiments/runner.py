"""Shared experiment scaffold for all CompSoft reproductions."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from competing_risks.compsoft_model import CompSoftNet, FunctionalCompSoftNet
from competing_risks.evaluation.checkpoints import (
    load_model_checkpoint,
    save_json,
    save_model_checkpoint,
    save_npz,
    save_pickle,
)
from competing_risks.evaluation.postprocess import isotonic_project_cif
from competing_risks.evaluation.simulation import evaluate_simulation
from competing_risks.evaluation.survival import build_evaluation_time_grid, evaluate_survival
from competing_risks.evaluation.visualize import plot_cif_comparison, plot_training_loss


@dataclass
class ModelSpec:
    name: str
    build_model: Callable[[Dict[str, Any]], torch.nn.Module]
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)
    post_process: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_compsoft_spec(
    hidden_dim: int,
    num_blocks: int,
    dropout: float,
    fit_kwargs: Dict[str, Any],
) -> ModelSpec:
    def build(data: Dict[str, Any]):
        return CompSoftNet(
            input_dim=data["X_train"].shape[1],
            num_causes=data["num_causes"],
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

    return ModelSpec(
        name="compsoft",
        build_model=build,
        fit_kwargs=fit_kwargs,
        post_process=lambda cif, t_grid: isotonic_project_cif(cif, t_grid),
    )


def make_functional_compsoft_spec(
    hidden_dim: int,
    num_blocks: int,
    embed_dim: int,
    dropout: float,
    fit_kwargs: Dict[str, Any],
) -> ModelSpec:
    def build(data: Dict[str, Any]):
        x = data["X_train"]
        return FunctionalCompSoftNet(
            num_covariates=x.shape[1],
            n_grid=x.shape[2],
            embed_dim=embed_dim,
            num_causes=data["num_causes"],
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

    return ModelSpec(
        name="compsoft",
        build_model=build,
        fit_kwargs=fit_kwargs,
        post_process=lambda cif, t_grid: isotonic_project_cif(cif, t_grid),
    )


def finite_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, (float, np.floating)):
            clean[key] = None if not math.isfinite(float(value)) else float(value)
        elif isinstance(value, (int, np.integer)):
            clean[key] = int(value)
        else:
            clean[key] = value
    return clean


def subset_dataset(data: Dict[str, Any], n_train: int = 512, n_test: int = 256) -> Dict[str, Any]:
    out = dict(data)
    train_n = min(n_train, data["X_train"].shape[0])
    test_n = min(n_test, data["X_test"].shape[0])
    for key in ["X_train", "Y_train", "Delta_train"]:
        out[key] = data[key][:train_n]
    for key in ["X_test", "Y_test", "Delta_test"]:
        out[key] = data[key][:test_n]
    return out


def run_experiment(
    dataset_name: str,
    data: Dict[str, Any],
    model_spec: ModelSpec,
    output_dir: str | Path = "outputs",
    n_grid: int = 100,
    percentile_cap: float = 90.0,
    force: bool = False,
    device: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    no_plots: bool = False,
) -> Dict[str, Any]:
    seed = int(model_spec.fit_kwargs.get("seed", 0))
    set_torch_seed(seed)
    out_dir = Path(output_dir) / dataset_name / model_spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    pred_path = out_dir / "predictions.npz"
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.pkl"

    model = model_spec.build_model(data)
    fit_kwargs = dict(model_spec.fit_kwargs)
    fit_kwargs.pop("seed", None)
    if epochs_override is not None:
        fit_kwargs["epochs"] = epochs_override
    if batch_size_override is not None:
        fit_kwargs["batch_size"] = batch_size_override
    if device is not None:
        fit_kwargs["device"] = device

    if force or not model_path.exists():
        history = model.fit(
            data["X_train"],
            data["Y_train"],
            data["Delta_train"],
            **fit_kwargs,
        )
        save_model_checkpoint(
            model,
            model_path,
            metadata={"dataset": dataset_name, "model": model_spec.name, "fit_kwargs": fit_kwargs},
        )
        save_pickle(history, history_path)
    else:
        load_model_checkpoint(model, model_path, map_location=device or "cpu")
        history = getattr(model, "history_", {})

    if force or not pred_path.exists():
        t_grid = build_evaluation_time_grid(
            data["Y_test"],
            data["Delta_test"],
            n_grid=n_grid,
            percentile_cap=percentile_cap,
        )
        cif, survival = model.predict_cif_grid(data["X_test"], t_grid, device=device, as_numpy=True)
        if model_spec.post_process is not None:
            cif = model_spec.post_process(cif, t_grid)
            survival = np.clip(1.0 - np.sum(cif, axis=1), 0.0, 1.0)
        save_npz(pred_path, t_grid=t_grid, cif=cif, survival=survival)
    else:
        cached = np.load(pred_path)
        t_grid = cached["t_grid"]
        cif = cached["cif"]
        survival = cached["survival"]

    if "true_cif_fn" in data:
        true_cif, _ = data["true_cif_fn"](data["X_test"], t_grid)
        metrics = evaluate_simulation(cif, survival, true_cif)
    else:
        true_cif = None
        metrics = evaluate_survival(
            cif,
            data["Y_test"],
            data["Delta_test"],
            t_grid,
            data["Y_train"],
            data["Delta_train"],
            num_causes=data["num_causes"],
        )
    metrics.update(
        {
            "dataset": dataset_name,
            "model": model_spec.name,
            "n_train": int(data["X_train"].shape[0]),
            "n_test": int(data["X_test"].shape[0]),
            "n_times": int(len(t_grid)),
        }
    )
    save_json(finite_json(metrics), metrics_path)

    if not no_plots:
        if history:
            plot_training_loss(history, out_dir / "training_loss.png")
        plot_cif_comparison(t_grid, cif, true_cif=true_cif, subject_idx=0, out_path=out_dir / "cif_subject0.png")
    return {
        "metrics": metrics,
        "history": history,
        "model_path": model_path,
        "prediction_path": pred_path,
        "metrics_path": metrics_path,
    }


def build_parser(dataset_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run CompSoft on {dataset_name}")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-path", default=None, help="CSV path for real-data experiments")
    parser.add_argument("--force", action="store_true", help="Retrain and recompute predictions")
    parser.add_argument("--quick", action="store_true", help="Small smoke run with fewer samples/epochs")
    parser.add_argument("--quick-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size")
    parser.add_argument("--n-grid", type=int, default=100)
    parser.add_argument("--percentile-cap", type=float, default=90.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def cli_main(
    dataset_name: str,
    load_data: Callable[[argparse.Namespace], Dict[str, Any]],
    model_spec_factory: Callable[[Dict[str, Any]], ModelSpec],
) -> Dict[str, Any]:
    parser = build_parser(dataset_name)
    args = parser.parse_args()
    data = load_data(args)
    if args.quick:
        data = subset_dataset(data)
    model_spec = model_spec_factory(data)
    epochs_override = args.quick_epochs if args.quick else args.epochs
    result = run_experiment(
        dataset_name,
        data,
        model_spec,
        output_dir=args.output_dir,
        n_grid=args.n_grid,
        percentile_cap=args.percentile_cap,
        force=args.force,
        device=args.device,
        epochs_override=epochs_override,
        batch_size_override=args.batch_size,
        no_plots=args.no_plots,
    )
    print(f"metrics: {result['metrics_path']}")
    for key, value in sorted(result["metrics"].items()):
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    return result
