"""SoftComp @ Synthetic."""

from competing_risks.data.synthetic import load_synthetic
from competing_risks.experiments.runner import cli_main, make_softcomp_spec


def load_data(args):
    return load_synthetic(args.data_path, seed=42, generate_if_missing=True)


def model_spec_factory(data):
    return make_softcomp_spec(
        hidden_dim=32,
        num_blocks=1,
        dropout=0.0,
        fit_kwargs={
            "epochs": 200,
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "batch_size": 512,
            "n_aug": 8,
            "aug_weight": 0.5,
            "brier_lambda": 5.0,
            "brier_n_times": 5,
            "verbose": True,
            "seed": 42,
        },
    )


if __name__ == "__main__":
    cli_main("synthetic", load_data, model_spec_factory)
