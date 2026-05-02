"""CompSoft @ Framingham."""

from competing_risks.data.framingham import load_framingham
from competing_risks.experiments.runner import cli_main, make_compsoft_spec


def load_data(args):
    if args.data_path is None:
        raise FileNotFoundError("Pass --data-path pointing to the Framingham CSV file")
    return load_framingham(args.data_path, seed=42)


def model_spec_factory(data):
    return make_compsoft_spec(
        hidden_dim=32,
        num_blocks=1,
        dropout=0.0,
        fit_kwargs={
            "epochs": 1000,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "n_aug": 2,
            "aug_weight": 0.5,
            "brier_lambda": 2.0,
            "brier_n_times": 5,
            "verbose": True,
            "seed": 0,
        },
    )


if __name__ == "__main__":
    cli_main("framingham", load_data, model_spec_factory)
