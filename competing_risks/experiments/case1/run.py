"""SoftComp @ Case I."""

from competing_risks.data.case1 import generate_case1
from competing_risks.experiments.runner import cli_main, make_softcomp_spec


def load_data(args):
    if args.quick:
        return generate_case1(n_train=512, n_test=256, seed=0)
    return generate_case1(seed=0)


def model_spec_factory(data):
    return make_softcomp_spec(
        hidden_dim=16,
        num_blocks=1,
        dropout=0.0,
        fit_kwargs={
            "epochs": 1000,
            "lr": 1e-3,
            "weight_decay": 3e-3,
            "batch_size": 256,
            "n_aug": 1,
            "aug_weight": 0.5,
            "verbose": True,
            "seed": 0,
        },
    )


if __name__ == "__main__":
    cli_main("case1", load_data, model_spec_factory)
