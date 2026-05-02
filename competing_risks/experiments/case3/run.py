"""SoftComp @ Case III."""

from competing_risks.data.case3 import generate_case3
from competing_risks.experiments.runner import cli_main, make_functional_softcomp_spec


def load_data(args):
    if args.quick:
        return generate_case3(n_train=512, n_test=256, seed=0)
    return generate_case3(seed=0)


def model_spec_factory(data):
    return make_functional_softcomp_spec(
        hidden_dim=32,
        num_blocks=1,
        embed_dim=8,
        dropout=0.0,
        fit_kwargs={
            "epochs": 500,
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
    cli_main("case3", load_data, model_spec_factory)
