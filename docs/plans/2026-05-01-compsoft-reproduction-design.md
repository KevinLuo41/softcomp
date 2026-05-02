# CompSoft Reproduction Design

## Goal

Reconstruct the CompSoft competing-risks project from the supplied HTML blueprint.
The repository should match the documented file structure and provide runnable
implementations for the model, training loop, postprocessing, simulation data,
real-data loader hooks, metrics, visualization, checkpoints, and six experiment
entry points.

## Scope

The HTML is a blueprint, not a source archive. The reconstruction therefore
targets 1:1 public structure and behavior from the document:

- `competing_risks/compsoft_model/` implements `CompSoftNet`,
  `BaseCompSoftNet`, `FunctionalCompSoftNet`, NLL loss, time augmentation,
  Brier-augmented training, fitting, and CIF prediction.
- `competing_risks/data/` implements simulation cases I/II/III plus loaders
  for PBC, Framingham, and Synthetic data with the preprocessing described in
  the blueprint.
- `competing_risks/evaluation/` implements simulation MSE/accuracy, IPCW IBS,
  Antolini-style time-dependent concordance, isotonic projection, checkpoints,
  and plots.
- `competing_risks/experiments/` implements shared experiment plumbing and one
  `run.py` for each of the six datasets.

## Data Contract

All loaders return a dictionary with train/test arrays:

- `X_train`, `Y_train`, `Delta_train`
- `X_test`, `Y_test`, `Delta_test`
- optional simulation truth callbacks or grids
- `num_causes`, `feature_names`, and metadata

Simulated cases are fully self-contained. Real datasets require local CSV files.
If a file is missing, the loader raises an actionable error that names the
expected path and columns.

## Model Contract

The scalar CompSoft model consumes `(x, t)` and outputs `K` logits. A zero logit is
prepended as the survival class, then softmax produces `(F_1, ..., F_K, S)`.
The functional model embeds each functional covariate over its grid, appends
time, and uses the same residual backbone.

Training uses Adam, cosine annealing, manual minibatching, optional class
weights, time augmentation, optional Brier augmentation, and optional patience.
By default experiments use all training data and no validation split.

## Verification

Tests should cover tensor shapes, probability normalization, monotone isotonic
projection, simulation generation, evaluation grids, metrics, checkpoint round
trips, and a short smoke training run.
