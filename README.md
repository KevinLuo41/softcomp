# SoftComp Competing Risks

This repository reconstructs the SoftComp competing-risks project from the
provided reproduction blueprint.

The implemented file structure mirrors the blueprint:

```text
competing_risks/
├── baseline_models/
│   ├── deephit.py
│   ├── dsm.py
│   ├── cs_cox.py
│   └── neural_fine_gray.py
├── softcomp_model/
│   ├── softcomp.py
│   ├── functional.py
│   └── __init__.py
├── evaluation/
│   ├── simulation.py
│   ├── survival.py
│   ├── postprocess.py
│   ├── checkpoints.py
│   └── visualize.py
├── data/
│   ├── utils.py
│   ├── case1.py
│   ├── case2.py
│   ├── case3.py
│   ├── pbc.py
│   ├── framingham.py
│   ├── synthetic.py
│   └── synthetic_comprisk.csv
├── experiments/
│   ├── runner.py
│   ├── case1/run.py
│   ├── case2/run.py
│   ├── case3/run.py
│   ├── pbc/run.py
│   ├── framingham/run.py
│   └── synthetic/run.py
└── README.md
cuda_check.py
```

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick smoke runs

```bash
python3 -m competing_risks.experiments.case1.run --quick --quick-epochs 3 --force
python3 -m competing_risks.experiments.case2.run --quick --quick-epochs 3 --force
python3 -m competing_risks.experiments.case3.run --quick --quick-epochs 3 --force
python3 -m competing_risks.experiments.synthetic.run --quick --quick-epochs 3 --force
python3 cuda_check.py
```

The local CLI also accepts the blueprint-style cache controls:

```bash
python3 -m competing_risks.experiments.case1.run --list
python3 -m competing_risks.experiments.case1.run --retrain softcomp
python3 -m competing_risks.experiments.case1.run --reeval softcomp
```

## Full reproduction runs

The simulation experiments are self-contained:

```bash
python3 -m competing_risks.experiments.case1.run --force
python3 -m competing_risks.experiments.case2.run --force
python3 -m competing_risks.experiments.case3.run --force
```

PBC and Framingham require local CSV files:

```bash
python3 -m competing_risks.experiments.pbc.run --data-path /path/to/pbc.csv --force
python3 -m competing_risks.experiments.framingham.run --data-path /path/to/framingham.csv --force
```

Synthetic uses `competing_risks/data/synthetic_comprisk.csv` when populated. If
that file is empty, the loader generates a deterministic DeepHit-like fallback:

```bash
python3 -m competing_risks.experiments.synthetic.run --force
```

Outputs are written to `outputs/<dataset>/softcomp/`:

- `model.pt`
- `predictions.npz`
- `metrics.json`
- `history.pkl`
- optional plots

The documented baseline modules are present under `competing_risks/baseline_models/`
with matching `fit`, `predict_cif`, and `load_from_checkpoint` APIs. They use a
lightweight empirical CIF estimator so the repository stays runnable with only
the dependencies in `requirements.txt`.

## Tests

```bash
pytest
```
