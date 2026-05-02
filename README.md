# CompSoft Competing Risks

This repository reconstructs the CompSoft competing-risks project from the
provided reproduction blueprint.

The implemented file structure mirrors the blueprint:

```text
competing_risks/
├── compsoft_model/
│   ├── compsoft.py
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

Outputs are written to `outputs/<dataset>/compsoft/`:

- `model.pt`
- `predictions.npz`
- `metrics.json`
- `history.pkl`
- optional plots

## Tests

```bash
pytest
```
