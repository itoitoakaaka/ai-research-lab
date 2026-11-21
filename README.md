# AI Research Lab

**Learning neural networks through controlled experiments** · Lab A: PyTorch nonlinear regression

An educational, reproducible experiment comparing how dataset size, observation noise, model capacity, and learning rate influence model fitting and generalization. All training data are **synthetic**; no human participant data are included.

## Research question

How do training-set size, noise, network architecture, and learning rate change training and validation performance on a simple nonlinear regression task?

Ground-truth function: `y = sin(3x) + 0.3x`

## Experiment

- Model: fully connected neural network with Tanh activations
- Optimizer: Adam; loss: mean squared error (MSE)
- Baseline: 60 training samples, noise SD = 0.2, 2 hidden layers × 32 units, learning rate = 0.01
- Conditions: baseline, n=20, n=300, noise SD=0.6, 4 hidden layers, learning rate=0.0001, learning rate=0.05, 128 units
- Previously reported demonstration: **8 conditions × 5 random seeds = 40 fits**, 200 epochs

## Results

Mean MSE across five seeds, as reported in the initial Lab A run (lower is better).

| Condition | Train MSE | Validation MSE | MSE vs true function |
|---|---:|---:|---:|
| Baseline | 0.0321 | 0.0565 | 0.0179 |
| 20 training samples | 0.0174 | 0.1046 | 0.0635 |
| 300 training samples | 0.0394 | 0.0459 | 0.0069 |
| Noise SD = 0.6 | 0.2543 | 0.4712 | 0.1197 |
| 4 hidden layers | 0.0318 | 0.0590 | 0.0203 |
| Learning rate = 0.0001 | 0.4536 | 0.4756 | 0.4372 |
| Learning rate = 0.05 | 0.0275 | 0.0574 | 0.0194 |
| Width = 128 units | 0.0387 | 0.0627 | 0.0240 |

Source: [reported summary](results/condition_summary_reported.csv). These numbers are documentation of the previous demonstration, **not freshly verified outputs from the cleaned-up script**.

### Interpretation

The 20-sample condition has a lower training error but a higher validation error than the baseline. This is consistent with impaired generalization, but does not alone prove a specific overfitting mechanism. More samples improved validation performance under the conditions tested. Very small learning rate appears to have left the model undertrained at 200 epochs. More parameters did not consistently improve validation error.

A five-seed comparison is a teaching demonstration, not a definitive hyperparameter search or hypothesis test.

## Reproduce the experiment

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py --epochs 200 --seeds 5
```

Outputs are saved in `results/`:
- `run_metrics.csv`: per-run metrics
- `learning_curves.csv`: training and validation loss by epoch
- `predictions.csv`: model predictions and true function
- `condition_summary.csv`: per-condition mean and standard deviation

**Reproducibility note:** This repository contains a cleaned-up implementation of the experiment. Exact numerical identity with the previously reported data has **not** been checked. Software versions, random streams, and platform can affect results. Avoid treating the published table as a verified regression test.

## Repository contents

| Path | Description |
|---|---|
| [run_experiments.py](run_experiments.py) | PyTorch training loop and eight configurations |
| [requirements.txt](requirements.txt) | Python dependencies |
| [results/condition_summary_reported.csv](results/condition_summary_reported.csv) | Previously reported result summary |
| [docs/EXPERIMENT_NOTE.md](docs/EXPERIMENT_NOTE.md) | Japanese experiment note / Zotero note |
| [docs/references.ris](docs/references.ris) | Reference import file for Zotero |

## Limitations and next steps

1. Use the same independent holdout/test data across conditions and evaluate sensitivity to dataset splits.
2. Track variance and show learning curves, confidence intervals, and prediction plots.
3. Distinguish model capacity effects from insufficient optimization.
4. Expand the project toward Transformer introspection (Lab B) and human sensorimotor learning models (Lab C).

This project is **educational work**, not a peer-reviewed research finding or clinical/neuroscience inference.
