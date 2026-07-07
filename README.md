# Gradient Descent and Optimizers

[![CI](https://github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers/actions/workflows/ci.yml/badge.svg)](https://github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **production-grade** optimizer benchmarking library and interactive demo.  
Every major gradient descent variant, adaptive optimizer, and LR scheduler —  
implemented from scratch in NumPy, tested with pytest, tracked with MLflow,  
and deployable as a Gradio app on Hugging Face Spaces.

---

## Repository Structure

```
Gradient-Descent-and-Optimizers/
│
├── src/gdo/                    ← pip-installable Python package
│   ├── optimizers/             ← NumPy implementations (SGD, Adam, AdamW, Lion, …)
│   ├── landscapes/             ← Loss surfaces (Rosenbrock, Himmelblau, …) + plotter
│   ├── training/               ← PyTorch Trainer, MLP, CNN, ConvergenceTracker
│   ├── experiment/             ← MLflow ExperimentLogger wrapper
│   └── config.py               ← Pydantic configs for all experiments
│
├── configs/                    ← One YAML file per experiment
│   ├── adam_mnist.yaml
│   ├── adamw_mnist.yaml
│   ├── lion_mnist.yaml
│   └── scheduler_comparison.yaml
│
├── notebooks/                  ← Three experiment notebooks (import from src/gdo)
│   ├── 01_gradient_descent_variants.ipynb
│   ├── 02_adaptive_optimizers.ipynb
│   └── 03_lr_scheduling_and_techniques.ipynb
│
├── app/                        ← Gradio app (3 tabs, deployed to HF Spaces)
│   ├── app.py
│   └── components/
│
├── tests/                      ← pytest unit tests (optimizer math verified numerically)
│
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## What Is Covered

### Optimizers (NumPy from scratch + PyTorch wrappers)

| Optimizer | Key Property | Paper |
|---|---|---|
| Batch GD | Full-dataset gradient | Classic |
| SGD | Single-sample, noisy | Classic |
| Mini-Batch GD | Industry-standard batching | Classic |
| SGD + Momentum | Velocity accumulation | Polyak, 1964 |
| RMSProp | Divide by RMS of squared grads | Hinton, 2012 |
| Adam | Momentum + RMSProp + bias correction | Kingma & Ba, 2014 |
| AdamW | Adam + **decoupled** weight decay | Loshchilov & Hutter, 2019 |
| Lion | Sign of gradient + momentum | Chen et al., Google Brain, 2023 |

### LR Schedulers

| Scheduler | Used In |
|---|---|
| StepLR | Legacy ResNet training |
| CosineAnnealingLR | HAN, TFT, most modern vision |
| OneCycleLR | fast.ai training recipes |
| CyclicalLR | Leslie Smith, 2017 |
| Warmup + Cosine | BERT, GPT, ViT — Transformer standard |
| ReduceLROnPlateau | Any task with val metric monitoring |

### Loss Surfaces

| Surface | Property |
|---|---|
| Quadratic (ill-conditioned) | Convex — exposes oscillation |
| Rosenbrock | Non-convex banana valley — classic benchmark |
| Beale | Multiple local minima |
| Himmelblau | Four equal global minima |

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_gradient_descent_variants` | Batch GD / SGD / Mini-Batch / Momentum — from scratch, trajectory visualization on Quadratic + Rosenbrock + Himmelblau |
| `02_adaptive_optimizers` | RMSProp / Adam / AdamW / Lion — NumPy trajectories + PyTorch training on MNIST with MLflow comparison |
| `03_lr_scheduling_and_techniques` | LR schedules, warmup, gradient clipping, weight init (Xavier vs He), batch size effect, CIFAR-10 scheduler comparison with MLflow |

---

## Quick Start

```bash
# Clone and install (sets up deps + pre-commit hooks)
git clone https://github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers
cd Gradient-Descent-and-Optimizers
make install            # == pip install -e ".[dev,app]" && pre-commit install

# Run a training experiment from the CLI
python -m gdo --config configs/adam_mnist.yaml

# Common dev tasks (see Makefile)
make test               # pytest with coverage gate
make lint               # ruff
make format             # ruff format
make type               # mypy
make app                # launch the Gradio app locally

# Or with Docker
docker compose up

# View MLflow experiment results
mlflow ui               # open http://localhost:5000
```

> `make install` also wires up **pre-commit**, which runs ruff lint + format on every commit
> so style never drifts. Run it manually anytime with `pre-commit run --all-files`.

---

## Run Training Experiments

All hyperparameters live in `configs/`. Change a config, run the CLI:

```bash
# Compare all optimizers on MNIST
python -m gdo --config configs/sgd_mnist.yaml
python -m gdo --config configs/momentum_sgd_mnist.yaml
python -m gdo --config configs/rmsprop_mnist.yaml
python -m gdo --config configs/adam_mnist.yaml
python -m gdo --config configs/adamw_mnist.yaml
python -m gdo --config configs/lion_mnist.yaml

# All runs appear in MLflow under the same experiment name
mlflow ui
```

---

## Key Design Decisions

**`src/gdo/` is the product — not the notebooks.**  
Notebooks, the Gradio app, and the CLI all import from the same package.  
The same `Adam` class that is unit-tested runs in the notebooks and the live demo.

**Configs over code.**  
Every hyperparameter lives in a YAML file. Changing LR, optimizer, scheduler,  
model architecture, or dataset requires editing a config file — not the source code.

**Tests verify the math.**  
`test_adaptive.py` verifies that `AdamW` applies weight decay **before** the gradient step,  
not as L2 regularization added to the gradient. This is a real bug in many implementations.

**MLflow for every training run.**  
Every optimizer comparison in the notebooks logs to MLflow automatically.  
`mlflow ui` gives a structured comparison table with metric plots — no manual log parsing.

---

## AdamW vs Adam — The Most Important Distinction

```python
# Adam with L2 regularization (WRONG weight decay):
grad_with_l2 = grad + weight_decay * param   # WD is scaled by adaptive LR
new_param = param - lr * adam_update(grad_with_l2)

# AdamW (CORRECT decoupled weight decay):
param_wd = (1 - lr * weight_decay) * param   # WD applied at full strength
new_param = param_wd - lr * adam_update(grad)  # gradient step on WD param
```

AdamW is the default optimizer for all Transformer training (BERT, GPT, ViT).  
This repo has a dedicated test (`test_adaptive.py::TestAdamW::test_adamw_not_equal_adam_with_l2`)  
that numerically verifies the two produce different results with the same weight decay value.

---

## Stack

| Layer | Technology |
|---|---|
| Optimizer implementations | Pure NumPy |
| Neural network training | PyTorch 2.x |
| Experiment tracking | MLflow |
| Config validation | Pydantic v2 |
| Web app | Gradio 4.x |
| Tests | pytest |
| Linting | ruff + black |
| Type checking | mypy |
| Containerization | Docker + docker-compose |
| CI | GitHub Actions |
| Deployment | Hugging Face Spaces |

---

## Author

**Shivani Bokka**
