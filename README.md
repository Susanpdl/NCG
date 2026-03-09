# NCG — Novelty-triggered Capacity Growth

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Self-regulating continual learning that grows capacity only when needed.

## The Core Idea

Standard neural networks trained sequentially on multiple tasks suffer from **catastrophic forgetting**: performance on earlier tasks drops sharply as the model adapts to new data. NCG (Novelty-triggered Capacity Growth) tackles this by monitoring its own learning state through a **novelty signal** and three **learnable meta-parameters** (α, β, λ). It **autonomously expands** its hidden capacity only when three conditions are met at once: low novelty, sufficient regularisation (λ > 0.3), and a validation-accuracy plateau.

A central component is the **knowledge embedding K** with **gated write**: a fixed-size buffer that accumulates task knowledge via a learned gating mechanism. This allows the model to preserve and reuse representations across tasks without manual replay or architectural hand-tuning.

## How It Works

| Component | Description |
|-----------|-------------|
| **Meta-parameters** | α (exploration), β (complexity penalty), λ (regularisation) — trained by gradient ascent on a meta-loss. |
| **Knowledge embedding K** | Gated-write buffer that accumulates task knowledge; gate = sigmoid(gate_layer(h_mean)), K = (1−gate)·K + gate·h_mean. |
| **Growth trigger** | Fires when: novelty < 0.5, λ > 0.3, and val-acc plateau < 0.005 over the last 3 epochs. |
| **Growth** | Adds 64 hidden units; existing weights preserved, new weights Kaiming-initialised (fc1) or zero (fc2). |

## Results

**Table 1 — Split-MNIST** (mean ± std over seeds)

NCG:            Avg Acc=0.551, Forgetting=0.331, BWT=-0.407, FWT=0.024
NCG-NoGrowth:   Avg Acc=0.552, Forgetting=0.373, BWT=-0.466, FWT=0.039
NCG-FixedMeta:  Avg Acc=0.557, Forgetting=0.356, BWT=-0.445, FWT=0.051
DEN:            Avg Acc=0.580, Forgetting=0.417, BWT=-0.521, FWT=0.032
StaticMLP-256:  Avg Acc=0.579, Forgetting=0.419, BWT=-0.524, FWT=0.035
StaticMLP-448:  Avg Acc=0.573, Forgetting=0.421, BWT=-0.531, FWT=0.027
StaticMLP-512:  Avg Acc=0.572, Forgetting=0.425, BWT=-0.531, FWT=0.034
EWC:            Avg Acc=0.732, Forgetting=0.229, BWT=-0.286, FWT=0.026

**Table 2 — Split-CIFAR-10** (mean ± std over seeds)

NCG:            Avg Acc=0.673, Forgetting=0.084, BWT=-0.086, FWT=0.061
NCG-NoGrowth:   Avg Acc=0.666, Forgetting=0.103, BWT=-0.108, FWT=0.076
NCG-FixedMeta:  Avg Acc=0.673, Forgetting=0.096, BWT=-0.119, FWT=0.077
DEN:            Avg Acc=0.688, Forgetting=0.222, BWT=-0.278, FWT=0.088
StaticMLP-256:  Avg Acc=0.683, Forgetting=0.230, BWT=-0.288, FWT=0.086
StaticMLP-512:  Avg Acc=0.687, Forgetting=0.227, BWT=-0.284, FWT=0.088
EWC:            Avg Acc=0.702, Forgetting=0.163, BWT=-0.203, FWT=0.088

**63% forgetting reduction vs StaticMLP-256 on Split-CIFAR-10 (p < 0.0001).**

## Installation

```bash
pip install ncg-torch
```

Or from source:

```bash
git clone https://github.com/Ami-Darshan/NCG.git
cd NCG
pip install -e ".[dev]"
```

## Quick Start

```python
import ncg
from ncg.metrics import compute_forgetting

device = ncg.get_device()
ncg.set_seed(42)
model = ncg.NCGModel(hidden_size=256, num_classes=2, max_hidden=512)
tasks = ncg.get_split_mnist_tasks(data_dir="./data", batch_size=64)
res = ncg.train_ncg(model, tasks, device, epochs_per_task=2, verbose=True)
forgetting = compute_forgetting({"NCG": res["task_accs"]})["NCG"]
print(f"Final forgetting: {forgetting:.4f}")
```

## Running Full Experiments

```bash
python scripts/main.py --benchmark split_mnist --seeds 42 43 44 45 46 47 48 49 50 51
python scripts/main.py --benchmark split_cifar10 --seeds 42 43 44 45 46 47 48 49 50 51
```

## Convergence Diagnostics

Check whether meta-parameters α, β, λ are converging to a fixed point or merely decaying:

```python
import pickle
from ncg.math.convergence import run_diagnostics

with open("results/ncg_logs.pkl", "rb") as f:
    data = pickle.load(f)
run_diagnostics(data["ncg_logs"][0])
```

## Citation

```bibtex
@article{poudel2025ncg,
  title   = {Novelty-triggered Capacity Growth for Continual Learning},
  author  = {Poudel, Darshan},
  year    = {2025},
  note    = {Preprint. Under review.},
  url     = {https://github.com/Ami-Darshan/NCG}
}
```

## License

MIT. See [LICENSE](LICENSE).
