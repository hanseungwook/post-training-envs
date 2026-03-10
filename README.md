# nnopt — Neural Network Optimization Environment

An RL environment where an LLM learns to write small, fast, high-performing neural networks in PyTorch. The LLM receives a task description with explicit constraints (max parameters, target metric, latency budget) and generates full code — model architecture, training loop, data augmentation, lr schedule, custom losses — everything except raw data loading. The code is executed in a sandbox, results are measured, and a reward is returned.

```
Prompt (task + constraints) → LLM generates full PyTorch code → Sandbox executes → Measure (perf, speed, size) → Reward JSON
```

## Quick Start

```bash
pip install -e .
```

### Evaluate LLM-generated code on a task

```python
from nnopt import evaluate_code

result = evaluate_code("mnist-10k", """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, y_train = train_data
    X_train, y_train = X_train.to(device), y_train.to(device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(10):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()

    return model
""", device="cuda:0")

print(result)
# {
#   "task_id": "mnist-10k",
#   "reward": 0.85,
#   "success": true,
#   "metric_name": "accuracy",
#   "metric_value": 0.967,
#   "param_count": 9098,
#   "latency_ms": 0.42,
#   "constraint_satisfied": true,
#   "error": null
# }
```

### CLI

```bash
python -m nnopt.evaluate --task mnist-10k --code-file solution.py --device cuda:0
```

### List available tasks

```python
from nnopt import list_tasks, get_task
print(list_tasks())  # ['california-10k', 'cifar10-100k', 'mnist-10k', ...]
```

### Build a prompt for an LLM

```python
from nnopt import get_task
from nnopt.prompt import build_prompt

prompt = build_prompt(get_task("cifar10-100k"))
# Send this prompt to an LLM, get code back, pass to evaluate_code()
```

## How It Works

### Interface Contract

The LLM writes a function `solution(train_data, test_data, device)` that:
- Receives `(X_train, y_train)` and `(X_test, y_test)` as PyTorch tensors
- Has full control: model architecture, DataLoaders, augmentation, optimizer, lr schedule, loss function, training loop
- Returns a trained `nn.Module`

### Sandbox Execution

LLM code runs in a forked subprocess with:
- **Timeout**: `max_train_time_s` + 30s buffer, hard kill
- **Import whitelist** (AST scan before exec): `torch`, `torchvision.transforms`, `math`, `numpy`, `collections`, `itertools`, `functools`, `random`
- **Blocked**: `os`, `sys`, `subprocess`, `socket`, `requests`, `shutil`, `pathlib`

### Reward Function

Gated multi-objective scalar in **[-1.0, 1.6]**:

| Outcome | Reward |
|---------|--------|
| Code fails to parse / banned import | -1.0 |
| Runtime crash or timeout | -0.5 |
| Parameter count exceeds budget | -0.2 |
| Latency exceeds budget | -0.1 |
| Valid solution | 0.0 to 1.6 |

For valid solutions:
```
perf       = clip((metric - baseline) / (target - baseline), 0, 1)        # [0, 1.0]
exceed     = 0.3 * clip((metric - target) / target, 0, 1)                 # [0, 0.3]
efficiency = 0.15 * (1 - params/max) + 0.15 * (1 - latency/max_ms)       # [0, 0.3]
reward     = perf + exceed + efficiency                                    # [0, 1.6]
```

Natural curriculum: valid code → satisfy constraints → hit target → beat target efficiently.

## Tasks

18 tasks across 5 categories and 3 difficulty tiers:

### Classification

| Task ID | Dataset | Max Params | Target | Tier |
|---------|---------|-----------|--------|------|
| mnist-10k | MNIST | 10K | 97% acc | 1 |
| mnist-2k | MNIST | 2K | 95% acc | 2 |
| fashion-mnist-20k | FashionMNIST | 20K | 87% acc | 1 |
| fashion-mnist-5k | FashionMNIST | 5K | 86% acc | 2 |
| kmnist-15k | KMNIST | 15K | 92% acc | 1 |
| cifar10-100k | CIFAR-10 | 100K | 85% acc | 2 |
| cifar10-30k | CIFAR-10 | 30K | 82% acc | 3 |
| cifar100-200k | CIFAR-100 | 200K | 50% acc | 2 |
| cifar100-100k | CIFAR-100 | 100K | 45% acc | 3 |
| svhn-50k | SVHN | 50K | 90% acc | 2 |

### Regression

| Task ID | Dataset | Max Params | Target | Tier |
|---------|---------|-----------|--------|------|
| sinusoidal-1k | Synthetic sin(x) | 1K | 0.001 MSE | 1 |
| california-10k | CA Housing | 10K | 0.55 R² | 1 |
| california-3k | CA Housing | 3K | 0.55 R² | 2 |

### Autoencoding

| Task ID | Dataset | Max Params | Target | Tier |
|---------|---------|-----------|--------|------|
| mnist-ae-20k | MNIST | 20K | 0.93 SSIM | 2 |
| mnist-ae-5k | MNIST | 5K | 0.88 SSIM | 3 |

### Segmentation

| Task ID | Dataset | Max Params | Target | Tier |
|---------|---------|-----------|--------|------|
| oxford-pet-150k | Oxford Pets | 150K | 75% pix-acc | 2 |
| voc-seg-300k | VOC2012 | 300K | 30% mIoU | 3 |

### Sequence

| Task ID | Dataset | Max Params | Target | Tier |
|---------|---------|-----------|--------|------|
| imdb-50k | IMDB | 50K | 80% acc | 2 |

## Project Structure

```
nnopt/
├── task.py              # TaskSpec dataclass + registry
├── sandbox.py           # Sandboxed subprocess execution
├── reward.py            # Multi-objective reward computation
├── prompt.py            # Prompt construction from TaskSpec
├── evaluate.py          # Standalone entry point: code in → reward out
├── utils.py             # Code extraction, import validation
└── tasks/
    ├── classification/  # MNIST, FashionMNIST, KMNIST, CIFAR-10/100, SVHN
    ├── regression/      # sinusoidal, California Housing
    ├── autoencoding/    # MNIST autoencoder
    ├── segmentation/    # Oxford Pets, VOC2012
    └── sequence/        # IMDB sentiment

scripts/
├── download_data.py     # Pre-download all datasets
└── validate_tasks.py    # Run reference solutions through all tasks

tests/
├── test_sandbox.py      # Safety: banned imports, timeouts, runtime errors
├── test_reward.py       # Reward ordering, ranges, edge cases
└── test_tasks.py        # Task loading, prompt generation, code extraction
```

## Scripts

### Download datasets

```bash
python scripts/download_data.py
```

### Validate reference solutions (requires GPU)

```bash
python scripts/validate_tasks.py --device cuda:0
python scripts/validate_tasks.py --tier 1          # only easy tasks
python scripts/validate_tasks.py --tasks mnist-10k sinusoidal-1k
```

## Tests

```bash
pytest tests/ -v
```

All 32 tests pass covering sandbox safety (banned imports, timeouts, crash handling), reward computation (ordering, ranges, lower-is-better metrics), task registry, prompt generation, and code extraction.
