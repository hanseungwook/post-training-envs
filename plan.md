# Post-Training Environment: Neural Network Optimization

## Context
We want an RL environment where an LLM learns to write small, fast, high-performing neural networks in PyTorch. The LLM receives a task description with explicit constraints (max parameters, target metric, latency budget) and generates **full code** — model architecture, training loop, data augmentation, lr schedule, custom losses — everything except raw data loading. We execute it in a sandbox, measure results, and return a reward. This is a **standalone** evaluator that any RL framework can consume.

## Flow
```
Prompt (task + constraints) → LLM generates full PyTorch code → Sandbox executes → Measure (perf, speed, size) → Reward JSON
```

## Directory Structure

```
post-training-envs/
├── nnopt/                              # Core package
│   ├── __init__.py
│   ├── task.py                         # TaskSpec dataclass + registry
│   ├── sandbox.py                      # Sandboxed subprocess execution
│   ├── reward.py                       # Multi-objective reward computation
│   ├── prompt.py                       # Prompt construction from TaskSpec
│   ├── evaluate.py                     # Standalone entry point: code in → reward out
│   ├── utils.py                        # Code extraction, import validation
│   │
│   └── tasks/
│       ├── __init__.py                 # Imports all task modules to register them
│       │
│       ├── classification/
│       │   ├── __init__.py
│       │   ├── mnist.py                # MNIST tiers + reference solution
│       │   ├── fashion_mnist.py
│       │   ├── cifar10.py
│       │   ├── cifar100.py
│       │   ├── svhn.py
│       │   └── kmnist.py
│       │
│       ├── segmentation/
│       │   ├── __init__.py
│       │   ├── oxford_pet.py
│       │   └── voc2012.py
│       │
│       ├── regression/
│       │   ├── __init__.py
│       │   ├── california_housing.py
│       │   └── sinusoidal.py
│       │
│       ├── autoencoding/
│       │   ├── __init__.py
│       │   └── mnist_ae.py
│       │
│       └── sequence/
│           ├── __init__.py
│           └── imdb_sentiment.py
│
├── scripts/
│   ├── download_data.py                # Pre-download all datasets
│   └── validate_tasks.py               # Run reference solutions through all tasks
│
├── tests/
│   ├── test_sandbox.py
│   ├── test_reward.py
│   └── test_tasks.py
│
└── pyproject.toml
```

## Core Design

### 1. TaskSpec — task definition

```python
@dataclass
class TaskSpec:
    task_id: str                        # "cifar10-100k"
    task_type: str                      # "classification" | "segmentation" | ...
    dataset_name: str                   # "CIFAR10"

    # Constraints
    max_params: int                     # Hard parameter budget
    target_metric: float                # Performance target
    metric_name: str                    # "accuracy", "miou", "mse", "ssim", "r2"
    higher_is_better: bool
    max_inference_ms: float             # Latency budget per batch

    # Training budget
    max_train_time_s: float             # Wall-clock timeout for entire execution
    train_subset_size: int              # Rows available for training
    eval_subset_size: int               # Rows used for evaluation

    # Data shape info (included in prompt so LLM knows dimensions)
    input_shape: tuple                  # (3, 32, 32) or (8,) for tabular
    num_classes: int | None             # For classification/segmentation
    output_description: str             # "logits (B, 10)" or "pixel mask (B, 21, H, W)"

    # Dataset access
    get_train_data: Callable            # Returns (X_train, y_train) as tensors
    get_test_data: Callable             # Returns (X_test, y_test) as tensors
    baseline_metric: float              # Random-chance performance (reward floor)

    # Difficulty tier
    tier: int                           # 1=easy, 2=medium, 3=hard

    # Reference solution (for validation)
    reference_code: str                 # Known-good solution
```

**Key decision: Full code control.** The LLM writes everything: model class, Dataset/DataLoader creation, transforms, augmentation, optimizer, lr schedule, training loop. The environment only provides raw data tensors (`X_train`, `y_train`, `X_test`, `y_test`) and evaluates the final model. This gives the LLM maximum creative freedom — it can invent custom augmentation, unusual architectures, novel training tricks.

**What the sandbox provides to the LLM's code:**
- `train_data`: tuple of `(X, y)` tensors (or just `X` for autoencoding)
- `test_data`: tuple of `(X, y)` tensors
- `device`: string like `"cuda:0"`
- The code must define a function `solution(train_data, test_data, device)` that returns a trained `nn.Module`

### 2. Sandbox — subprocess execution

Run LLM code in `multiprocessing.Process`:
- **Timeout**: `max_train_time_s + 30s` buffer, hard kill
- **GPU**: `CUDA_VISIBLE_DEVICES` isolation
- **Import whitelist** (AST scan before exec): `torch`, `torchvision.transforms`, `math`, `numpy`, `collections`, `itertools`, `functools`, `random`
- **No**: `os`, `sys`, `subprocess`, `socket`, `requests`, `shutil`, `pathlib`

**Execution protocol:**
```python
# In subprocess:
1. Load raw data tensors
2. exec(llm_code, namespace)    # LLM code defines solution()
3. model = namespace["solution"](train_data, test_data, device)
4. param_count = count_params(model)
5. metric = evaluate(model, test_data, task_spec)
6. latency = measure_latency(model, input_shape, device)
7. Return {metric, param_count, latency, success, error}
```

**Implementation note:** Data is loaded in the parent process and passed as tensors to the forked subprocess. This avoids pickling issues with callable fields on `TaskSpec` and keeps dataset download logic out of the sandbox.

### 3. Reward — gated multi-objective scalar

```
Code fails to parse:           -1.0
Code crashes at runtime:       -0.5
Param count > max_params:      -0.2
Latency > max_inference_ms:    -0.1
Otherwise:
  perf = clip((metric - baseline) / (target - baseline), 0, 1)    # [0, 1]
  exceed_bonus = 0.3 * clip((metric - target) / target, 0, 1)     # [0, 0.3]
  efficiency = 0.15*(1 - params/max) + 0.15*(1 - latency/max_ms)  # [0, 0.3]
  reward = perf + exceed_bonus + efficiency                        # [0, 1.6]
```

Total range: **[-1.0, 1.6]**. Natural curriculum: valid code → satisfy constraints → hit target → beat target efficiently.

### 4. Standalone evaluator interface

```python
# nnopt/evaluate.py
def evaluate_code(task_id: str, code: str, device: str = "cuda:0") -> dict:
    """
    Main entry point. Returns:
    {
        "task_id": "cifar10-100k",
        "reward": 0.85,
        "success": true,
        "metric_name": "accuracy",
        "metric_value": 0.867,
        "param_count": 87432,
        "latency_ms": 1.23,
        "constraint_satisfied": true,
        "error": null
    }
    """
```

Any RL framework calls this function (or invokes it as CLI: `python -m nnopt.evaluate --task cifar10-100k --code-file solution.py`).

### 5. Prompt template

```
You are an expert PyTorch engineer. Write efficient neural network code for the following task.

## Task
{description} — dataset: {dataset_name}, input shape: {input_shape}

## Constraints
- Maximum parameters: {max_params:,}
- Target {metric_name}: {">=" if higher_is_better else "<="} {target_metric}
- Maximum inference latency: {max_inference_ms}ms (batch of 32)
- Training time budget: {max_train_time_s}s

## Interface
Write a function `solution(train_data, test_data, device)` that:
- `train_data` = (X_train, y_train) where X_train: {input_shape_with_N}, y_train: {output_desc}
- `test_data` = (X_test, y_test) same format
- `device` = "cuda:0" or "cpu"
- Returns: a trained `nn.Module` ready for evaluation

You have full control: define your model, create DataLoaders, choose augmentation,
optimizer, lr schedule, loss function, and training loop.

## Allowed imports
torch, torch.nn, torch.nn.functional, torch.optim, torch.optim.lr_scheduler,
torch.utils.data, torchvision.transforms, math, numpy, random, collections,
itertools, functools

Return your code in a ```python block.
```

## Task List

### Classification

| Task ID | Dataset | Max Params | Target | Latency | Tier | Notes |
|---------|---------|-----------|--------|---------|------|-------|
| mnist-10k | MNIST | 10K | 97% acc | 2ms | 1 | Trivial — learn to write valid code |
| mnist-2k | MNIST | 2K | 95% acc | 1ms | 2 | Forces efficient architecture |
| fashion-mnist-20k | FashionMNIST | 20K | 87% acc | 2ms | 1 | Slightly harder |
| fashion-mnist-5k | FashionMNIST | 5K | 86% acc | 1ms | 2 | Tight param budget |
| kmnist-15k | KMNIST | 15K | 92% acc | 2ms | 1 | Japanese chars, same shape as MNIST |
| cifar10-100k | CIFAR-10 | 100K | 85% acc | 3ms | 2 | Needs convolutions |
| cifar10-30k | CIFAR-10 | 30K | 82% acc | 1.5ms | 3 | Needs depthwise-sep or similar |
| cifar100-200k | CIFAR-100 | 200K | 50% acc | 5ms | 2 | 100 classes, capacity matters |
| cifar100-100k | CIFAR-100 | 100K | 45% acc | 3ms | 3 | Very tight for 100 classes |
| svhn-50k | SVHN | 50K | 90% acc | 2ms | 2 | House numbers, real-world data |

### Regression

| Task ID | Dataset | Max Params | Target | Latency | Tier | Notes |
|---------|---------|-----------|--------|---------|------|-------|
| sinusoidal-1k | Synthetic sin(x) | 1K | 0.001 MSE | 0.5ms | 1 | Trivial MLP task |
| california-10k | CA Housing | 10K | 0.55 R² | 0.5ms | 1 | Tabular regression |
| california-3k | CA Housing | 3K | 0.55 R² | 0.5ms | 2 | Tight budget |

### Autoencoding

| Task ID | Dataset | Max Params | Target | Latency | Tier | Notes |
|---------|---------|-----------|--------|---------|------|-------|
| mnist-ae-20k | MNIST | 20K | 0.93 SSIM | 2ms | 2 | Encoder-decoder pattern |
| mnist-ae-5k | MNIST | 5K | 0.88 SSIM | 1ms | 3 | Very tight |

### Segmentation

| Task ID | Dataset | Max Params | Target | Latency | Tier | Notes |
|---------|---------|-----------|--------|---------|------|-------|
| oxford-pet-150k | Oxford Pets | 150K | 75% pix-acc | 10ms | 2 | Binary fg/bg |
| voc-seg-300k | VOC2012 | 300K | 30% mIoU | 15ms | 3 | 21 classes, challenging |

### Sequence

| Task ID | Dataset | Max Params | Target | Latency | Tier | Notes |
|---------|---------|-----------|--------|---------|------|-------|
| imdb-50k | IMDB (200 tokens) | 50K | 80% acc | 3ms | 2 | Needs embeddings + temporal modeling |

**Total: 18 tasks across 5 categories, 3 difficulty tiers.**

## Implementation Plan

### Phase 1: Core framework
1. `nnopt/task.py` — TaskSpec dataclass + registry (`register_task`, `get_task`, `list_tasks`)
2. `nnopt/utils.py` — Extract python code from markdown, AST-based import whitelist check
3. `nnopt/sandbox.py` — Subprocess executor: fork, exec, timeout, measure metrics, return result
4. `nnopt/reward.py` — Gated multi-objective reward computation
5. `nnopt/prompt.py` — Build prompt string from TaskSpec
6. `nnopt/evaluate.py` — Standalone entry point: `evaluate_code(task_id, code) → dict`

### Phase 2: All tasks (each task file = ~50-80 lines: register specs + get_data + reference solution)
7. `nnopt/tasks/classification/` — mnist, fashion_mnist, cifar10, cifar100, svhn, kmnist
8. `nnopt/tasks/regression/` — sinusoidal, california_housing
9. `nnopt/tasks/autoencoding/` — mnist_ae
10. `nnopt/tasks/segmentation/` — oxford_pet, voc2012
11. `nnopt/tasks/sequence/` — imdb_sentiment

### Phase 3: Scripts & tests
12. `scripts/download_data.py` — Pre-download all datasets to a cache dir
13. `scripts/validate_tasks.py` — Run each task's reference solution, verify metrics + reward
14. `tests/` — Unit tests for sandbox safety, reward computation, task loading
15. `pyproject.toml`

## Verification
1. `python scripts/validate_tasks.py` — runs every reference solution through the full pipeline (sandbox → metrics → reward), prints pass/fail per task
2. **Failure mode tests**: feed broken code, infinite loops, OOM code, banned imports → verify graceful failure + correct negative reward
3. **Reward ordering**: verify crash < constraint violation < bad performance < good performance < efficient solution
4. **CLI test**: `python -m nnopt.evaluate --task mnist-10k --code-file my_solution.py` works end-to-end
5. **LLM integration test**: prompt an actual LLM, pipe output through evaluator, verify sensible reward

## Deviations from Original Plan
- **Removed `RLIMIT_AS`**: The 4GB virtual memory limit was too restrictive for PyTorch's memory mapping and killed worker processes before they could load torch. Removed in favor of timeout-based safety.
- **Data loading moved to parent process**: Originally data was loaded inside the subprocess, but this caused pickling issues with `spawn` context and redundant dataset downloads. Now data tensors are loaded in the parent and passed to the forked child.
- **18 tasks instead of 19**: The original plan counted 19 but the actual unique task list has 18.
