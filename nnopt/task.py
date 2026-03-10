"""TaskSpec dataclass and task registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

_REGISTRY: dict[str, "TaskSpec"] = {}


@dataclass
class TaskSpec:
    task_id: str
    task_type: str  # "classification" | "segmentation" | "regression" | "autoencoding" | "sequence"
    dataset_name: str

    # Constraints
    max_params: int
    target_metric: float
    metric_name: str  # "accuracy", "miou", "mse", "ssim", "r2", "pixel_accuracy"
    higher_is_better: bool
    max_inference_ms: float

    # Training budget
    max_train_time_s: float
    train_subset_size: int
    eval_subset_size: int

    # Data shape info
    input_shape: tuple
    num_classes: int | None
    output_description: str

    # Dataset access
    get_train_data: Callable = field(repr=False)
    get_test_data: Callable = field(repr=False)
    baseline_metric: float

    # Difficulty tier
    tier: int  # 1=easy, 2=medium, 3=hard

    # Reference solution
    reference_code: str = field(repr=False, default="")


def register_task(spec: TaskSpec) -> None:
    """Register a task spec in the global registry."""
    if spec.task_id in _REGISTRY:
        raise ValueError(f"Task '{spec.task_id}' already registered")
    _REGISTRY[spec.task_id] = spec


def get_task(task_id: str) -> TaskSpec:
    """Retrieve a registered task by ID."""
    # Ensure all task modules are imported
    _ensure_loaded()
    if task_id not in _REGISTRY:
        raise KeyError(f"Unknown task '{task_id}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[task_id]


def list_tasks() -> list[str]:
    """Return sorted list of all registered task IDs."""
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


_loaded = False


def _ensure_loaded():
    global _loaded
    if not _loaded:
        import nnopt.tasks  # noqa: F401
        _loaded = True
