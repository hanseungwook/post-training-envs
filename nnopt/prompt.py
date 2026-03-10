"""Prompt construction from TaskSpec."""

from __future__ import annotations

from nnopt.task import TaskSpec


def build_prompt(task: TaskSpec) -> str:
    """Build the full prompt string for an LLM from a TaskSpec."""
    compare_op = ">=" if task.higher_is_better else "<="

    input_shape_str = ", ".join(str(d) for d in task.input_shape)
    input_with_batch = f"(N, {input_shape_str})"

    # Task description by type
    descriptions = {
        "classification": f"Train a classifier on {task.dataset_name} ({task.num_classes} classes).",
        "regression": f"Train a regression model on {task.dataset_name}.",
        "autoencoding": f"Train an autoencoder to reconstruct {task.dataset_name} images.",
        "segmentation": f"Train a segmentation model on {task.dataset_name} ({task.num_classes} classes).",
        "sequence": f"Train a sequence classifier on {task.dataset_name} ({task.num_classes} classes).",
    }
    description = descriptions.get(task.task_type, f"Train a model on {task.dataset_name}.")

    prompt = f"""\
You are an expert PyTorch engineer. Write efficient neural network code for the following task.

## Task
{description}
- Dataset: {task.dataset_name}
- Input shape: {input_shape_str}
- Training samples: {task.train_subset_size:,}
- Evaluation samples: {task.eval_subset_size:,}

## Constraints
- Maximum parameters: {task.max_params:,}
- Target {task.metric_name}: {compare_op} {task.target_metric}
- Maximum inference latency: {task.max_inference_ms}ms (batch of 32)
- Training time budget: {task.max_train_time_s:.0f}s

## Interface
Write a function `solution(train_data, test_data, device)` that:
- `train_data` = (X_train, y_train) where X_train: {input_with_batch}, y_train: {task.output_description}
- `test_data` = (X_test, y_test) — same format
- `device` = "{("cuda:0" if True else "cpu")}"
- Returns: a trained `nn.Module` ready for evaluation

You have full control: define your model, create DataLoaders, choose augmentation,
optimizer, lr schedule, loss function, and training loop.

## Allowed imports
torch, torch.nn, torch.nn.functional, torch.optim, torch.optim.lr_scheduler,
torch.utils.data, torchvision.transforms, math, numpy, random, collections,
itertools, functools

Return your code in a ```python block.
"""
    return prompt
