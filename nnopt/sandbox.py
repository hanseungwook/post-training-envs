"""Sandboxed subprocess execution of LLM-generated code."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass

import torch

from nnopt.task import TaskSpec
from nnopt.utils import validate_imports, count_parameters


@dataclass
class SandboxResult:
    success: bool
    metric_value: float | None = None
    param_count: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    error_type: str | None = None  # "parse", "import", "runtime", "timeout", "constraint"


@dataclass
class _WorkerConfig:
    """Picklable config passed to the subprocess (no callables)."""
    metric_name: str
    higher_is_better: bool
    input_shape: tuple
    num_classes: int | None
    task_type: str


def run_in_sandbox(
    code: str,
    task: TaskSpec,
    device: str = "cuda:0",
    timeout_buffer: float = 30.0,
) -> SandboxResult:
    """
    Execute LLM code in a subprocess, measure results, return SandboxResult.
    """
    # 1. Validate imports before execution
    violations = validate_imports(code)
    if violations:
        return SandboxResult(
            success=False,
            error="; ".join(violations),
            error_type="import",
        )

    # 2. Load data in the parent process (avoids pickling callables)
    try:
        train_data = task.get_train_data()
        test_data = task.get_test_data()
    except Exception as e:
        return SandboxResult(
            success=False,
            error=f"Failed to load data: {e}",
            error_type="runtime",
        )

    # 3. Build picklable config
    config = _WorkerConfig(
        metric_name=task.metric_name,
        higher_is_better=task.higher_is_better,
        input_shape=task.input_shape,
        num_classes=task.num_classes,
        task_type=task.task_type,
    )

    # 4. Run in subprocess
    result_queue = mp.Queue()
    timeout = task.max_train_time_s + timeout_buffer

    proc = mp.Process(
        target=_worker,
        args=(code, train_data, test_data, config, device, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return SandboxResult(
            success=False,
            error=f"Execution timed out after {timeout:.0f}s",
            error_type="timeout",
        )

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return SandboxResult(
            success=False,
            error="Worker process died without returning results",
            error_type="runtime",
        )

    return result


def _worker(
    code: str,
    train_data: tuple,
    test_data: tuple,
    config: _WorkerConfig,
    device: str,
    result_queue: mp.Queue,
) -> None:
    """Worker function that runs in a subprocess."""
    try:
        # Execute user code
        namespace = {}
        exec(compile(code, "<llm_code>", "exec"), namespace)

        if "solution" not in namespace:
            result_queue.put(SandboxResult(
                success=False,
                error="Code must define a `solution(train_data, test_data, device)` function",
                error_type="runtime",
            ))
            return

        # Run solution
        model = namespace["solution"](train_data, test_data, device)

        if model is None:
            result_queue.put(SandboxResult(
                success=False,
                error="solution() returned None instead of nn.Module",
                error_type="runtime",
            ))
            return

        model.eval()
        if device.startswith("cuda"):
            model = model.to(device)

        # Count parameters
        param_count = count_parameters(model)

        # Evaluate metric
        metric_value = _evaluate_metric(model, test_data, config, device)

        # Measure latency
        latency_ms = _measure_latency(model, config.input_shape, device)

        result_queue.put(SandboxResult(
            success=True,
            metric_value=metric_value,
            param_count=param_count,
            latency_ms=latency_ms,
        ))

    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(SandboxResult(
            success=False,
            error=f"{type(e).__name__}: {e}\n{tb}",
            error_type="runtime",
        ))


def _evaluate_metric(
    model: torch.nn.Module,
    test_data: tuple,
    config: _WorkerConfig,
    device: str,
) -> float:
    """Compute the task's metric on test data."""
    X_test, y_test = test_data
    X_test = X_test.to(device)

    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 256
        outputs = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i : i + batch_size]
            out = model(batch)
            outputs.append(out.cpu())
        preds_raw = torch.cat(outputs, dim=0)

    if config.metric_name == "accuracy":
        preds = preds_raw.argmax(dim=1)
        correct = (preds == y_test).float().mean().item()
        return correct

    elif config.metric_name == "mse":
        mse = ((preds_raw.squeeze() - y_test.float()) ** 2).mean().item()
        return mse

    elif config.metric_name == "r2":
        y = y_test.float()
        ss_res = ((preds_raw.squeeze() - y) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        return 1 - ss_res / (ss_tot + 1e-8)

    elif config.metric_name == "ssim":
        return _compute_ssim(preds_raw, y_test)

    elif config.metric_name == "pixel_accuracy":
        preds = preds_raw.argmax(dim=1)
        correct = (preds == y_test).float().mean().item()
        return correct

    elif config.metric_name == "miou":
        return _compute_miou(preds_raw, y_test, config.num_classes)

    else:
        raise ValueError(f"Unknown metric: {config.metric_name}")


def _compute_ssim(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute SSIM between predicted and target images."""
    preds = preds.float().clamp(0, 1)
    targets = targets.float()
    if targets.max() > 1:
        targets = targets / 255.0

    # Ensure 4D (B, C, H, W)
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = preds.mean(dim=(-2, -1), keepdim=True)
    mu_y = targets.mean(dim=(-2, -1), keepdim=True)

    sigma_x_sq = ((preds - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_y_sq = ((targets - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_xy = ((preds - mu_x) * (targets - mu_y)).mean(dim=(-2, -1), keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean().item()


def _compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Compute mean IoU for segmentation."""
    preds = preds.argmax(dim=1).flatten()
    targets = targets.flatten()

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0.0


def _measure_latency(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str,
    batch_size: int = 32,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    """Measure inference latency in milliseconds for a batch."""
    dummy = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(dummy)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / repeats) * 1000  # ms
