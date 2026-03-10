"""Gated multi-objective reward computation."""

from __future__ import annotations

from nnopt.sandbox import SandboxResult
from nnopt.task import TaskSpec


def compute_reward(result: SandboxResult, task: TaskSpec) -> float:
    """
    Compute scalar reward from sandbox result and task spec.

    Reward ranges:
        -1.0  : code failed to parse or import violation
        -0.5  : code crashed at runtime
        -0.2  : parameter count exceeds max_params
        -0.1  : latency exceeds max_inference_ms
        [0, 1.6]: success — performance + exceed bonus + efficiency
    """
    if not result.success:
        if result.error_type in ("parse", "import"):
            return -1.0
        return -0.5  # runtime error or timeout

    # Constraint checks
    if result.param_count is not None and result.param_count > task.max_params:
        return -0.2

    if result.latency_ms is not None and result.latency_ms > task.max_inference_ms:
        return -0.1

    # Performance reward
    metric = result.metric_value
    baseline = task.baseline_metric
    target = task.target_metric

    if task.higher_is_better:
        # Higher is better (accuracy, r2, ssim, miou, pixel_accuracy)
        perf_raw = (metric - baseline) / (target - baseline + 1e-8)
        perf = max(0.0, min(1.0, perf_raw))

        exceed_raw = (metric - target) / (abs(target) + 1e-8)
        exceed_bonus = 0.3 * max(0.0, min(1.0, exceed_raw))
    else:
        # Lower is better (mse)
        # baseline is worst (high), target is best (low)
        perf_raw = (baseline - metric) / (baseline - target + 1e-8)
        perf = max(0.0, min(1.0, perf_raw))

        exceed_raw = (target - metric) / (abs(target) + 1e-8)
        exceed_bonus = 0.3 * max(0.0, min(1.0, exceed_raw))

    # Efficiency bonus
    param_eff = 1.0 - (result.param_count / task.max_params) if result.param_count else 0.0
    latency_eff = 1.0 - (result.latency_ms / task.max_inference_ms) if result.latency_ms else 0.0
    param_eff = max(0.0, min(1.0, param_eff))
    latency_eff = max(0.0, min(1.0, latency_eff))
    efficiency = 0.15 * param_eff + 0.15 * latency_eff

    return perf + exceed_bonus + efficiency
