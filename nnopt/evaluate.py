"""Standalone entry point: code in -> reward out."""

from __future__ import annotations

import argparse
import json
import sys

from nnopt.task import get_task, list_tasks
from nnopt.sandbox import run_in_sandbox
from nnopt.reward import compute_reward
from nnopt.utils import extract_code


def evaluate_code(
    task_id: str,
    code: str,
    device: str = "cuda:0",
) -> dict:
    """
    Main entry point. Evaluate LLM-generated code on a task.

    Returns:
        {
            "task_id": str,
            "reward": float,
            "success": bool,
            "metric_name": str,
            "metric_value": float | None,
            "param_count": int | None,
            "latency_ms": float | None,
            "constraint_satisfied": bool,
            "error": str | None,
        }
    """
    task = get_task(task_id)

    # Extract code from markdown if needed
    code = extract_code(code)

    # Run in sandbox
    result = run_in_sandbox(code, task, device=device)

    # Compute reward
    reward = compute_reward(result, task)

    # Check constraints
    constraint_satisfied = (
        result.success
        and (result.param_count is not None and result.param_count <= task.max_params)
        and (result.latency_ms is not None and result.latency_ms <= task.max_inference_ms)
    )

    return {
        "task_id": task_id,
        "reward": round(reward, 4),
        "success": result.success,
        "metric_name": task.metric_name,
        "metric_value": round(result.metric_value, 6) if result.metric_value is not None else None,
        "param_count": result.param_count,
        "latency_ms": round(result.latency_ms, 3) if result.latency_ms is not None else None,
        "constraint_satisfied": constraint_satisfied,
        "error": result.error,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate neural network code on a task")
    parser.add_argument("--task", required=True, help="Task ID (e.g., mnist-10k)")
    parser.add_argument("--code-file", required=True, help="Path to Python code file")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks")

    args = parser.parse_args()

    if args.list_tasks:
        for t in list_tasks():
            print(t)
        return

    with open(args.code_file, "r") as f:
        code = f.read()

    result = evaluate_code(args.task, code, device=args.device)
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
