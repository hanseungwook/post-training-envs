#!/usr/bin/env python3
"""Run each task's reference solution through the full pipeline and report results."""

import argparse
import json
import sys
import time

import torch

from nnopt.task import get_task, list_tasks
from nnopt.evaluate import evaluate_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", help="Specific task IDs (default: all)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tier", type=int, help="Only run tasks of this tier")
    args = parser.parse_args()

    task_ids = args.tasks or list_tasks()

    results = []
    passed = 0
    failed = 0

    for task_id in task_ids:
        task = get_task(task_id)

        if args.tier and task.tier != args.tier:
            continue

        if not task.reference_code.strip():
            print(f"  SKIP  {task_id} (no reference code)")
            continue

        print(f"  RUN   {task_id} (tier {task.tier}, {task.metric_name})...", end=" ", flush=True)
        start = time.time()

        try:
            result = evaluate_code(task_id, task.reference_code, device=args.device)
            elapsed = time.time() - start

            if result["success"] and result["reward"] > 0:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            print(
                f"{status}  "
                f"{task.metric_name}={result['metric_value']}  "
                f"params={result['param_count']}  "
                f"latency={result['latency_ms']}ms  "
                f"reward={result['reward']:.3f}  "
                f"({elapsed:.1f}s)"
            )

            if result["error"]:
                print(f"         Error: {result['error'][:200]}")

            results.append(result)

        except Exception as e:
            failed += 1
            print(f"ERROR  {e}")

    print(f"\n{'='*60}")
    print(f"Passed: {passed}/{passed+failed}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
