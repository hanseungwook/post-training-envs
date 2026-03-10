"""Tests for reward computation."""

import pytest
from nnopt.reward import compute_reward
from nnopt.sandbox import SandboxResult
from nnopt.task import TaskSpec


def _make_task(**overrides):
    defaults = dict(
        task_id="test",
        task_type="classification",
        dataset_name="Test",
        max_params=10_000,
        target_metric=0.90,
        metric_name="accuracy",
        higher_is_better=True,
        max_inference_ms=5.0,
        max_train_time_s=10.0,
        train_subset_size=100,
        eval_subset_size=50,
        input_shape=(1, 8, 8),
        num_classes=2,
        output_description="",
        get_train_data=lambda: None,
        get_test_data=lambda: None,
        baseline_metric=0.10,
        tier=1,
        reference_code="",
    )
    defaults.update(overrides)
    return TaskSpec(**defaults)


class TestFailureRewards:
    def test_parse_error(self):
        result = SandboxResult(success=False, error_type="parse")
        reward = compute_reward(result, _make_task())
        assert reward == -1.0

    def test_import_error(self):
        result = SandboxResult(success=False, error_type="import")
        reward = compute_reward(result, _make_task())
        assert reward == -1.0

    def test_runtime_error(self):
        result = SandboxResult(success=False, error_type="runtime")
        reward = compute_reward(result, _make_task())
        assert reward == -0.5

    def test_timeout(self):
        result = SandboxResult(success=False, error_type="timeout")
        reward = compute_reward(result, _make_task())
        assert reward == -0.5


class TestConstraintViolation:
    def test_param_count_exceeded(self):
        result = SandboxResult(
            success=True, metric_value=0.95, param_count=15_000, latency_ms=1.0,
        )
        task = _make_task(max_params=10_000)
        assert compute_reward(result, task) == -0.2

    def test_latency_exceeded(self):
        result = SandboxResult(
            success=True, metric_value=0.95, param_count=5_000, latency_ms=10.0,
        )
        task = _make_task(max_inference_ms=5.0)
        assert compute_reward(result, task) == -0.1


class TestRewardOrdering:
    def test_ordering(self):
        """crash < constraint < bad perf < good perf < efficient."""
        task = _make_task(
            baseline_metric=0.10, target_metric=0.90,
            max_params=10_000, max_inference_ms=5.0,
        )

        crash = compute_reward(
            SandboxResult(success=False, error_type="runtime"), task
        )
        constraint = compute_reward(
            SandboxResult(success=True, metric_value=0.95, param_count=20_000, latency_ms=1.0), task
        )
        bad_perf = compute_reward(
            SandboxResult(success=True, metric_value=0.30, param_count=5_000, latency_ms=2.0), task
        )
        good_perf = compute_reward(
            SandboxResult(success=True, metric_value=0.90, param_count=5_000, latency_ms=2.0), task
        )
        efficient = compute_reward(
            SandboxResult(success=True, metric_value=0.95, param_count=1_000, latency_ms=0.5), task
        )

        assert crash < constraint < bad_perf < good_perf < efficient

    def test_perfect_score_range(self):
        """Max reward should not exceed 1.6."""
        task = _make_task(
            baseline_metric=0.10, target_metric=0.90,
            max_params=10_000, max_inference_ms=5.0,
        )
        result = SandboxResult(
            success=True, metric_value=1.0, param_count=1, latency_ms=0.01,
        )
        reward = compute_reward(result, task)
        assert 0 < reward <= 1.6


class TestLowerIsBetterMetric:
    def test_mse_reward(self):
        task = _make_task(
            metric_name="mse", higher_is_better=False,
            baseline_metric=0.5, target_metric=0.001,
        )
        # Good MSE
        good = compute_reward(
            SandboxResult(success=True, metric_value=0.001, param_count=500, latency_ms=0.2), task
        )
        # Bad MSE
        bad = compute_reward(
            SandboxResult(success=True, metric_value=0.4, param_count=500, latency_ms=0.2), task
        )
        assert good > bad > 0
