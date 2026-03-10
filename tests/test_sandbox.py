"""Tests for sandbox safety and execution."""

import pytest
import torch
from nnopt.sandbox import run_in_sandbox, SandboxResult
from nnopt.task import TaskSpec


def _make_dummy_task(**overrides):
    """Create a minimal task for testing."""
    defaults = dict(
        task_id="test-task",
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
        output_description="class labels (N,) in [0, 1]",
        get_train_data=lambda: (torch.randn(100, 1, 8, 8), torch.randint(0, 2, (100,))),
        get_test_data=lambda: (torch.randn(50, 1, 8, 8), torch.randint(0, 2, (50,))),
        baseline_metric=0.50,
        tier=1,
        reference_code="",
    )
    defaults.update(overrides)
    return TaskSpec(**defaults)


class TestImportValidation:
    def test_banned_import_os(self):
        code = "import os\ndef solution(td,ed,d): pass"
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "import"
        assert "os" in result.error

    def test_banned_import_subprocess(self):
        code = "import subprocess\ndef solution(td,ed,d): pass"
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "import"

    def test_banned_import_socket(self):
        code = "from socket import socket\ndef solution(td,ed,d): pass"
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "import"

    def test_allowed_imports(self):
        code = """
import torch
import torch.nn as nn
import math
import numpy as np
import random

def solution(train_data, test_data, device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 2)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    model = M().to(device)
    return model
"""
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert result.success


class TestTimeout:
    def test_infinite_loop_killed(self):
        code = """
import torch.nn as nn
def solution(train_data, test_data, device):
    while True:
        pass
"""
        task = _make_dummy_task(max_train_time_s=2.0)
        result = run_in_sandbox(code, task, device="cpu", timeout_buffer=1.0)
        assert not result.success
        assert result.error_type == "timeout"


class TestRuntimeErrors:
    def test_missing_solution_function(self):
        code = """
import torch
def train_model(data):
    pass
"""
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "runtime"
        assert "solution" in result.error

    def test_solution_returns_none(self):
        code = """
def solution(train_data, test_data, device):
    return None
"""
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "runtime"

    def test_runtime_exception(self):
        code = """
def solution(train_data, test_data, device):
    raise ValueError("something went wrong")
"""
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "runtime"

    def test_syntax_error(self):
        code = "def solution( broken syntax"
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert not result.success
        assert result.error_type == "import"  # caught in validate_imports


class TestSuccessfulExecution:
    def test_simple_model(self):
        code = """
import torch
import torch.nn as nn

def solution(train_data, test_data, device):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 2)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    return M().to(device)
"""
        task = _make_dummy_task()
        result = run_in_sandbox(code, task, device="cpu")
        assert result.success
        assert result.param_count is not None
        assert result.param_count == 64 * 2 + 2  # weights + bias
        assert result.metric_value is not None
        assert result.latency_ms is not None
