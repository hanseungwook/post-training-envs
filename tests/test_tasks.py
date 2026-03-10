"""Tests for task loading and prompt generation."""

import pytest
from nnopt.task import get_task, list_tasks
from nnopt.prompt import build_prompt
from nnopt.utils import extract_code, validate_imports


class TestTaskRegistry:
    def test_list_tasks_not_empty(self):
        tasks = list_tasks()
        assert len(tasks) >= 18

    def test_all_expected_tasks(self):
        tasks = list_tasks()
        expected = [
            "mnist-10k", "mnist-2k",
            "fashion-mnist-20k", "fashion-mnist-5k",
            "kmnist-15k",
            "cifar10-100k", "cifar10-30k",
            "cifar100-200k", "cifar100-100k",
            "svhn-50k",
            "sinusoidal-1k",
            "california-10k", "california-3k",
            "mnist-ae-20k", "mnist-ae-5k",
            "oxford-pet-150k",
            "voc-seg-300k",
            "imdb-50k",
        ]
        for t in expected:
            assert t in tasks, f"Missing task: {t}"

    def test_get_task(self):
        task = get_task("mnist-10k")
        assert task.task_id == "mnist-10k"
        assert task.max_params == 10_000
        assert task.metric_name == "accuracy"
        assert task.input_shape == (1, 28, 28)

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            get_task("nonexistent-task")

    def test_all_tasks_have_reference_code(self):
        for task_id in list_tasks():
            task = get_task(task_id)
            assert task.reference_code.strip(), f"{task_id} has no reference code"

    def test_reference_code_has_solution(self):
        for task_id in list_tasks():
            task = get_task(task_id)
            assert "def solution(" in task.reference_code, f"{task_id} ref code missing solution()"

    def test_reference_code_passes_import_check(self):
        for task_id in list_tasks():
            task = get_task(task_id)
            violations = validate_imports(task.reference_code)
            assert not violations, f"{task_id} ref code has import violations: {violations}"


class TestPromptGeneration:
    def test_prompt_contains_constraints(self):
        task = get_task("mnist-10k")
        prompt = build_prompt(task)
        assert "10,000" in prompt
        assert "accuracy" in prompt
        assert "97" in prompt or "0.97" in prompt
        assert "solution" in prompt

    def test_prompt_contains_input_shape(self):
        task = get_task("cifar10-100k")
        prompt = build_prompt(task)
        assert "3, 32, 32" in prompt

    def test_prompt_lower_is_better(self):
        task = get_task("sinusoidal-1k")
        prompt = build_prompt(task)
        assert "<=" in prompt


class TestCodeExtraction:
    def test_extract_from_markdown(self):
        text = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        code = extract_code(text)
        assert code.strip() == "print('hello')"

    def test_extract_plain_code(self):
        text = "import torch\nprint('hello')"
        code = extract_code(text)
        assert code == text

    def test_extract_multiple_blocks(self):
        text = "```python\nblock1\n```\ntext\n```python\nblock2\n```"
        code = extract_code(text)
        assert "block1" in code
        assert "block2" in code
