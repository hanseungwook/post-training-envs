"""Utilities: code extraction from markdown, import validation via AST."""

from __future__ import annotations

import ast
import re

# Modules the LLM code is allowed to import
ALLOWED_MODULES = frozenset({
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.optim.lr_scheduler",
    "torch.utils.data",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "math",
    "numpy",
    "random",
    "collections",
    "itertools",
    "functools",
})

# Top-level packages that are allowed (for prefix matching)
ALLOWED_TOP_LEVEL = frozenset({
    "torch", "torchvision", "math", "numpy", "random",
    "collections", "itertools", "functools",
})


def extract_code(text: str) -> str:
    """Extract Python code from markdown fenced blocks, or return as-is."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    # If no code block found, try generic blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    # Assume the whole text is code
    return text


def validate_imports(code: str) -> list[str]:
    """
    Parse code with AST and check all imports against the whitelist.
    Returns a list of violation strings (empty = all OK).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed(alias.name):
                    violations.append(f"Banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not _is_allowed(module):
                violations.append(f"Banned import: from {module}")

    return violations


def _is_allowed(module_name: str) -> bool:
    """Check if a module name is in the allowed set."""
    if module_name in ALLOWED_MODULES:
        return True
    # Check if it's a submodule of an allowed top-level package
    parts = module_name.split(".")
    if parts[0] in ALLOWED_TOP_LEVEL:
        # Allow torch.* and torchvision.transforms*
        if parts[0] == "torchvision":
            # Only allow torchvision.transforms*
            return len(parts) >= 2 and parts[1] == "transforms"
        if parts[0] == "torch":
            return True
        # For math, numpy, etc. — only exact match (no submodules needed)
        return len(parts) == 1
    return False


def count_parameters(model) -> int:
    """Count total trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
