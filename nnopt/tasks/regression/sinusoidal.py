"""Sinusoidal regression task."""

import torch
import numpy as np

from nnopt.task import TaskSpec, register_task


def _get_sin_train():
    def fn():
        np.random.seed(42)
        x = np.random.uniform(-2 * np.pi, 2 * np.pi, 2000).astype(np.float32)
        y = np.sin(x).astype(np.float32)
        return torch.tensor(x).unsqueeze(1), torch.tensor(y)
    return fn


def _get_sin_test():
    def fn():
        np.random.seed(123)
        x = np.random.uniform(-2 * np.pi, 2 * np.pi, 500).astype(np.float32)
        y = np.sin(x).astype(np.float32)
        return torch.tensor(x).unsqueeze(1), torch.tensor(y)
    return fn


register_task(TaskSpec(
    task_id="sinusoidal-1k",
    task_type="regression",
    dataset_name="Synthetic sin(x)",
    max_params=1_000,
    target_metric=0.001,
    metric_name="mse",
    higher_is_better=False,
    max_inference_ms=0.5,
    max_train_time_s=30.0,
    train_subset_size=2_000,
    eval_subset_size=500,
    input_shape=(1,),
    num_classes=None,
    output_description="scalar prediction (N,)",
    get_train_data=_get_sin_train(),
    get_test_data=_get_sin_test(),
    baseline_metric=0.5,  # random guessing MSE ~ 0.5
    tier=1,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, y_train = train_data
    X_train, y_train = X_train.to(device), y_train.to(device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    for epoch in range(100):
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()

    return model
""",
))
