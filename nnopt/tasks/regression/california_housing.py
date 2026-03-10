"""California Housing regression tasks."""

import torch
import numpy as np

from nnopt.task import TaskSpec, register_task


def _get_ca_train(subset_size: int):
    def fn():
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = data.data[:subset_size].astype(np.float32)
        y = data.target[:subset_size].astype(np.float32)
        # Standardize features
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        return torch.tensor(X), torch.tensor(y)
    return fn


def _get_ca_test(subset_size: int, train_size: int):
    def fn():
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        # Use training data stats for standardization
        X_train_raw = data.data[:train_size].astype(np.float32)
        mean = X_train_raw.mean(axis=0)
        std = X_train_raw.std(axis=0) + 1e-8

        X = data.data[train_size:train_size + subset_size].astype(np.float32)
        y = data.target[train_size:train_size + subset_size].astype(np.float32)
        X = (X - mean) / std
        return torch.tensor(X), torch.tensor(y)
    return fn


register_task(TaskSpec(
    task_id="california-10k",
    task_type="regression",
    dataset_name="California Housing",
    max_params=10_000,
    target_metric=0.55,
    metric_name="r2",
    higher_is_better=True,
    max_inference_ms=0.5,
    max_train_time_s=30.0,
    train_subset_size=15_000,
    eval_subset_size=3_000,
    input_shape=(8,),
    num_classes=None,
    output_description="scalar prediction (N,) — median house value in $100k",
    get_train_data=_get_ca_train(15_000),
    get_test_data=_get_ca_test(3_000, 15_000),
    baseline_metric=0.0,  # R2 of predicting mean
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
                nn.Linear(8, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.005)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(50):
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()

    return model
""",
))

register_task(TaskSpec(
    task_id="california-3k",
    task_type="regression",
    dataset_name="California Housing",
    max_params=3_000,
    target_metric=0.55,
    metric_name="r2",
    higher_is_better=True,
    max_inference_ms=0.5,
    max_train_time_s=30.0,
    train_subset_size=15_000,
    eval_subset_size=3_000,
    input_shape=(8,),
    num_classes=None,
    output_description="scalar prediction (N,) — median house value in $100k",
    get_train_data=_get_ca_train(15_000),
    get_test_data=_get_ca_test(3_000, 15_000),
    baseline_metric=0.0,
    tier=2,
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
                nn.Linear(8, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.005)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(80):
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()

    return model
""",
))
