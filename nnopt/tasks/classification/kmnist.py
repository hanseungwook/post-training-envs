"""KMNIST classification task."""

import torch
import torchvision

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"


def _get_kmnist_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.KMNIST(_DATA_ROOT, train=True, download=True)
        X = ds.data[:subset_size].float().unsqueeze(1) / 255.0
        y = ds.targets[:subset_size]
        return X, y
    return fn


def _get_kmnist_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.KMNIST(_DATA_ROOT, train=False, download=True)
        X = ds.data[:subset_size].float().unsqueeze(1) / 255.0
        y = ds.targets[:subset_size]
        return X, y
    return fn


register_task(TaskSpec(
    task_id="kmnist-15k",
    task_type="classification",
    dataset_name="KMNIST",
    max_params=15_000,
    target_metric=0.92,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=2.0,
    max_train_time_s=90.0,
    train_subset_size=10_000,
    eval_subset_size=2_000,
    input_shape=(1, 28, 28),
    num_classes=10,
    output_description="class labels (N,) in [0, 9]",
    get_train_data=_get_kmnist_train(10_000),
    get_test_data=_get_kmnist_test(2_000),
    baseline_metric=0.10,
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
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()

    return model
""",
))
