"""CIFAR-10 classification tasks."""

import torch
import torchvision
import torchvision.transforms as T

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"


def _get_cifar10_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True)
        X = torch.tensor(ds.data[:subset_size]).permute(0, 3, 1, 2).float() / 255.0
        y = torch.tensor(ds.targets[:subset_size])
        return X, y
    return fn


def _get_cifar10_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.CIFAR10(_DATA_ROOT, train=False, download=True)
        X = torch.tensor(ds.data[:subset_size]).permute(0, 3, 1, 2).float() / 255.0
        y = torch.tensor(ds.targets[:subset_size])
        return X, y
    return fn


register_task(TaskSpec(
    task_id="cifar10-100k",
    task_type="classification",
    dataset_name="CIFAR-10",
    max_params=100_000,
    target_metric=0.85,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=3.0,
    max_train_time_s=180.0,
    train_subset_size=40_000,
    eval_subset_size=5_000,
    input_shape=(3, 32, 32),
    num_classes=10,
    output_description="class labels (N,) in [0, 9]",
    get_train_data=_get_cifar10_train(40_000),
    get_test_data=_get_cifar10_test(5_000),
    baseline_metric=0.10,
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
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
        scheduler.step()

    return model
""",
))

register_task(TaskSpec(
    task_id="cifar10-30k",
    task_type="classification",
    dataset_name="CIFAR-10",
    max_params=30_000,
    target_metric=0.82,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=1.5,
    max_train_time_s=180.0,
    train_subset_size=40_000,
    eval_subset_size=5_000,
    input_shape=(3, 32, 32),
    num_classes=10,
    output_description="class labels (N,) in [0, 9]",
    get_train_data=_get_cifar10_train(40_000),
    get_test_data=_get_cifar10_test(5_000),
    baseline_metric=0.10,
    tier=3,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, y_train = train_data
    X_train, y_train = X_train.to(device), y_train.to(device)

    class DepthwiseSep(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
            self.pw = nn.Conv2d(in_c, out_c, 1)
            self.bn = nn.BatchNorm2d(out_c)

        def forward(self, x):
            return torch.relu(self.bn(self.pw(self.dw(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                DepthwiseSep(16, 32),
                nn.MaxPool2d(2),
                DepthwiseSep(32, 64),
                nn.AdaptiveAvgPool2d(2),
            )
            self.fc = nn.Linear(64 * 2 * 2, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(40):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
        scheduler.step()

    return model
""",
))
