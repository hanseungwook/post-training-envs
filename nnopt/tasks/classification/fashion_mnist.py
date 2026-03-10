"""FashionMNIST classification tasks."""

import torch
import torchvision
import torchvision.transforms as T

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"


def _get_fmnist_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.FashionMNIST(_DATA_ROOT, train=True, download=True, transform=T.ToTensor())
        X = ds.data[:subset_size].float().unsqueeze(1) / 255.0
        y = ds.targets[:subset_size]
        return X, y
    return fn


def _get_fmnist_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.FashionMNIST(_DATA_ROOT, train=False, download=True, transform=T.ToTensor())
        X = ds.data[:subset_size].float().unsqueeze(1) / 255.0
        y = ds.targets[:subset_size]
        return X, y
    return fn


register_task(TaskSpec(
    task_id="fashion-mnist-20k",
    task_type="classification",
    dataset_name="FashionMNIST",
    max_params=20_000,
    target_metric=0.87,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=2.0,
    max_train_time_s=90.0,
    train_subset_size=15_000,
    eval_subset_size=3_000,
    input_shape=(1, 28, 28),
    num_classes=10,
    output_description="class labels (N,) in [0, 9]",
    get_train_data=_get_fmnist_train(15_000),
    get_test_data=_get_fmnist_test(3_000),
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
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002)
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

register_task(TaskSpec(
    task_id="fashion-mnist-5k",
    task_type="classification",
    dataset_name="FashionMNIST",
    max_params=5_000,
    target_metric=0.86,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=1.0,
    max_train_time_s=90.0,
    train_subset_size=15_000,
    eval_subset_size=3_000,
    input_shape=(1, 28, 28),
    num_classes=10,
    output_description="class labels (N,) in [0, 9]",
    get_train_data=_get_fmnist_train(15_000),
    get_test_data=_get_fmnist_test(3_000),
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
            self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
            self.fc = nn.Linear(16 * 7 * 7, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()

    return model
""",
))
