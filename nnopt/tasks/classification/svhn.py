"""SVHN classification task."""

import torch
import torchvision

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"


def _get_svhn_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.SVHN(_DATA_ROOT, split="train", download=True)
        X = torch.tensor(ds.data[:subset_size]).float() / 255.0  # already (N, 3, 32, 32)
        y = torch.tensor(ds.labels[:subset_size])
        return X, y
    return fn


def _get_svhn_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.SVHN(_DATA_ROOT, split="test", download=True)
        X = torch.tensor(ds.data[:subset_size]).float() / 255.0
        y = torch.tensor(ds.labels[:subset_size])
        return X, y
    return fn


register_task(TaskSpec(
    task_id="svhn-50k",
    task_type="classification",
    dataset_name="SVHN",
    max_params=50_000,
    target_metric=0.90,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=2.0,
    max_train_time_s=180.0,
    train_subset_size=40_000,
    eval_subset_size=5_000,
    input_shape=(3, 32, 32),
    num_classes=10,
    output_description="class labels (N,) in [0, 9]",
    get_train_data=_get_svhn_train(40_000),
    get_test_data=_get_svhn_test(5_000),
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
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(2),
            )
            self.fc = nn.Linear(64 * 2 * 2, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

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
