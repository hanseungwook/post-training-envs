"""MNIST autoencoder tasks."""

import torch
import torchvision

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"


def _get_mnist_ae_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.MNIST(_DATA_ROOT, train=True, download=True)
        X = ds.data[:subset_size].float().unsqueeze(1) / 255.0
        # For autoencoding, y is the same as X (target = input)
        return X, X.clone()
    return fn


def _get_mnist_ae_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.MNIST(_DATA_ROOT, train=False, download=True)
        X = ds.data[:subset_size].float().unsqueeze(1) / 255.0
        return X, X.clone()
    return fn


register_task(TaskSpec(
    task_id="mnist-ae-20k",
    task_type="autoencoding",
    dataset_name="MNIST",
    max_params=20_000,
    target_metric=0.93,
    metric_name="ssim",
    higher_is_better=True,
    max_inference_ms=2.0,
    max_train_time_s=90.0,
    train_subset_size=10_000,
    eval_subset_size=2_000,
    input_shape=(1, 28, 28),
    num_classes=None,
    output_description="reconstructed images (N, 1, 28, 28) in [0, 1]",
    get_train_data=_get_mnist_ae_train(10_000),
    get_test_data=_get_mnist_ae_test(2_000),
    baseline_metric=0.0,
    tier=2,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, _ = train_data
    X_train = X_train.to(device)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(X_train, X_train), batch_size=128, shuffle=True)

    for epoch in range(30):
        for xb, _ in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), xb).backward()
            opt.step()

    return model
""",
))

register_task(TaskSpec(
    task_id="mnist-ae-5k",
    task_type="autoencoding",
    dataset_name="MNIST",
    max_params=5_000,
    target_metric=0.88,
    metric_name="ssim",
    higher_is_better=True,
    max_inference_ms=1.0,
    max_train_time_s=90.0,
    train_subset_size=10_000,
    eval_subset_size=2_000,
    input_shape=(1, 28, 28),
    num_classes=None,
    output_description="reconstructed images (N, 1, 28, 28) in [0, 1]",
    get_train_data=_get_mnist_ae_train(10_000),
    get_test_data=_get_mnist_ae_test(2_000),
    baseline_metric=0.0,
    tier=3,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, _ = train_data
    X_train = X_train.to(device)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(X_train, X_train), batch_size=128, shuffle=True)

    for epoch in range(40):
        for xb, _ in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), xb).backward()
            opt.step()

    return model
""",
))
