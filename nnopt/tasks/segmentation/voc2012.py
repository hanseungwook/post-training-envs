"""VOC2012 semantic segmentation task."""

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"
_IMG_SIZE = 64
_NUM_CLASSES = 21


def _get_voc_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.VOCSegmentation(
            _DATA_ROOT, year="2012", image_set="train", download=True,
        )
        images, masks = [], []
        for i in range(min(subset_size, len(ds))):
            img, mask = ds[i]
            img = TF.resize(img, [_IMG_SIZE, _IMG_SIZE])
            img = TF.to_tensor(img)
            mask = TF.resize(mask, [_IMG_SIZE, _IMG_SIZE], interpolation=T.InterpolationMode.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)
            # VOC uses 255 for border/ignore — map to 0
            mask[mask == 255] = 0
            images.append(img)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks)
    return fn


def _get_voc_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.VOCSegmentation(
            _DATA_ROOT, year="2012", image_set="val", download=True,
        )
        images, masks = [], []
        for i in range(min(subset_size, len(ds))):
            img, mask = ds[i]
            img = TF.resize(img, [_IMG_SIZE, _IMG_SIZE])
            img = TF.to_tensor(img)
            mask = TF.resize(mask, [_IMG_SIZE, _IMG_SIZE], interpolation=T.InterpolationMode.NEAREST)
            mask = torch.tensor(np.array(mask), dtype=torch.long)
            mask[mask == 255] = 0
            images.append(img)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks)
    return fn


register_task(TaskSpec(
    task_id="voc-seg-300k",
    task_type="segmentation",
    dataset_name="VOC2012",
    max_params=300_000,
    target_metric=0.30,
    metric_name="miou",
    higher_is_better=True,
    max_inference_ms=15.0,
    max_train_time_s=300.0,
    train_subset_size=1_500,
    eval_subset_size=500,
    input_shape=(3, _IMG_SIZE, _IMG_SIZE),
    num_classes=_NUM_CLASSES,
    output_description=f"pixel class logits (N, {_NUM_CLASSES}, {_IMG_SIZE}, {_IMG_SIZE})",
    get_train_data=_get_voc_train(1_500),
    get_test_data=_get_voc_test(500),
    baseline_metric=1.0 / _NUM_CLASSES,  # ~0.048
    tier=3,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, y_train = train_data
    X_train, y_train = X_train.to(device), y_train.to(device)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec2 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.dec1 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.out = nn.Conv2d(32, 21, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = torch.cat([self.up(e3), e2], dim=1)
            d2 = self.dec2(d2)
            d1 = torch.cat([self.up(d2), e1], dim=1)
            d1 = self.dec1(d1)
            return self.out(d1)

    model = UNet().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()

    return model
""",
))
