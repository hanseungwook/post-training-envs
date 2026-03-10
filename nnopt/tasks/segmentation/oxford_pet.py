"""Oxford-IIIT Pets binary segmentation task."""

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from nnopt.task import TaskSpec, register_task

_DATA_ROOT = "/tmp/nnopt_data"
_IMG_SIZE = 64  # Resize to 64x64 for tractability


def _get_pet_train(subset_size: int):
    def fn():
        ds = torchvision.datasets.OxfordIIITPet(
            _DATA_ROOT, split="trainval", target_types="segmentation", download=True,
        )
        images, masks = [], []
        for i in range(min(subset_size, len(ds))):
            img, mask = ds[i]
            img = TF.resize(img, [_IMG_SIZE, _IMG_SIZE])
            img = TF.to_tensor(img)
            mask = TF.resize(mask, [_IMG_SIZE, _IMG_SIZE], interpolation=T.InterpolationMode.NEAREST)
            mask = torch.tensor(TF.pil_to_tensor(mask).squeeze(0), dtype=torch.long)
            # Oxford Pets: 1=foreground, 2=background, 3=border -> binary: 0=bg, 1=fg
            mask = (mask == 1).long()
            images.append(img)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks)
    return fn


def _get_pet_test(subset_size: int):
    def fn():
        ds = torchvision.datasets.OxfordIIITPet(
            _DATA_ROOT, split="test", target_types="segmentation", download=True,
        )
        images, masks = [], []
        for i in range(min(subset_size, len(ds))):
            img, mask = ds[i]
            img = TF.resize(img, [_IMG_SIZE, _IMG_SIZE])
            img = TF.to_tensor(img)
            mask = TF.resize(mask, [_IMG_SIZE, _IMG_SIZE], interpolation=T.InterpolationMode.NEAREST)
            mask = torch.tensor(TF.pil_to_tensor(mask).squeeze(0), dtype=torch.long)
            mask = (mask == 1).long()
            images.append(img)
            masks.append(mask)
        return torch.stack(images), torch.stack(masks)
    return fn


register_task(TaskSpec(
    task_id="oxford-pet-150k",
    task_type="segmentation",
    dataset_name="Oxford-IIIT Pets",
    max_params=150_000,
    target_metric=0.75,
    metric_name="pixel_accuracy",
    higher_is_better=True,
    max_inference_ms=10.0,
    max_train_time_s=180.0,
    train_subset_size=2_000,
    eval_subset_size=500,
    input_shape=(3, _IMG_SIZE, _IMG_SIZE),
    num_classes=2,
    output_description="pixel class logits (N, 2, 64, 64)",
    get_train_data=_get_pet_train(2_000),
    get_test_data=_get_pet_test(500),
    baseline_metric=0.50,
    tier=2,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, y_train = train_data
    X_train, y_train = X_train.to(device), y_train.to(device)

    class UNetSmall(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.dec1 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.out = nn.Conv2d(32, 2, 1)

        def forward(self, x):
            e1 = self.enc1(x)          # 64x64
            e2 = self.enc2(self.pool(e1))  # 32x32
            d1 = self.up(e2)           # 64x64
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)
            return self.out(d1)

    model = UNetSmall().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()

    return model
""",
))
