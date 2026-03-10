#!/usr/bin/env python3
"""Pre-download all datasets to the cache directory."""

import sys

DATA_ROOT = "/tmp/nnopt_data"


def main():
    print("Downloading datasets to", DATA_ROOT)

    # MNIST-family
    import torchvision
    for ds_cls in [
        torchvision.datasets.MNIST,
        torchvision.datasets.FashionMNIST,
        torchvision.datasets.KMNIST,
    ]:
        name = ds_cls.__name__
        print(f"  {name}...", end=" ", flush=True)
        ds_cls(DATA_ROOT, train=True, download=True)
        ds_cls(DATA_ROOT, train=False, download=True)
        print("OK")

    # CIFAR
    for ds_cls in [
        torchvision.datasets.CIFAR10,
        torchvision.datasets.CIFAR100,
    ]:
        name = ds_cls.__name__
        print(f"  {name}...", end=" ", flush=True)
        ds_cls(DATA_ROOT, train=True, download=True)
        ds_cls(DATA_ROOT, train=False, download=True)
        print("OK")

    # SVHN
    print("  SVHN...", end=" ", flush=True)
    torchvision.datasets.SVHN(DATA_ROOT, split="train", download=True)
    torchvision.datasets.SVHN(DATA_ROOT, split="test", download=True)
    print("OK")

    # Oxford Pets
    print("  OxfordIIITPet...", end=" ", flush=True)
    try:
        torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="trainval", target_types="segmentation", download=True)
        torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="test", target_types="segmentation", download=True)
        print("OK")
    except Exception as e:
        print(f"SKIP ({e})")

    # VOC2012
    print("  VOC2012...", end=" ", flush=True)
    try:
        torchvision.datasets.VOCSegmentation(DATA_ROOT, year="2012", image_set="train", download=True)
        print("OK")
    except Exception as e:
        print(f"SKIP ({e})")

    # California Housing (sklearn)
    print("  California Housing...", end=" ", flush=True)
    from sklearn.datasets import fetch_california_housing
    fetch_california_housing()
    print("OK")

    # IMDB (HuggingFace)
    print("  IMDB...", end=" ", flush=True)
    try:
        from datasets import load_dataset
        load_dataset("imdb", split="train[:10]")
        print("OK")
    except Exception as e:
        print(f"SKIP ({e})")

    print("Done.")


if __name__ == "__main__":
    main()
