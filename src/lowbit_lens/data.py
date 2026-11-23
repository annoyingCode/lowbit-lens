"""BloodMNIST loading.

BloodMNIST (MedMNIST v2, Yang et al. 2023) has 17,092 RGB microscope images of
individual blood cells in 8 classes, with an official train/val/test split of
11,959 / 1,712 / 3,421. MedMNIST+ provides it at 28, 64, 128 and 224 px.

Loading strategy: each split is kept in RAM as one uint8 tensor (the 224 px train
split is about 1.8 GB) and each batch is copied to the GPU before augmentation and
normalisation. This avoids DataLoader worker processes, which on Windows each get
their own pickled copy of the dataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Pretrained torchvision weights expect inputs normalised with ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Split:
    images: torch.Tensor  # (N, 3, H, W) uint8
    labels: torch.Tensor  # (N,) int64

    def __len__(self) -> int:
        return self.labels.numel()


def class_names(dataset: str = "bloodmnist") -> list[str]:
    from medmnist import INFO

    labels = INFO[dataset]["label"]
    return [labels[str(i)] for i in range(len(labels))]


def load_split(dataset: str, split: str, size: int, data_dir: Path) -> Split:
    """Download (first time only) and load one split as tensors."""
    import medmnist
    from medmnist import INFO

    data_dir.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, INFO[dataset]["python_class"])
    ds = cls(split=split, download=True, size=size, root=str(data_dir))
    images = torch.from_numpy(ds.imgs).permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    labels = torch.from_numpy(ds.labels.reshape(-1)).long()
    if images.shape[1] != 3:
        raise ValueError(f"expected RGB images, got shape {tuple(images.shape)}")
    return Split(images, labels)


def load_labels(dataset: str, split: str, size: int, data_dir: Path) -> np.ndarray:
    """Labels of one split without loading the images (reads the downloaded file)."""
    name = f"{dataset}.npz" if size == 28 else f"{dataset}_{size}.npz"
    with np.load(data_dir / name) as z:
        return z[f"{split}_labels"].reshape(-1).astype(np.int64)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Float images in [0, 1] -> ImageNet-normalised model input."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def random_dihedral(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Apply one of the 8 rotations/reflections of the square to each image.

    A blood cell under a microscope has no canonical orientation, so these
    transforms create new valid training images without changing the label.
    The random draws come from the CPU generator `g` so runs are repeatable.
    """
    n = x.shape[0]
    flips = torch.rand(n, 3, generator=g) < 0.5  # transpose, flip H, flip W
    flips = flips.to(x.device)

    def where(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.where(mask.view(n, 1, 1, 1), a, b)

    x = where(flips[:, 0], x.transpose(-1, -2), x)  # needs H == W
    x = where(flips[:, 1], x.flip(-2), x)
    x = where(flips[:, 2], x.flip(-1), x)
    return x


def iterate_batches(
    split: Split,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool = False,
    augment: bool = False,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (normalised float images, labels) batches already on `device`."""
    if (shuffle or augment) and generator is None:
        raise ValueError("shuffle/augment need a seeded generator for reproducibility")
    n = len(split)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        x = split.images[idx].to(device, non_blocking=True).float().div_(255.0)
        y = split.labels[idx].to(device, non_blocking=True)
        if augment:
            x = random_dihedral(x, generator)
        yield normalize(x), y
