"""CIFAR-10 loader — intentionally no augmentation (autoresearch opportunity)."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _train_transform() -> transforms.Compose:
    # NO augmentation on purpose.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])


def _val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])


def get_loaders(
    data_root: str = "./data",
    batch_size: int = 16,
    download: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    Path(data_root).mkdir(parents=True, exist_ok=True)
    train = datasets.CIFAR10(data_root, train=True, download=download, transform=_train_transform())
    val = datasets.CIFAR10(data_root, train=False, download=download, transform=_val_transform())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
