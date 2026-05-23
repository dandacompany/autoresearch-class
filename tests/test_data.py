from baseline.data import get_loaders


def test_get_loaders_shapes_and_classes(tmp_path):
    train_loader, val_loader = get_loaders(data_root=str(tmp_path), batch_size=4, download=True)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (4, 3, 32, 32)
    assert yb.shape == (4,)
    assert yb.min() >= 0 and yb.max() <= 9


def test_get_loaders_no_augmentation():
    """Baseline must have no data augmentation — autoresearch opportunity."""
    from baseline.data import _train_transform
    from torchvision import transforms

    ops = _train_transform().transforms
    forbidden = (transforms.RandomCrop, transforms.RandomHorizontalFlip, transforms.ColorJitter)
    assert not any(isinstance(op, forbidden) for op in ops)
