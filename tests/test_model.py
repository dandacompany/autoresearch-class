import torch

from baseline.model import WeakCNN


def test_weak_cnn_forward_shape():
    model = WeakCNN()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), f"expected (2, 10), got {tuple(y.shape)}"


def test_weak_cnn_has_no_batchnorm():
    """Baseline is intentionally weak — leave BatchNorm as an autoresearch opportunity."""
    model = WeakCNN()
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bn_layers == [], "baseline must not include BatchNorm"
