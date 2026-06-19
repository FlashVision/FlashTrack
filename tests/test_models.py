"""Tests for FlashTrack models."""

import torch
import pytest


def test_flashtracker_forward_m():
    """Test forward pass for FlashTrack-m."""
    from flashtrack.models.tracker import FlashTracker

    model = FlashTracker(
        backbone_size="1.0x", reid_dim=128, encoder_channels=256,
        num_ids=100, pretrained=False, input_size=(128, 64),
    )
    model.eval()
    x = torch.randn(2, 3, 128, 64)
    with torch.no_grad():
        out = model(x)
    assert "embedding" in out
    assert out["embedding"].shape == (2, 128)
    assert "logits" in out
    assert out["logits"].shape == (2, 100)


def test_flashtracker_forward_m05x():
    """Test forward pass for FlashTrack-m-0.5x."""
    from flashtrack.models.tracker import FlashTracker

    model = FlashTracker(
        backbone_size="0.5x", reid_dim=128, encoder_channels=128,
        num_ids=0, pretrained=False, input_size=(128, 64),
    )
    model.eval()
    x = torch.randn(1, 3, 128, 64)
    with torch.no_grad():
        out = model(x)
    assert out["embedding"].shape == (1, 128)
    assert "logits" not in out


def test_flashtracker_forward_m15x():
    """Test forward pass for FlashTrack-m-1.5x."""
    from flashtrack.models.tracker import FlashTracker

    model = FlashTracker(
        backbone_size="1.5x", reid_dim=256, encoder_channels=384,
        num_ids=50, pretrained=False, input_size=(128, 64),
    )
    model.eval()
    x = torch.randn(1, 3, 128, 64)
    with torch.no_grad():
        out = model(x)
    assert out["embedding"].shape == (1, 256)


def test_model_size_ordering():
    """Verify that larger models have more parameters."""
    from flashtrack.models.tracker import FlashTracker

    models = {}
    for name, bs, rd, ec in [("m-0.5x", "0.5x", 128, 128), ("m", "1.0x", 128, 256), ("m-1.5x", "1.5x", 256, 384)]:
        m = FlashTracker(backbone_size=bs, reid_dim=rd, encoder_channels=ec, num_ids=0, pretrained=False)
        models[name] = sum(p.numel() for p in m.parameters())

    assert models["m-0.5x"] < models["m"] < models["m-1.5x"]


def test_extract_features():
    """Test ReID feature extraction."""
    from flashtrack.models.tracker import FlashTracker

    model = FlashTracker(
        backbone_size="1.0x", reid_dim=128, encoder_channels=256,
        num_ids=0, pretrained=False,
    )
    x = torch.randn(4, 3, 128, 64)
    feats = model.extract_features(x)
    assert feats.shape == (4, 128)
    # Check L2 normalization
    norms = feats.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=0.01)


def test_lora_application():
    """Test LoRA can be applied to FlashTracker."""
    from flashtrack.models.tracker import FlashTracker
    from flashtrack.models.lora import apply_lora

    model = FlashTracker(
        backbone_size="1.0x", reid_dim=128, encoder_channels=256,
        num_ids=0, pretrained=False,
    )
    total_before = sum(p.numel() for p in model.parameters())
    model = apply_lora(model, rank=4, target_modules=["backbone", "encoder"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_after = sum(p.numel() for p in model.parameters())

    assert total_after > total_before
    assert trainable < total_after

    # Forward pass still works
    model.eval()
    x = torch.randn(1, 3, 128, 64)
    with torch.no_grad():
        out = model(x)
    assert out["embedding"].shape == (1, 128)


def test_build_model():
    """Test building model from config."""
    from flashtrack.cfg import get_config
    from flashtrack.models import build_model

    cfg = get_config(model_size="m", input_size=(128, 64), num_ids=10)
    model = build_model(cfg)
    model.eval()
    x = torch.randn(1, 3, 128, 64)
    with torch.no_grad():
        out = model(x)
    assert out["embedding"].shape == (1, 128)
