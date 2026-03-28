"""
Tests — Model unit tests
"""

import torch
import pytest

from app.models.classifier import StrokeClassifier, NUM_CLASSES
from app.models.segmentation import UNet


def test_classifier_output_shape():
    model = StrokeClassifier(pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {output.shape}"


def test_classifier_sigmoid_range():
    model = StrokeClassifier(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_unet_output_shape():
    model = UNet(in_channels=1)
    model.eval()
    dummy = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, 1, 256, 256), f"Expected (1,1,256,256), got {output.shape}"


def test_unet_odd_dimensions():
    """U-Net should handle non-power-of-2 dimensions gracefully."""
    model = UNet(in_channels=1)
    model.eval()
    dummy = torch.randn(1, 1, 253, 251)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape[0] == 1
    assert output.shape[1] == 1
