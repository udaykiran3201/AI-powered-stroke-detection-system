"""
Tests — Preprocessing unit tests
"""

import numpy as np
import torch
import pytest

from app.services.preprocessing import (
    apply_brain_window,
    preprocess_for_classification,
    preprocess_for_segmentation,
    create_overlay,
)


def test_brain_window_range():
    """Windowed output should be in [0, 255]."""
    raw = np.random.uniform(-100, 200, (512, 512)).astype(np.float32)
    windowed = apply_brain_window(raw)
    assert windowed.min() >= 0
    assert windowed.max() <= 255
    assert windowed.dtype == np.uint8


def test_classification_tensor_shape():
    raw = np.random.uniform(0, 100, (512, 512)).astype(np.float32)
    tensor = preprocess_for_classification(raw)
    assert tensor.shape == (1, 3, 256, 256)


def test_segmentation_tensor_shape():
    raw = np.random.uniform(0, 100, (512, 512)).astype(np.float32)
    tensor = preprocess_for_segmentation(raw)
    assert tensor.shape == (1, 1, 256, 256)


def test_create_overlay_shape():
    original = np.random.uniform(0, 100, (512, 512)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    overlay = create_overlay(original, mask)
    assert overlay.shape == (256, 256, 3)
