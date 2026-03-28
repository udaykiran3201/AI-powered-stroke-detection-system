"""
Stroke Detection System — Image Preprocessing Pipeline

Handles DICOM / PNG / JPEG CT scans:
  1. Read & normalise pixel data
  2. Apply brain-window (W:80, L:40) for hemorrhage visibility
  3. Resize to model input dimensions
  4. Convert to PyTorch tensor
"""

import io
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
import torch
from torchvision import transforms
from loguru import logger

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    logger.warning("pydicom not installed — DICOM support disabled")


# ── Constants ───────────────────────────────
BRAIN_WINDOW_CENTER = 40
BRAIN_WINDOW_WIDTH = 80
MODEL_INPUT_SIZE = (256, 256)


def apply_brain_window(
    image: np.ndarray,
    window_center: int = BRAIN_WINDOW_CENTER,
    window_width: int = BRAIN_WINDOW_WIDTH,
) -> np.ndarray:
    """Apply CT brain window (default: W80/L40) and rescale to 0-255."""
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    windowed = np.clip(image, min_val, max_val)
    windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return windowed


def read_dicom(file_bytes: bytes) -> np.ndarray:
    """Read a DICOM file and return the pixel array as float32."""
    if not HAS_PYDICOM:
        raise RuntimeError("pydicom is required to read DICOM files")
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale slope / intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    return pixel_array


def read_standard_image(file_bytes: bytes) -> np.ndarray:
    """Read PNG / JPEG bytes into a grayscale numpy array."""
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return np.array(image, dtype=np.float32)


def preprocess_for_classification(
    pixel_array: np.ndarray,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> torch.Tensor:
    """
    Pre-process a 2-D CT slice for the EfficientNet classifier.

    Returns a (1, 3, H, W) tensor (3-channel duplicate for pretrained backbone).
    """
    windowed = apply_brain_window(pixel_array)
    resized = cv2.resize(windowed, target_size, interpolation=cv2.INTER_AREA)

    transform = transforms.Compose([
        transforms.ToTensor(),               # (1, H, W) float [0,1]
        transforms.Normalize([0.485], [0.229]),  # single-channel approx ImageNet
    ])
    tensor = transform(resized)  # (1, H, W)
    tensor = tensor.repeat(3, 1, 1)  # (3, H, W) — duplicate for RGB backbone
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def preprocess_for_segmentation(
    pixel_array: np.ndarray,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> torch.Tensor:
    """
    Pre-process a 2-D CT slice for U-Net segmentation.

    Returns a (1, 1, H, W) tensor.
    """
    windowed = apply_brain_window(pixel_array)
    resized = cv2.resize(windowed, target_size, interpolation=cv2.INTER_AREA)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tensor = transform(resized)  # (1, H, W)
    return tensor.unsqueeze(0)  # (1, 1, H, W)


def create_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> np.ndarray:
    """
    Overlay a binary mask on the original CT image.

    Parameters
    ----------
    original : 2-D grayscale array
    mask : 2-D binary mask (0 or 1)
    color : BGR highlight colour
    alpha : overlay transparency

    Returns
    -------
    np.ndarray — BGR overlay image (H, W, 3)
    """
    windowed = apply_brain_window(original)
    resized = cv2.resize(windowed, target_size, interpolation=cv2.INTER_AREA)
    base_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    mask_resized = cv2.resize(
        mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST
    )

    highlight = np.zeros_like(base_bgr)
    highlight[mask_resized > 0] = color

    overlay = cv2.addWeighted(base_bgr, 1 - alpha, highlight, alpha, 0)
    return overlay
