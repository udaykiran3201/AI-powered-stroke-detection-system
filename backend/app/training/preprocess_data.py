"""
Stroke Detection System — Data Preprocessing Utilities

Converts raw DICOM / image datasets into a standardised format
ready for model training.

Usage:
    python -m app.training.preprocess_data \
        --input_dir data/raw \
        --output_dir data/processed
"""

import argparse
import os
import csv
import numpy as np
import cv2
from PIL import Image
from loguru import logger

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from app.services.preprocessing import apply_brain_window


TARGET_SIZE = (256, 256)


def convert_dicom_to_png(dicom_path: str, output_path: str) -> bool:
    """Convert a single DICOM file to a windowed PNG."""
    if not HAS_PYDICOM:
        logger.error("pydicom required for DICOM conversion")
        return False
    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept

        windowed = apply_brain_window(pixel_array)
        resized = cv2.resize(windowed, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized)
        return True
    except Exception as e:
        logger.warning(f"Failed to convert {dicom_path}: {e}")
        return False


def process_image(image_path: str, output_path: str) -> bool:
    """Resize a standard image to target size."""
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize(TARGET_SIZE, Image.BILINEAR)
        img.save(output_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to process {image_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw CT data")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    for split in ["train", "val"]:
        input_images = os.path.join(args.input_dir, split, "images")
        output_images = os.path.join(args.output_dir, split, "images")
        os.makedirs(output_images, exist_ok=True)

        if not os.path.isdir(input_images):
            logger.warning(f"Skipping {split} — directory not found: {input_images}")
            continue

        count = 0
        for fname in sorted(os.listdir(input_images)):
            src = os.path.join(input_images, fname)
            ext = os.path.splitext(fname)[1].lower()
            dst_name = os.path.splitext(fname)[0] + ".png"
            dst = os.path.join(output_images, dst_name)

            if ext in (".dcm", ".dicom"):
                ok = convert_dicom_to_png(src, dst)
            elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                ok = process_image(src, dst)
            else:
                continue

            if ok:
                count += 1

        logger.info(f"[{split}] Processed {count} images → {output_images}")

        # Copy labels if present
        labels_src = os.path.join(args.input_dir, split, "labels.csv")
        labels_dst = os.path.join(args.output_dir, split, "labels.csv")
        if os.path.exists(labels_src):
            import shutil
            shutil.copy2(labels_src, labels_dst)
            logger.info(f"[{split}] Copied labels.csv")

    logger.info("Preprocessing complete ✓")


if __name__ == "__main__":
    main()
