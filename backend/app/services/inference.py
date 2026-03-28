"""
Stroke Detection System — Inference Service

Orchestrates the full diagnostic pipeline:
  upload → preprocess → classify → segment → report
"""

import os
import time
import uuid
import numpy as np
import cv2
import torch
from typing import Optional, Dict
from loguru import logger

from app.core.config import get_settings
from app.models.classifier import (
    StrokeClassifier,
    load_classifier,
    predict as classify_predict,
    STROKE_CLASSES,
)
from app.models.segmentation import UNet, load_unet, segment as unet_segment
from app.services.preprocessing import (
    read_dicom,
    read_standard_image,
    preprocess_for_classification,
    preprocess_for_segmentation,
    create_overlay,
)
from app.schemas.schemas import (
    StrokeType,
    SeverityLevel,
    ClassificationResult,
    SegmentationResult,
    DiagnosisReport,
)

settings = get_settings()


class InferenceService:
    """Singleton-style service that holds loaded models in memory."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier: Optional[StrokeClassifier] = None
        self.segmentor: Optional[UNet] = None
        self._models_loaded = False
        logger.info(f"InferenceService created — device={self.device}")

    # ── Model lifecycle ─────────────────────
    def load_models(self) -> None:
        """Load both models into memory."""
        logger.info("Loading classification model …")
        self.classifier = load_classifier(
            weights_path=settings.classification_model_path,
            device=self.device,
        )
        logger.info("Loading segmentation model …")
        self.segmentor = load_unet(
            weights_path=settings.segmentation_model_path,
            device=self.device,
        )
        self._models_loaded = True
        logger.info("All models loaded ✓")

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    # ── File I/O ────────────────────────────
    @staticmethod
    def _read_scan(file_bytes: bytes, filename: str) -> np.ndarray:
        """Read CT scan bytes into a 2-D numpy array."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".dcm", ".dicom"):
            return read_dicom(file_bytes)
        else:
            return read_standard_image(file_bytes)

    @staticmethod
    def _save_image(image: np.ndarray, directory: str, name: str) -> str:
        """Save an image to disk and return the relative path."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        cv2.imwrite(path, image)
        return path

    # ── Classification ──────────────────────
    def run_classification(
        self, pixel_array: np.ndarray, scan_id: str
    ) -> ClassificationResult:
        """Classify a CT scan slice."""
        t0 = time.perf_counter()
        tensor = preprocess_for_classification(pixel_array)
        probs = classify_predict(self.classifier, tensor, device=self.device)
        elapsed = (time.perf_counter() - t0) * 1000

        # Determine dominant stroke type
        max_class = max(probs, key=probs.get)
        max_prob = probs[max_class]

        if max_prob < 0.3:
            stroke_type = StrokeType.NONE
        elif max_class == "ischemic":
            stroke_type = StrokeType.ISCHEMIC
        else:
            stroke_type = StrokeType.HEMORRHAGIC

        severity = self._compute_severity(max_prob, stroke_type)

        return ClassificationResult(
            scan_id=scan_id,
            stroke_type=stroke_type,
            subtype_probabilities=probs,
            confidence=max_prob,
            severity=severity,
            inference_time_ms=round(elapsed, 2),
        )

    # ── Segmentation ────────────────────────
    def run_segmentation(
        self, pixel_array: np.ndarray, scan_id: str
    ) -> SegmentationResult:
        """Segment lesion region and create overlay."""
        t0 = time.perf_counter()
        tensor = preprocess_for_segmentation(pixel_array)
        mask_tensor = unet_segment(self.segmentor, tensor, device=self.device)
        elapsed = (time.perf_counter() - t0) * 1000

        mask_np = mask_tensor.squeeze().numpy()
        total_pixels = mask_np.size
        lesion_pixels = int(mask_np.sum())
        lesion_pct = round((lesion_pixels / total_pixels) * 100, 2)

        # Save mask & overlay
        output_dir = os.path.join(settings.upload_dir, scan_id)
        mask_path = self._save_image(
            mask_np * 255, output_dir, "segmentation_mask.png"
        )
        overlay = create_overlay(pixel_array, mask_np)
        overlay_path = self._save_image(overlay, output_dir, "overlay.png")

        return SegmentationResult(
            scan_id=scan_id,
            mask_url=f"/static/{scan_id}/segmentation_mask.png",
            overlay_url=f"/static/{scan_id}/overlay.png",
            lesion_area_percentage=lesion_pct,
            inference_time_ms=round(elapsed, 2),
        )

    # ── Full pipeline ───────────────────────
    def diagnose(self, file_bytes: bytes, filename: str) -> DiagnosisReport:
        """End-to-end diagnosis: classify + segment + report."""
        if not self.models_loaded:
            logger.info("On-demand model loading triggered...")
            self.load_models()

        scan_id = str(uuid.uuid4())
        logger.info(f"Starting diagnosis — scan_id={scan_id}, file={filename}")

        pixel_array = self._read_scan(file_bytes, filename)

        classification = self.run_classification(pixel_array, scan_id)
        segmentation = self.run_segmentation(pixel_array, scan_id)

        emergency = classification.severity in (
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
        )
        recommendation = self._generate_recommendation(classification, segmentation)

        report = DiagnosisReport(
            scan_id=scan_id,
            classification=classification,
            segmentation=segmentation,
            emergency_alert=emergency,
            recommendation=recommendation,
        )
        logger.info(
            f"Diagnosis complete — scan_id={scan_id}, "
            f"type={classification.stroke_type}, severity={classification.severity}"
        )
        return report

    # ── Helpers ──────────────────────────────
    @staticmethod
    def _compute_severity(confidence: float, stroke_type: StrokeType) -> SeverityLevel:
        if stroke_type == StrokeType.NONE:
            return SeverityLevel.NORMAL
        if confidence >= 0.9:
            return SeverityLevel.CRITICAL
        if confidence >= 0.75:
            return SeverityLevel.HIGH
        if confidence >= 0.5:
            return SeverityLevel.MODERATE
        return SeverityLevel.LOW

    @staticmethod
    def _generate_recommendation(
        classification: ClassificationResult,
        segmentation: SegmentationResult,
    ) -> str:
        if classification.stroke_type == StrokeType.NONE:
            return (
                "No signs of acute stroke detected. "
                "Clinical correlation recommended."
            )

        parts = [
            f"Detected {classification.stroke_type.value} stroke "
            f"with {classification.confidence:.0%} confidence."
        ]
        if classification.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
            parts.append(
                "⚠️ EMERGENCY — Immediate neurology consultation required."
            )
        if segmentation.lesion_area_percentage > 5:
            parts.append(
                f"Large lesion area ({segmentation.lesion_area_percentage:.1f}% of brain). "
                "Consider urgent intervention."
            )
        parts.append(
            "Recommend confirmatory MRI and continuous monitoring."
        )
        return " ".join(parts)


# ── Module-level singleton ──────────────────
inference_service = InferenceService()
