import os
import uuid
import time
from typing import Optional, Dict
from loguru import logger

from app.core.config import get_settings
from app.schemas.schemas import DiagnosisReport, ClassificationResult, SegmentationResult

settings = get_settings()

class InferenceService:
    """
    ULTRA-LIGHT DEMO SERVICE.
    This version removes all heavy AI loading to ensure it works on 
    Render's Free Tier (512MB RAM) without crashing.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance.models_loaded = True # Always True in Demo Mode
        return cls._instance

    def load_models(self):
        """No-op for Demo Mode."""
        logger.info("Demo Mode: Skipping heavy model loading to save memory.")
        pass

    def _read_scan(self, file_bytes: bytes, filename: str):
        """Stub — in production this would decode DICOM / image pixels."""
        return file_bytes  # just pass through for demo

    def run_classification(self, pixel_array, scan_id: str) -> ClassificationResult:
        """Demo classification."""
        time.sleep(0.5)
        is_ischemic = True  # default demo
        return ClassificationResult(
            scan_id=scan_id,
            stroke_type="ischemic" if is_ischemic else "hemorrhagic",
            subtype_probabilities={
                "epidural": 0.03,
                "intraparenchymal": 0.05,
                "subarachnoid": 0.02,
                "intraventricular": 0.01,
                "subdural": 0.04,
                "ischemic": 0.92,
            },
            confidence=0.92,
            severity="low",
            inference_time_ms=120.5,
        )

    def run_segmentation(self, pixel_array, scan_id: str) -> SegmentationResult:
        """Demo segmentation."""
        time.sleep(0.5)
        return SegmentationResult(
            scan_id=scan_id,
            mask_url="/static/demo_mask.png",
            overlay_url="/static/demo_overlay.png",
            lesion_area_percentage=3.2,
            inference_time_ms=95.3,
        )

    def diagnose(self, file_bytes: bytes, filename: str) -> DiagnosisReport:
        """Simulated diagnosis that works every time."""
        scan_id = str(uuid.uuid4())
        logger.info(f"DEMO MODE: Processing {filename} (ID: {scan_id})")

        # Simulate a small delay for 'realism'
        time.sleep(1.5)
        
        # Determine a result based on filename or just random
        is_ischemic = "isch" in filename.lower()
        
        classification = ClassificationResult(
            scan_id=scan_id,
            stroke_type="ischemic" if is_ischemic else "hemorrhagic",
            subtype_probabilities={
                "epidural": 0.03 if is_ischemic else 0.85,
                "intraparenchymal": 0.05 if is_ischemic else 0.72,
                "subarachnoid": 0.02 if is_ischemic else 0.45,
                "intraventricular": 0.01 if is_ischemic else 0.30,
                "subdural": 0.04 if is_ischemic else 0.60,
                "ischemic": 0.92 if is_ischemic else 0.08,
            },
            confidence=0.92,
            severity="low" if is_ischemic else "high",
            inference_time_ms=142.7,
        )

        segmentation = SegmentationResult(
            scan_id=scan_id,
            mask_url="/static/demo_mask.png",
            overlay_url="/static/demo_overlay.png",
            lesion_area_percentage=3.2 if is_ischemic else 8.7,
            inference_time_ms=98.4,
        )

        return DiagnosisReport(
            scan_id=scan_id,
            classification=classification,
            segmentation=segmentation,
            recommendation="Simulation complete. In a production environment, this would be the AI output.",
            emergency_alert=not is_ischemic,
        )

inference_service = InferenceService()

