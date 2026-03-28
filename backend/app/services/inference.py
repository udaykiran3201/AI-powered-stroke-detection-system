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

    def diagnose(self, file_bytes: bytes, filename: str) -> DiagnosisReport:
        """Simulated diagnosis that works every time."""
        scan_id = str(uuid.uuid4())
        logger.info(f"DEMO MODE: Processing {filename} (ID: {scan_id})")

        # Simulate a small delay for 'realism'
        time.sleep(1.5)
        
        # Determine a result based on filename or just random
        is_ischemic = "isch" in filename.lower()
        
        return DiagnosisReport(
            scan_id=scan_id,
            filename=filename,
            classification=ClassificationResult(
                stroke_type="ischemic" if is_ischemic else "hemorrhagic",
                confidence=0.92,
                severity="low"
            ),
            segmentation=SegmentationResult(
                has_lesion=True,
                area_px=840,
                overlay_url=f"/static/demo_overlay.png" # We will ensure a demo file exists later
            ),
            recommendation="Simulation complete. In a production environment, this would be the AI output.",
            emergency_alert=not is_ischemic
        )

inference_service = InferenceService()
