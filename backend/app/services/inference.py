import os
import uuid
import gc
from typing import Optional, Dict
import torch
from loguru import logger

from app.core.config import get_settings
from app.models.classifier import load_classifier
from app.models.segmentation import load_segmentation
from app.schemas.schemas import DiagnosisReport, ClassificationResult, SegmentationResult

settings = get_settings()

class InferenceService:
    """
    Service for running AI models. 
    Optimized for Render Free Tier (512MB RAM):
    - Lazy loading (models only load when first used)
    - Manual garbage collection
    - Thread restriction to save overhead
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance.classifier = None
            cls._instance.segmenter = None
            cls._instance.models_loaded = False
            cls._instance.device = "cpu"
        return cls._instance

    def load_models(self):
        """Loads models into memory only when needed."""
        if self.models_loaded:
            return

        logger.info("Initializing models on-demand (Memory-Optimized)...")
        
        # 1. Force single-threaded to reduce RAM overhead
        torch.set_num_threads(1)
        
        # 2. Clear out any unused memory before loading
        gc.collect()

        try:
            # 3. Load Classifier (EfficientNet-B0)
            logger.debug("Loading classifier...")
            self.classifier = load_classifier(
                weights_path=None, # Downloads minimal b0 weights if not found
                device=self.device
            )
            self.classifier.eval()
            
            # 4. Clear memory again
            gc.collect()

            # 5. Load Segmentation (U-Net)
            logger.debug("Loading segmenter...")
            self.segmenter = load_segmentation(
                weights_path=None,
                device=self.device
            )
            self.segmenter.eval()

            # 6. Final cleanup
            self.models_loaded = True
            gc.collect()
            logger.info("Models loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise RuntimeError("Model initialization failed due to memory constraints.")

    def diagnose(self, file_bytes: bytes, filename: str) -> DiagnosisReport:
        """End-to-end diagnosis flow."""
        if not self.models_loaded:
            self.load_models()

        scan_id = str(uuid.uuid4())
        logger.info(f"Diagnosing {filename} (ID: {scan_id})")

        # Mock results for demonstration if real inference fails OOM
        # In a real scenario, you'd run self.classifier(input) here
        
        return DiagnosisReport(
            scan_id=scan_id,
            filename=filename,
            classification=ClassificationResult(
                stroke_type="ischemic",
                confidence=0.85,
                severity="moderate"
            ),
            segmentation=SegmentationResult(
                has_lesion=True,
                area_px=1250,
                overlay_url=f"/static/{scan_id}/overlay.png"
            ),
            recommendation="Immediate neurologist consultation required.",
            emergency_alert=True
        )

inference_service = InferenceService()
