"""
Stroke Detection System — Diagnosis API Routes

Endpoints for stroke detection, classification, segmentation, and full diagnosis.
"""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from loguru import logger

from app.core.config import get_settings
from app.schemas.schemas import (
    ClassificationResult,
    SegmentationResult,
    DiagnosisReport,
)
from app.services.inference import inference_service

settings = get_settings()
router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


@router.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Classify stroke type from CT scan",
)
async def classify_scan(
    file: UploadFile = File(..., description="CT scan image file"),
):
    """
    Upload a CT scan and receive stroke classification results.

    Returns multi-label probabilities for hemorrhage sub-types and ischemic stroke,
    along with an overall severity assessment.
    """
    if not inference_service.models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet. Please try again later.",
        )

    file_bytes = await file.read()
    pixel_array = inference_service._read_scan(file_bytes, file.filename or "scan.png")
    result = inference_service.run_classification(pixel_array, scan_id="temp")
    return result


@router.post(
    "/segment",
    response_model=SegmentationResult,
    summary="Segment lesion regions in CT scan",
)
async def segment_scan(
    file: UploadFile = File(..., description="CT scan image file"),
):
    """
    Upload a CT scan and receive lesion segmentation results.

    Returns paths to the binary mask and overlay images,
    plus the percentage of affected brain area.
    """
    if not inference_service.models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet. Please try again later.",
        )

    file_bytes = await file.read()
    pixel_array = inference_service._read_scan(file_bytes, file.filename or "scan.png")
    result = inference_service.run_segmentation(pixel_array, scan_id="temp")
    return result


@router.post(
    "/full",
    response_model=DiagnosisReport,
    summary="Full stroke diagnosis (classify + segment + report)",
)
async def full_diagnosis(
    file: UploadFile = File(..., description="CT scan image file"),
):
    """
    Run the complete diagnostic pipeline on a single CT scan:
    classification → segmentation → severity assessment → recommendation.

    Returns a comprehensive DiagnosisReport including emergency alerts.
    """
    if not inference_service.models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet. Please try again later.",
        )

    file_bytes = await file.read()
    report = inference_service.diagnose(file_bytes, file.filename or "scan.png")
    return report
