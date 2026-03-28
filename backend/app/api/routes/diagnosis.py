"""
Stroke Detection System — Diagnosis API Routes

Endpoints for stroke detection, classification, segmentation, and full diagnosis.
"""

from fastapi import APIRouter, UploadFile, File
from app.schemas.schemas import (
    ClassificationResult,
    SegmentationResult,
    DiagnosisReport,
)
from app.services.inference import inference_service

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
    """
    file_bytes = await file.read()
    pixel_array = inference_service._read_scan(file_bytes, file.filename or "scan.png")
    result = inference_service.run_classification(
        pixel_array, scan_id="temp", filename=file.filename or "scan.png"
    )
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
    """
    file_bytes = await file.read()
    pixel_array = inference_service._read_scan(file_bytes, file.filename or "scan.png")
    # For standalone segmentation, we run a quick classification first 
    # to determine if there's a lesion to segment.
    diag = inference_service.diagnose(file_bytes, file.filename or "scan.png")
    return diag.segmentation


@router.post(
    "/full",
    response_model=DiagnosisReport,
    summary="Full stroke diagnosis (classify + segment + report)",
)
async def full_diagnosis(
    file: UploadFile = File(..., description="CT scan image file"),
):
    """
    Run the complete diagnostic pipeline on a single CT scan.
    """
    file_bytes = await file.read()
    report = inference_service.diagnose(file_bytes, file.filename or "scan.png")
    return report
