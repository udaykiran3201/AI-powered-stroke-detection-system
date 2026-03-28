"""
Stroke Detection System — Upload API Routes

Endpoints for uploading CT brain scans (DICOM, PNG, JPEG).
"""

import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from loguru import logger

from app.core.config import get_settings
from app.schemas.schemas import UploadResponse

settings = get_settings()
router = APIRouter(prefix="/upload", tags=["Upload"])

ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@router.post(
    "/",
    response_model=UploadResponse,
    summary="Upload a CT brain scan",
    status_code=status.HTTP_201_CREATED,
)
async def upload_scan(file: UploadFile = File(..., description="CT scan image file")):
    """
    Upload a CT brain scan for analysis.

    Accepted formats: DICOM (.dcm), PNG, JPEG, TIFF.
    Maximum file size is configured via `MAX_UPLOAD_SIZE_MB`.
    """
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read file bytes
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Validate size
    if file_size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB",
        )

    # Save locally
    scan_id = str(uuid.uuid4())
    scan_dir = os.path.join(settings.upload_dir, scan_id)
    os.makedirs(scan_dir, exist_ok=True)

    save_path = os.path.join(scan_dir, f"original{ext}")
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    logger.info(
        f"Scan uploaded — id={scan_id}, file={file.filename}, size={file_size}"
    )

    return UploadResponse(
        scan_id=scan_id,
        filename=file.filename or "unknown",
        file_size_bytes=file_size,
        upload_time=datetime.now(timezone.utc),
    )
