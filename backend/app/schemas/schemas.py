"""
Stroke Detection System — Pydantic Schemas

Request / response models used across the API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timezone
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────
class StrokeType(str, Enum):
    HEMORRHAGIC = "hemorrhagic"
    ISCHEMIC = "ischemic"
    NONE = "none"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NORMAL = "normal"


# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────
class UploadResponse(BaseModel):
    scan_id: str = Field(..., description="Unique identifier for the uploaded scan")
    filename: str
    file_size_bytes: int
    upload_time: datetime
    message: str = "CT scan uploaded successfully"


# ──────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────
class ClassificationResult(BaseModel):
    scan_id: str
    stroke_type: StrokeType
    subtype_probabilities: Dict[str, float] = Field(
        ...,
        description="Per-class sigmoid probabilities",
        json_schema_extra={"example": {"epidural": 0.05, "ischemic": 0.92}},
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Max probability across all classes"
    )
    severity: SeverityLevel
    inference_time_ms: float


# ──────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────
class SegmentationResult(BaseModel):
    scan_id: str
    mask_url: str = Field(
        ..., description="URL / path to the generated segmentation mask image"
    )
    overlay_url: str = Field(
        ..., description="URL / path to the overlay (scan + highlighted region)"
    )
    lesion_area_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of brain area affected"
    )
    inference_time_ms: float


# ──────────────────────────────────────────────
# Full Diagnosis Report
# ──────────────────────────────────────────────
class DiagnosisReport(BaseModel):
    scan_id: str
    classification: ClassificationResult
    segmentation: SegmentationResult
    emergency_alert: bool = Field(
        ..., description="True when severity is CRITICAL or HIGH"
    )
    recommendation: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    models_loaded: bool
    environment: str
